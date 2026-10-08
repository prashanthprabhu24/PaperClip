import os

# MUST be set before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import csv
import io
import json
import zipfile

import numpy as np
import keras
import torch

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException


# ============================================================================
# Custom layer
# ============================================================================

@keras.saving.register_keras_serializable()
class LocalResponseNormalization(keras.layers.Layer):
    def call(self, x):

        original_dtype = x.dtype

        # PyTorch CPU does not support the FP16 kernel used internally
        # by local_response_norm().
        if (
            x.device.type == "cpu"
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            x = x.float()

            x = torch.nn.functional.local_response_norm(
                x,
                size=5,
                alpha=1e-4,
                beta=0.75,
                k=2.0,
            )

            return x.to(original_dtype)

        return torch.nn.functional.local_response_norm(
            x,
            size=5,
            alpha=1e-4,
            beta=0.75,
            k=2.0,
        )


# ============================================================================
# Global configuration
# ============================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

keras.mixed_precision.set_global_policy("mixed_float16")
keras.config.set_image_data_format("channels_first")

print("Device:", device)


# ============================================================================
# Model loading
# ============================================================================

def load_int8_model(
    arch_path: str,
    zip_path: str,
):
    # ------------------------------------------------------------------------
    # Load architecture
    # ------------------------------------------------------------------------

    with open(
        arch_path,
        "r",
        encoding="utf-8",
    ) as f:

        model = keras.models.model_from_json(
            f.read(),
            custom_objects={
                "LocalResponseNormalization":
                    LocalResponseNormalization
            },
        )

    # ------------------------------------------------------------------------
    # Read model ZIP
    #
    # alexnet_int8_weights.zip
    # ├── alexnet_int8_weights.npz
    # └── alexnet_int8_metadata.json
    # ------------------------------------------------------------------------

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as z:

        metadata = json.loads(
            z.read(
                "alexnet_int8_metadata.json"
            )
        )

        npz_bytes = z.read(
            "alexnet_int8_weights.npz"
        )

    # ------------------------------------------------------------------------
    # NumPy directly handles the NPZ
    # ------------------------------------------------------------------------

    with np.load(
        io.BytesIO(npz_bytes),
        allow_pickle=False,
    ) as weights:

        for layer_index, layer in enumerate(
            model.layers
        ):

            original_weights = layer.get_weights()

            # Layers without weights
            if not original_weights:
                continue

            new_weights = []

            for weight_index in range(
                len(original_weights)
            ):

                key = (
                    f"layer_{layer_index}"
                    f"_weight_{weight_index}"
                )

                # INT8 quantized weight
                int8_weight = weights[key]

                # Quantization scale
                scale = float(
                    metadata[key]["scale"]
                )

                # Dequantize
                float_weight = (
                    int8_weight.astype(
                        np.float32
                    )
                    * scale
                )

                new_weights.append(
                    float_weight
                )

            # Put reconstructed weights into Keras layer
            layer.set_weights(
                new_weights
            )

    return model


# ============================================================================
# Load model
# ============================================================================

MODEL = load_int8_model(
    "../model/alexnet_architecture.json",
    "../model/alexnet_int8_weights.zip",
)


# ============================================================================
# Class map
#
# No Pandas needed.
# ============================================================================

CLASS_MAP_FILE = "../classes/IDX_WNID_CLASS.csv"

idx_to_class = {}

with open(
    CLASS_MAP_FILE,
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        idx = int(row["idx"])

        idx_to_class[idx] = row["class_name"]


def get_class_name(
    target_id: int,
) -> str:

    return idx_to_class[target_id]


# ============================================================================
# Image preprocessing
#
# PIL:
#   decode
#   RGB conversion
#   resize
#   center crop
#
# NumPy:
#   HWC -> CHW
#   uint8 -> float32
#
# PyTorch:
#   NumPy -> Tensor
#   CPU -> GPU
# ============================================================================

def preprocess(
    image_bytes: bytes,
) -> torch.Tensor:

    # ------------------------------------------------------------------------
    # Decode image
    # ------------------------------------------------------------------------

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # ------------------------------------------------------------------------
    # Resize shortest side to 256
    #
    # Equivalent to:
    # torchvision.transforms.Resize(256)
    # ------------------------------------------------------------------------

    width, height = image.size

    if height < width:

        new_height = 256
        new_width = round(
            width * 256 / height
        )

    else:

        new_width = 256
        new_height = round(
            height * 256 / width
        )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    # ------------------------------------------------------------------------
    # Center crop 224x224
    #
    # Equivalent to:
    # torchvision.transforms.CenterCrop(224)
    # ------------------------------------------------------------------------

    width, height = image.size

    left = (width - 224) // 2
    top = (height - 224) // 2

    image = image.crop(
        (
            left,
            top,
            left + 224,
            top + 224,
        )
    )

    # ------------------------------------------------------------------------
    # PIL -> NumPy
    #
    # Shape:
    #   [H, W, C]
    #
    # dtype:
    #   uint8
    # ------------------------------------------------------------------------

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # HWC -> CHW
    # [H,W,C] -> [C,H,W]
    # ------------------------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    # ------------------------------------------------------------------------
    # [0,255] -> [0,1]
    # ------------------------------------------------------------------------

    image /= 255.0

    # ------------------------------------------------------------------------
    # NumPy -> PyTorch
    #
    # [C,H,W] -> [1,C,H,W]
    # ------------------------------------------------------------------------

    x = torch.from_numpy(
        image
    ).unsqueeze(0)

    # ------------------------------------------------------------------------
    # Move to inference device
    # ------------------------------------------------------------------------

    return x.to(device)


# ============================================================================
# FastAPI
# ============================================================================

app = FastAPI(
    title="PaperClip AlexNet API"
)


# ============================================================================
# Health endpoint
# ============================================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(device),
    }


# ============================================================================
# Prediction endpoint
# ============================================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    # ------------------------------------------------------------------------
    # Validate image type
    # ------------------------------------------------------------------------

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image type",
        )

    # ------------------------------------------------------------------------
    # Read + preprocess
    # ------------------------------------------------------------------------

    try:

        image_bytes = await file.read()

        x = preprocess(
            image_bytes
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image processing failed: {e}"
            ),
        )

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------

    try:

        with torch.inference_mode():

            predictions = MODEL(x)

            # Convert mixed-float output to float32
            predictions = predictions.float()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference failed: {e}"
            ),
        )

    # ------------------------------------------------------------------------
    # Top-5 predictions
    #
    # torch.topk replaces NumPy argsort
    # ------------------------------------------------------------------------

    probs = predictions[0]

    top5 = torch.topk(
        probs,
        k=5,
        largest=True,
        sorted=True,
    )

    results = []

    for idx, confidence in zip(
        top5.indices.tolist(),
        top5.values.tolist(),
    ):

        results.append(
            {
                "index": int(idx),
                "label": get_class_name(
                    int(idx)
                ),
                "confidence": float(
                    confidence
                ),
            }
        )

    # FastAPI serializes this dictionary to JSON
    return {
        "predictions": results
    }