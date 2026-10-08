import os
os.environ["KERAS_BACKEND"] = "torch"
import ast
import csv
import io
import json
import math
import zipfile
import keras
import torch
import torch.nn.functional as F
import torchvision
from fastapi import FastAPI, File, UploadFile, HTTPException

@keras.saving.register_keras_serializable()
class LocalResponseNormalization(keras.layers.Layer):
    def call(self, x):
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
# Minimal NPY reader
#
# Reads int8 .npy data without NumPy.
# Your .npz file is simply a ZIP containing .npy files.
# ============================================================================

def read_npy_int8(raw: bytes) -> torch.Tensor:

    # NumPy magic header
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("Invalid NPY file")

    major = raw[6]
    minor = raw[7]

    # ------------------------------------------------------------------------
    # Header size
    # ------------------------------------------------------------------------

    if major == 1:
        header_len = int.from_bytes(
            raw[8:10],
            byteorder="little",
        )
        header_start = 10

    elif major in (2, 3):
        header_len = int.from_bytes(
            raw[8:12],
            byteorder="little",
        )
        header_start = 12

    else:
        raise ValueError(
            f"Unsupported NPY version: {major}.{minor}"
        )

    header_end = header_start + header_len

    # NumPy header is a Python literal dictionary
    header = raw[
        header_start:header_end
    ].decode("latin1")

    metadata = ast.literal_eval(header)

    dtype = metadata["descr"]
    fortran_order = metadata["fortran_order"]
    shape = metadata["shape"]

    # ------------------------------------------------------------------------
    # We expect int8 weights
    # ------------------------------------------------------------------------

    if dtype not in {
        "|i1",
        "<i1",
        ">i1",
        "i1",
    }:
        raise ValueError(
            f"Expected int8 NPY data, got {dtype}"
        )

    if fortran_order:
        raise ValueError(
            "Fortran-ordered NPY arrays are not supported."
        )

    if isinstance(shape, int):
        shape = (shape,)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------

    data_start = header_end

    num_elements = math.prod(shape)

    data = memoryview(raw)[
        data_start:data_start + num_elements
    ]

    if len(data) < num_elements:
        raise ValueError(
            "NPY payload is smaller than expected."
        )

    tensor = torch.frombuffer(
        data,
        dtype=torch.int8,
    ).clone()

    return tensor.reshape(shape)


# ============================================================================
# Load INT8 model
#
# No NumPy
# No temporary directory
# No extraction to disk
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
    # Open outer ZIP directly
    # ------------------------------------------------------------------------

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as outer_zip:

        metadata = json.loads(
            outer_zip.read(
                "alexnet_int8_metadata.json"
            )
        )

        # The NPZ itself is another ZIP archive.
        npz_bytes = outer_zip.read(
            "alexnet_int8_weights.npz"
        )

    # ------------------------------------------------------------------------
    # Read NPZ directly from memory
    # ------------------------------------------------------------------------

    with zipfile.ZipFile(
        io.BytesIO(npz_bytes),
        "r",
    ) as weights_zip:

        for layer_index, layer in enumerate(
            model.layers
        ):

            original_weights = layer.get_weights()

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

                npy_name = f"{key}.npy"

                # Read raw NPY from NPZ
                raw_npy = weights_zip.read(
                    npy_name
                )

                # int8 tensor
                int8_weight = read_npy_int8(
                    raw_npy
                )

                # Quantization scale
                scale = float(
                    metadata[key]["scale"]
                )

                # Dequantize
                float_weight = (
                    int8_weight.float()
                    * scale
                )

                new_weights.append(
                    float_weight
                )

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
# Pandas -> Python csv
# ============================================================================

CLASS_MAP_FILE = (
    "../../../Datasets/"
    "ILSVRC2010_images/"
    "IDX_WNID_CLASS.csv"
)

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
# torchvision.io replaces PIL
# F.interpolate replaces torchvision.transforms.Resize
# Tensor slicing replaces CenterCrop
# ============================================================================

def preprocess(
    image_bytes: bytes,
) -> torch.Tensor:

    # ------------------------------------------------------------------------
    # Raw encoded image
    # JPEG / PNG / WEBP
    # ------------------------------------------------------------------------

    encoded = torch.frombuffer(
        image_bytes,
        dtype=torch.uint8,
    )

    # ------------------------------------------------------------------------
    # Decode directly to Tensor
    #
    # Result:
    # [C, H, W]
    # ------------------------------------------------------------------------

    image = torchvision.io.decode_image(
        encoded,
        mode=torchvision.io.ImageReadMode.RGB,
    )

    # ------------------------------------------------------------------------
    # Convert:
    #
    # uint8 [0, 255]
    #      ->
    # float32 [0, 1]
    # ------------------------------------------------------------------------

    image = image.float().div(255.0)

    # ------------------------------------------------------------------------
    # Add batch dimension
    #
    # [C,H,W] -> [1,C,H,W]
    # ------------------------------------------------------------------------

    image = image.unsqueeze(0)

    # ------------------------------------------------------------------------
    # Resize shortest edge to 256
    # Equivalent to:
    #
    # transforms.Resize(256)
    # ------------------------------------------------------------------------

    _, _, h, w = image.shape

    if h < w:

        new_h = 256
        new_w = int(
            math.floor(
                w * 256 / h
            )
        )

    else:

        new_w = 256
        new_h = int(
            math.floor(
                h * 256 / w
            )
        )

    image = F.interpolate(
        image,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    # ------------------------------------------------------------------------
    # Center crop 224x224
    #
    # Equivalent to:
    #
    # transforms.CenterCrop(224)
    # ------------------------------------------------------------------------

    _, _, h, w = image.shape

    top = (h - 224) // 2
    left = (w - 224) // 2

    image = image[
        :,
        :,
        top:top + 224,
        left:left + 224,
    ]

    # ------------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------------

    return image.to(device)


# ============================================================================
# FastAPI
# ============================================================================

app = FastAPI(
    title="PaperClip AlexNet API"
)


# ============================================================================
# Health
# ============================================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(device),
    }


# ============================================================================
# Prediction
# ============================================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    # ------------------------------------------------------------------------
    # Validate content type
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

            predictions = predictions.float()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference failed: {e}"
            ),
        )

    # ------------------------------------------------------------------------
    # Top 5
    #
    # torch.topk replaces np.argsort
    # ------------------------------------------------------------------------

    probs = predictions[0]

    top5 = torch.topk(
        probs,
        k=5,
        largest=True,
        sorted=True,
    )

    top5_indices = top5.indices
    top5_values = top5.values

    results = []

    for idx, confidence in zip(
        top5_indices.tolist(),
        top5_values.tolist(),
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

    # FastAPI automatically serializes dict -> JSON
    return {
        "predictions": results
    }