import os
os.environ["KERAS_BACKEND"] = "torch"
import csv
import io
import json
import time
import zipfile
from collections import defaultdict, deque
import numpy as np
import keras
import torch
from PIL import Image
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware


@keras.saving.register_keras_serializable()
class LocalResponseNormalization(keras.layers.Layer):

    def call(self, x):

        original_dtype = x.dtype
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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

keras.mixed_precision.set_global_policy("mixed_float16")
keras.config.set_image_data_format("channels_first")

print("Device:", device)


MAX_REQUESTS = 10
RATE_WINDOW = 60 * 60  # 1 hour

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096

REQUEST_LOG = defaultdict(deque)


def get_client_ip(request: Request) -> str:
    """
    Get the client IP.

    Cloud Run places the original client address in X-Forwarded-For.
    For local development, fall back to request.client.host.
    """

    forwarded_for = request.headers.get(
        "x-forwarded-for"
    )

    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def check_rate_limit(ip: str) -> bool:
    """
    Allow at most MAX_REQUESTS requests from an IP
    during RATE_WINDOW seconds.
    """

    now = time.monotonic()

    timestamps = REQUEST_LOG[ip]

    # Remove timestamps outside the current window.
    while timestamps:

        if now - timestamps[0] >= RATE_WINDOW:
            timestamps.popleft()
        else:
            break

    # Limit exceeded.
    if len(timestamps) >= MAX_REQUESTS:
        return False

    timestamps.append(now)

    return True


def load_int8_model(
    arch_path: str,
    zip_path: str,
):

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

    with np.load(
        io.BytesIO(npz_bytes),
        allow_pickle=False,
    ) as weights:

        for layer_index, layer in enumerate(
            model.layers
        ):

            original_weights = layer.get_weights()

            # Skip layers without trainable/non-trainable weights.
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

                # Dequantize to FP32
                float_weight = (
                    int8_weight.astype(
                        np.float32
                    )
                    * scale
                )

                new_weights.append(
                    float_weight
                )

            # Load reconstructed weights.
            layer.set_weights(
                new_weights
            )

    return model

MODEL = load_int8_model(
    "../model/alexnet_architecture.json",
    "../model/alexnet_int8_weights.zip",
)

CLASS_MAP_FILE = (
    "../classes/IDX_WNID_CLASS.csv"
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

def preprocess(
    image_bytes: bytes,
) -> torch.Tensor:

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

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

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image /= 255.0

    x = torch.from_numpy(
        image
    ).unsqueeze(0)

    return x.to(device)


app = FastAPI(
    title="PaperClip AlexNet API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://prashanthprabhu24.github.io",
    ],
    allow_origin_regex=r"^https?://localhost(?::\d+)?$",
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
    ],
)


@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(device),
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
):


    client_ip = get_client_ip(
        request
    )

    if not check_rate_limit(
        client_ip
    ):

        raise HTTPException(
            status_code=429,
            detail=(
                "Rate limit exceeded. "
                "Maximum 10 predictions per hour."
            ),
            headers={
                "Retry-After": str(
                    RATE_WINDOW
                )
            },
        )


    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image type",
        )

    if (
        file.size is not None
        and file.size > MAX_FILE_SIZE
    ):

        raise HTTPException(
            status_code=413,
            detail=(
                "Image too large. "
                "Maximum size is 5 MB."
            ),
        )

    try:

        image_bytes = await file.read(
            MAX_FILE_SIZE + 1
        )

        if len(image_bytes) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Image too large. "
                    "Maximum size is 5 MB."
                ),
            )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image upload failed: {e}"
            ),
        )

    try:

        # Open once for validation.
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Check source image dimensions.
        if (
            image.width > MAX_IMAGE_WIDTH
            or image.height > MAX_IMAGE_HEIGHT
        ):

            raise HTTPException(
                status_code=413,
                detail=(
                    "Image dimensions too large. "
                    "Maximum is 4096 x 4096 pixels."
                ),
            )

        # Actual preprocessing.
        x = preprocess(
            image_bytes
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image processing failed: {e}"
            ),
        )

    try:

        with torch.inference_mode():

            predictions = MODEL(x)

            # Convert mixed-float output to FP32
            predictions = predictions.float()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference failed: {e}"
            ),
        )

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

    return {
        "predictions": results
    }