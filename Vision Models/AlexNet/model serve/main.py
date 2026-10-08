import os
os.environ["KERAS_BACKEND"] = "torch"   # MUST be before keras import

import io
import json
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import keras
import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Custom layer (matches notebook exactly)
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class LocalResponseNormalization(keras.layers.Layer):
    def call(self, x):
        return torch.nn.functional.local_response_norm(
            x, size=5, alpha=1e-4, beta=0.75, k=2.0
        )


# ---------------------------------------------------------------------------
# Keras global config (matches notebook)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

keras.mixed_precision.set_global_policy("mixed_float16")
keras.config.set_image_data_format("channels_first")

print("Device:", device)


# ---------------------------------------------------------------------------
# Model loading (matches notebook exactly)
# ---------------------------------------------------------------------------
def load_int8_model(arch_path, zip_path):
    with open(arch_path, "r") as f:
        model = keras.models.model_from_json(
            f.read(),
            custom_objects={"LocalResponseNormalization": LocalResponseNormalization}
        )

    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        npz_path = os.path.join(temp_dir, "alexnet_int8_weights.npz")
        meta_path = os.path.join(temp_dir, "alexnet_int8_metadata.json")

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        with np.load(npz_path) as data:
            weight_arrays = {k: data[k].copy() for k in data.files}

        for i, layer in enumerate(model.layers):
            original_weights = layer.get_weights()
            if not original_weights:
                continue
            new_weights = []
            for j in range(len(original_weights)):
                key = f"layer_{i}_weight_{j}"
                int8_w = weight_arrays[key]
                scale = metadata[key]["scale"]
                float_w = int8_w.astype(np.float32) * scale
                new_weights.append(float_w)
            layer.set_weights(new_weights)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return model


# ---------------------------------------------------------------------------
# Load model and class map at startup
# ---------------------------------------------------------------------------
MODEL = load_int8_model(
    "../model/alexnet_architecture.json",
    "../model/alexnet_int8_weights.zip"
)

CLASS_MAP_FILE = "../../../Datasets/ILSVRC2010_images/IDX_WNID_CLASS.csv"
class_map = pd.read_csv(CLASS_MAP_FILE)
idx_to_wnid = dict(zip(class_map["idx"], class_map["wnid"]))
wnid_to_class = dict(zip(class_map["wnid"], class_map["class_name"]))

def get_class_name(target_id: int) -> str:
    return wnid_to_class[idx_to_wnid[target_id]]


# ---------------------------------------------------------------------------
# Preprocessing (matches notebook exactly — torchvision transforms)
# ---------------------------------------------------------------------------
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = PREPROCESS(img)
    x = x.unsqueeze(0).to(device)
    return x


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="PaperClip AlexNet API")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image type")

    try:
        image_bytes = await file.read()
        x = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

    try:
        with torch.inference_mode():
            predictions = MODEL(x)
            predictions = predictions.float()
            predictions = predictions.cpu().numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    probs = predictions[0]
    top5_idx = np.argsort(probs)[-5:][::-1]
    results = [
        {
            "index": int(idx),
            "label": get_class_name(int(idx)),
            "confidence": float(probs[idx]),
        }
        for idx in top5_idx
    ]
    return JSONResponse({"predictions": results})