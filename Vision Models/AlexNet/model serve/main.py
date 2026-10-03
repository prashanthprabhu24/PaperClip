import os

from huggingface_hub import hf_hub_download

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
keras.mixed_precision.set_global_policy("mixed_float16")
keras.config.set_image_data_format("channels_first")


@keras.saving.register_keras_serializable()
class LocalResponseNormalization(keras.layers.Layer):
    def call(self, x):
        return torch.nn.functional.local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0, )


#model = keras.models.load_model("../model/alexnet.keras")
model_path = hf_hub_download(repo_id="PrashanthDev24/alexnet", filename="alexnet.keras")
model = keras.models.load_model(model_path)

class_map = pd.read_csv("../../../Datasets/ILSVRC2010_images/IDX_WNID_CLASS.csv")

idx_to_wnid = dict(zip(class_map["idx"], class_map["wnid"]))
wnid_to_class = dict(zip(class_map["wnid"], class_map["class_name"]))


def get_class_name(idx):
    return wnid_to_class[idx_to_wnid[idx]]


preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), ])

app = FastAPI(title="AlexNet ImageNet API", description="ImageNet classification using AlexNet", version="1.0.0",)


def predict(image: Image.Image):
    image = image.convert("RGB")
    x = preprocess(image)
    x = x.unsqueeze(0)
    x = x.to(device)
    with torch.inference_mode():
        predictions = model(x)
        predictions = predictions.float()
        predictions = predictions.cpu().numpy()[0]
    top5 = np.argsort(predictions)[-5:][::-1]
    results = []
    for idx in top5:
        results.append({"class_id": int(idx), "wnid": idx_to_wnid[int(idx)], "class": get_class_name(int(idx)),
                        "probability": float(predictions[idx]), })
    return results


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        from io import BytesIO
        image = Image.open(BytesIO(contents))
        predictions = predict(image)
        return {"filename": file.filename, "top_1": predictions[0], "top_5": predictions, }

    except Exception as e:
        raise HTTPException(status_code=400,detail=f"Could not process image: {str(e)}")