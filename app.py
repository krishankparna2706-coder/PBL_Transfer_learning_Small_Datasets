"""
Basic Inference API for the Transfer Learning MVP.
REST endpoint: POST /predict with an image file returns the predicted class.
"""

import json
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from config import DEFAULT_MODEL_SAVE_PATH, CLASS_NAMES_PATH, API_HOST, API_PORT
from data_utils import get_inference_transform
from model_utils import create_feature_extractor

# -----------------------------------------------------------------------------
# App and global model (loaded once at startup)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Transfer Learning MVP - Inference API",
    description="Upload an image to get a classification prediction (e.g., ants vs bees).",
    version="1.0.0",
)

# Allow requests from GitHub Pages and same-origin (for live demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production you can restrict to your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_model = None
_class_names = None
_device = None
_inference_transform = None


def load_model_for_api(model_path=None, class_names_path=None):
    """Load model and class names at startup."""
    global _model, _class_names, _device, _inference_transform

    model_path = model_path or DEFAULT_MODEL_SAVE_PATH
    class_names_path = class_names_path or CLASS_NAMES_PATH

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Train the model first with: python main.py"
        )
    if not os.path.isfile(class_names_path):
        raise FileNotFoundError(
            f"Class names not found at {class_names_path}. Train the model first with: python main.py"
        )

    with open(class_names_path) as f:
        _class_names = json.load(f)
    num_classes = len(_class_names)

    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _model = create_feature_extractor(num_classes)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model = _model.to(_device)
    _model.eval()
    _inference_transform = get_inference_transform()


@app.on_event("startup")
def startup_event():
    """Load model when the API starts (if files exist)."""
    try:
        load_model_for_api()
    except FileNotFoundError as e:
        # Allow server to start without a model for health checks; predict will return 503
        pass


@app.get("/")
def root():
    return {
        "message": "Transfer Learning MVP Inference API",
        "docs": "/docs",
        "predict": "POST /predict with multipart form 'file' = image",
    }


@app.get("/health")
def health():
    """Health check; indicates whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return the predicted class and confidence.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first with: python main.py",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Expected an image file (e.g., image/jpeg, image/png).",
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    input_tensor = _inference_transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = _model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    pred_class = _class_names[pred_idx.item()]
    prob_list = probs.cpu().numpy()[0].tolist()
    prob_dict = {c: round(prob_list[i], 4) for i, c in enumerate(_class_names)}

    return JSONResponse(
        content={
            "prediction": pred_class,
            "confidence": round(confidence.item(), 4),
            "probabilities": prob_dict,
        }
    )


def run_api(host=None, port=None):
    """Run the API server (for use from command line)."""
    import uvicorn
    uvicorn.run(
        "app:app",
        host=host or API_HOST,
        port=port or API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run_api()
