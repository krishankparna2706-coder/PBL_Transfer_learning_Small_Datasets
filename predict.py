"""
Command-line inference for the Transfer Learning MVP.
Load a trained model and predict the class for a given image path.
"""

import argparse
import json
import os

import torch
from PIL import Image

from config import DEFAULT_MODEL_SAVE_PATH, CLASS_NAMES_PATH
from data_utils import get_inference_transform
from model_utils import create_feature_extractor


def load_model_and_classes(model_path=None, class_names_path=None, device=None):
    """Load saved state dict and class names; return model and class_names list."""
    model_path = model_path or DEFAULT_MODEL_SAVE_PATH
    class_names_path = class_names_path or CLASS_NAMES_PATH

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Run main.py to train first."
        )
    if not os.path.isfile(class_names_path):
        raise FileNotFoundError(
            f"Class names file not found at {class_names_path}. Run main.py to train first."
        )

    with open(class_names_path) as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_feature_extractor(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(model, image_path, class_names, device, transform=None):
    """
    Run inference on a single image file.
    Returns predicted class name and optional confidence/probs.
    """
    if transform is None:
        transform = get_inference_transform()

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    pred_class = class_names[pred_idx.item()]
    return pred_class, confidence.item(), probs.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Predict image class using trained model")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_SAVE_PATH,
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=CLASS_NAMES_PATH,
        help="Path to class_names.json",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return

    model, class_names, device = load_model_and_classes(
        model_path=args.model, class_names_path=args.class_names
    )
    pred_class, confidence, probs = predict_image(
        model, args.image_path, class_names, device
    )

    print(f"Prediction: {pred_class} (confidence: {confidence:.4f})")
    for i, c in enumerate(class_names):
        print(f"  {c}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
