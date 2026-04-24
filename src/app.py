"""
Gradio web demo for Diabetic Retinopathy detection.

Upload a retinal fundus image and get:
- Preprocessed image view
- Predicted DR grade with confidence
- Class probability bar chart

Works on Colab (share=True) and locally.
"""

import os

import gradio as gr
import numpy as np
import torch

from . import config
from .augmentations import get_valid_transforms
from .model import build_model
from .preprocessing import preprocess_image


def load_model(checkpoint_path: str | None = None, device: str = "auto"):
    """Load the trained model from a checkpoint."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = build_model(pretrained=False)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.CHECKPOINTS_DIR,
            "best_model_aptos_finetune.pth",
        )

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"No trained model checkpoint found at {checkpoint_path}. "
            f"Run notebook 03_training.ipynb first to train the model."
        )

    model = model.to(dev)
    model.eval()
    return model, dev


def predict_image(image: np.ndarray, model, device) -> tuple[np.ndarray, dict]:
    """Run prediction on a single image (numpy RGB from Gradio)."""
    preprocessed = preprocess_image(image, img_size=config.IMG_SIZE)

    transform = get_valid_transforms(config.IMG_SIZE)
    tensor = transform(image=preprocessed)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    confidences = {config.CLASS_NAMES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)}

    return preprocessed, confidences


def create_app(checkpoint_path: str | None = None):
    """Build and return the Gradio app."""
    model, device = load_model(checkpoint_path)

    def gradio_predict(image):
        if image is None:
            return None, {}
        preprocessed, confidences = predict_image(image, model, device)
        return preprocessed, confidences

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(type="numpy", label="Upload Retinal Image"),
        outputs=[
            gr.Image(type="numpy", label="Preprocessed Image"),
            gr.Label(num_top_classes=5, label="DR Grade Prediction"),
        ],
        title="Diabetic Retinopathy Detection",
        description=(
            "Upload a retinal fundus image to detect the severity of "
            "Diabetic Retinopathy. The model uses EfficientNetV2-S with "
            "CLAHE preprocessing and two-stage transfer learning.\n\n"
            "**Disclaimer:** This tool is for research and educational "
            "purposes only. It is NOT a certified medical device and must "
            "not be used as a substitute for professional medical diagnosis."
        ),
        examples=None,
        flagging_mode="never",
    )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
