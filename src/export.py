"""
ONNX export for production deployment.

Converts the trained PyTorch checkpoint into a standalone ONNX model that
can be served by any framework (FastAPI, Flask, Django, etc.) using only
onnxruntime — no PyTorch or timm required at inference time.
"""

import json
import os

import numpy as np
import onnx
import torch

from . import config
from .model import build_model
from .utils import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    img_size: int = config.STAGE2_IMG_SIZE,
    opset_version: int = 17,
) -> str:
    """
    Export the trained DR model to ONNX format.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to the ``.pth`` checkpoint. Defaults to ``best_model_aptos_finetune.pth``.
    output_path : str, optional
        Where to write the ``.onnx`` file. Defaults to ``CHECKPOINTS_DIR/model.onnx``.
    img_size : int
        Input image resolution (must match training).
    opset_version : int
        ONNX opset version. 17 is recommended for broad compatibility.

    Returns
    -------
    str
        Absolute path to the exported ONNX file.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.CHECKPOINTS_DIR,
            "best_model_aptos_finetune.pth",
        )
    if output_path is None:
        output_path = os.path.join(config.CHECKPOINTS_DIR, "model.onnx")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")

    device = torch.device("cpu")
    model = build_model(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"QWK={ckpt.get('metrics', {}).get('kappa', '?')})"
    )

    dummy_input = torch.randn(1, 3, img_size, img_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    metadata = {
        "class_names": json.dumps(config.CLASS_NAMES),
        "img_size": str(img_size),
        "img_mean": json.dumps(config.IMG_MEAN),
        "img_std": json.dumps(config.IMG_STD),
        "model_name": config.MODEL_NAME,
        "num_classes": str(config.NUM_CLASSES),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, output_path)

    _validate_export(model, output_path, dummy_input)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"ONNX model exported: {output_path} ({file_size_mb:.1f} MB, opset {opset_version})"
    )

    return os.path.abspath(output_path)


def _validate_export(
    torch_model: torch.nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    atol: float = 1e-5,
) -> None:
    """Verify ONNX output matches PyTorch output within tolerance."""
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.abs(torch_out - ort_out).max()
    if max_diff > atol:
        logger.warning(
            f"ONNX validation: max difference {max_diff:.2e} exceeds tolerance {atol:.2e}"
        )
    else:
        logger.info(f"ONNX validation passed (max diff: {max_diff:.2e})")


def test_onnx_inference(
    onnx_path: str | None = None,
    image_path: str | None = None,
) -> dict:
    """
    Run inference on a single image using the exported ONNX model.

    Parameters
    ----------
    onnx_path : str, optional
        Path to the ONNX file. Defaults to ``CHECKPOINTS_DIR/model.onnx``.
    image_path : str, optional
        Path to a retinal fundus image. If None, uses a random test tensor.

    Returns
    -------
    dict
        Prediction result with class name, confidence, and all probabilities.
    """
    import onnxruntime as ort

    if onnx_path is None:
        onnx_path = os.path.join(config.CHECKPOINTS_DIR, "model.onnx")

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = ort.InferenceSession(onnx_path)

    if image_path is not None:
        import cv2

        from .augmentations import get_valid_transforms
        from .preprocessing import preprocess_image

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(img, img_size=config.STAGE2_IMG_SIZE)
        transform = get_valid_transforms(config.STAGE2_IMG_SIZE)
        tensor = transform(image=preprocessed)["image"].unsqueeze(0).numpy()
    else:
        tensor = np.random.randn(
            1,
            3,
            config.STAGE2_IMG_SIZE,
            config.STAGE2_IMG_SIZE,
        ).astype(np.float32)

    logits = session.run(None, {"input": tensor})[0]

    probs = _softmax(logits[0])
    pred_idx = int(probs.argmax())

    result = {
        "predicted_class": config.CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            name: float(p) for name, p in zip(config.CLASS_NAMES, probs, strict=True)
        },
    }

    logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
    return result


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
