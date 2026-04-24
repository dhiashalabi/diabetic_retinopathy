"""
Grad-CAM implementation for EfficientNetV2-S explainability.

Generates class-discriminative heatmaps highlighting the retinal
regions most influential for each DR severity prediction — critical
for clinical trust and model debugging.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .augmentations import get_valid_transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a specified convolutional layer to capture forward
    activations and backward gradients, then produces a heatmap
    indicating which spatial regions drive the predicted class.

    Parameters
    ----------
    model : nn.Module
        The full model (backbone + head).
    target_layer : nn.Module
        The convolutional layer to explain (typically the last conv
        block of the backbone).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove all registered hooks to prevent memory leaks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W), normalised.
        target_class : int or None
            Class index to explain.  If None, uses the predicted class.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H_input, W_input), values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Channel-wise global-average-pooled gradients as weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap onto the original image.

        Parameters
        ----------
        image : np.ndarray
            Original RGB image, uint8, shape (H, W, 3).
        heatmap : np.ndarray
            Heatmap in [0, 1], shape (H, W).
        alpha : float
            Blending weight for the heatmap.

        Returns
        -------
        np.ndarray
            Blended RGB image, uint8.
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        heatmap_color = cv2.resize(heatmap_color, (w, h))

        blended = cv2.addWeighted(image, 1.0 - alpha, heatmap_color, alpha, 0)
        return blended


def get_target_layer(model) -> nn.Module:
    """
    Automatically locate the last convolutional block of a timm backbone.

    For EfficientNetV2-S, this is the last block in ``model.backbone.blocks``.
    """
    backbone = model.backbone if hasattr(model, "backbone") else model

    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    elif hasattr(backbone, "features"):
        return backbone.features[-1]
    else:
        modules = list(backbone.modules())
        conv_layers = [m for m in modules if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]
        raise ValueError("Cannot find a suitable target layer for Grad-CAM")


def visualize_gradcam(
    model: nn.Module,
    image: np.ndarray,
    input_tensor: torch.Tensor,
    device: torch.device,
    target_class: int | None = None,
    class_names: list[str] = config.CLASS_NAMES,
    save_path: str | None = None,
):
    """
    Create a side-by-side visualisation:
    Original | Heatmap | Overlay

    Parameters
    ----------
    image : np.ndarray
        Original RGB image, uint8 (before normalisation).
    input_tensor : torch.Tensor
        Normalised tensor (1, 3, H, W) on the correct device.
    """
    target_layer = get_target_layer(model)
    cam = GradCAM(model, target_layer)

    try:
        input_tensor = input_tensor.to(device)
        heatmap = cam.generate_heatmap(input_tensor, target_class=target_class)
        overlay = cam.overlay_heatmap(image, heatmap)
    finally:
        cam.remove_hooks()

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
    pred_cls = probs.argmax()
    pred_name = class_names[pred_cls]
    confidence = probs[pred_cls]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    target_name = class_names[target_class] if target_class is not None else pred_name
    axes[1].set_title(f"Grad-CAM: {target_name}", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {pred_name} ({confidence:.1%})",
        fontsize=13,
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def visualize_gradcam_grid(
    model: nn.Module,
    dataset,
    device: torch.device,
    indices: list[int] | None = None,
    n_per_class: int = 3,
    class_names: list[str] = config.CLASS_NAMES,
    save_path: str | None = None,
):
    """
    Grid of Grad-CAM overlays: rows = classes, columns = examples.

    If ``indices`` is None, picks ``n_per_class`` random samples per class.
    """
    target_layer = get_target_layer(model)
    cam = GradCAM(model, target_layer)

    try:
        if indices is None:
            labels = (
                np.array(dataset.labels)
                if hasattr(dataset, "labels")
                else np.array([dataset[i][1] for i in range(len(dataset))])
            )
            indices = []
            for cls in range(config.NUM_CLASSES):
                cls_idx = np.where(labels == cls)[0]
                if len(cls_idx) == 0:
                    continue
                chosen = np.random.choice(
                    cls_idx,
                    size=min(n_per_class, len(cls_idx)),
                    replace=False,
                )
                indices.extend(chosen.tolist())

        n_cols = n_per_class
        n_rows = config.NUM_CLASSES
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        fig.suptitle("Grad-CAM Explanations per DR Grade", fontsize=18, y=1.02)

        sample_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row, col] if n_rows > 1 else axes[col]
                if sample_idx >= len(indices):
                    ax.axis("off")
                    continue

                idx = indices[sample_idx]
                img_tensor, label = dataset[idx]
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = img_np * np.array(config.IMG_STD) + np.array(config.IMG_MEAN)
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    input_t = img_tensor.unsqueeze(0).to(device)
                else:
                    img_np = img_tensor
                    img_arr = img_np.astype(np.float32) / 255.0
                    img_arr = (img_arr - np.array(config.IMG_MEAN)) / np.array(config.IMG_STD)
                    input_t = (
                        torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    )

                heatmap = cam.generate_heatmap(input_t, target_class=label)
                overlay = cam.overlay_heatmap(img_np, heatmap)

                ax.imshow(overlay)
                ax.set_title(
                    f"{class_names[label]}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax.axis("off")
                sample_idx += 1
    finally:
        cam.remove_hooks()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# -----------------------------------------------------------------------
# Notebook-friendly API (used by ``05_gradcam.ipynb``)
# -----------------------------------------------------------------------


def _resolve_aptos_image_path(image_id: str, images_dir: str) -> str | None:
    """Try preprocessed cache first, then ``images_dir``, with common extensions."""
    cache_dir = os.path.join(config.APTOS_DATA_DIR, f"preprocessed_{config.IMG_SIZE}_aptos")
    for base in (cache_dir, images_dir):
        for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG"):
            p = os.path.join(base, str(image_id) + ext)
            if os.path.isfile(p):
                return p
    return None


class GradCAMExplainer:
    """
    High-level wrapper: validation transforms + Grad-CAM for raw RGB uint8 images.

    ``explain`` returns a heatmap at model input resolution, an overlay on the
    full-resolution ``img_rgb``, the predicted class index, and a probability vector.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: list[str] = config.CLASS_NAMES,
        img_size: int = config.IMG_SIZE,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.img_size = img_size
        self.target_layer = get_target_layer(model)
        self._valid_tfm = get_valid_transforms(img_size)

    def explain(
        self,
        img_rgb: np.ndarray,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Returns
        -------
        heatmap : np.ndarray
            2D array in [0, 1], shape (``img_size``, ``img_size``).
        overlay : np.ndarray
            RGB uint8, same height/width as ``img_rgb``.
        pred_class : int
            Argmax class index.
        proba : np.ndarray
            Shape (num_classes,), float32.
        """
        t = self._valid_tfm(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        cam = GradCAM(self.model, self.target_layer)
        try:
            heatmap = cam.generate_heatmap(t, target_class=target_class)
            overlay = cam.overlay_heatmap(img_rgb, heatmap)
        finally:
            cam.remove_hooks()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(t)
            proba = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        pred_class = int(proba.argmax())
        return heatmap, overlay, pred_class, proba


def generate_gradcam_grid(
    explainer: GradCAMExplainer,
    test_df,
    images_dir: str,
    n_per_class: int = 3,
    class_names: list[str] = config.CLASS_NAMES,
    save_path: str | None = None,
):
    """
    Grid of Grad-CAM overlays: rows = DR class, columns = up to ``n_per_class`` examples.
    Resolves file paths the same way as the Grad-CAM notebook (cache, then raw dir).
    """
    n_rows = config.NUM_CLASSES
    n_cols = n_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    fig.suptitle("Grad-CAM Explanations per DR Grade", fontsize=16, y=1.01)

    for row in range(n_rows):
        subset = test_df[test_df["level"] == row].head(n_per_class)
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(subset):
                ax.axis("off")
                continue
            image_id = subset.iloc[col]["image"]
            p = _resolve_aptos_image_path(image_id, images_dir)
            if p is None:
                ax.axis("off")
                continue
            bgr = cv2.imread(p)
            if bgr is None:
                ax.axis("off")
                continue
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _heatmap, overlay, _pred, _p = explainer.explain(img_rgb)
            ax.imshow(overlay)
            ax.set_title(class_names[row], fontsize=10, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
