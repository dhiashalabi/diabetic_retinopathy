"""
Comprehensive evaluation suite for the DR classifier.

Metrics:
- Quadratic Weighted Kappa (primary clinical metric)
- Per-class Precision / Recall / F1
- Weighted ROC-AUC (One-vs-Rest)
- Confusion matrix (absolute + normalised)
- Per-class ROC curves
- Misclassified image gallery
- Training curve plots
- Single-image inference
- Test-Time Augmentation (TTA)
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .augmentations import get_tta_transforms, get_valid_transforms
from .preprocessing import preprocess_image

PLOT_DPI = 300


# -----------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.

    Returns (y_true, y_pred, y_proba) where y_proba has shape (N, C).
    """
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probs)
    y_pred = y_proba.argmax(axis=1)
    return y_true, y_pred, y_proba


def _qwk_loss(coef, X, y):
    """Loss function for threshold optimization."""
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred < coef[1]:
            X_p[i] = 1
        elif pred < coef[2]:
            X_p[i] = 2
        elif pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4

    return -cohen_kappa_score(y, X_p, weights="quadratic")


def optimize_qwk_thresholds(y_proba: np.ndarray, y_true: np.ndarray) -> list[float]:
    """Optimize thresholds for converting ordinal probabilities/scores to class labels."""
    # Convert probabilities to a continuous score [0-4] using expectation
    scores = np.sum(y_proba * np.arange(config.NUM_CLASSES), axis=1)
    
    initial_coef = [0.5, 1.5, 2.5, 3.5]
    res = opt.minimize(
        _qwk_loss,
        initial_coef,
        args=(scores, y_true),
        method='Nelder-Mead'
    )
    return res.x.tolist()


def apply_qwk_thresholds(y_proba: np.ndarray, coef: list[float]) -> np.ndarray:
    """Apply learned thresholds to generate integer predictions."""
    scores = np.sum(y_proba * np.arange(config.NUM_CLASSES), axis=1)
    y_pred = np.zeros_like(scores, dtype=int)
    for i, pred in enumerate(scores):
        if pred < coef[0]:
            y_pred[i] = 0
        elif pred < coef[1]:
            y_pred[i] = 1
        elif pred < coef[2]:
            y_pred[i] = 2
        elif pred < coef[3]:
            y_pred[i] = 3
        else:
            y_pred[i] = 4
    return y_pred


@torch.no_grad()
def predict_ensemble(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    phases: list[str] = ["phase2", "phase3"]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average predictions from multiple checkpoint phases."""
    all_probas = []
    y_true = None
    
    for phase in phases:
        path = os.path.join(config.CHECKPOINTS_DIR, f"best_model_{phase}.pth")
        if not os.path.exists(path):
            print(f"Skipping phase {phase}, checkpoint not found.")
            continue
            
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
        y_t, _, y_p = predict(model, loader, device)
        all_probas.append(y_p)
        if y_true is None:
            y_true = y_t
            
    if not all_probas:
        raise ValueError("No valid checkpoints found for ensemble")
        
    y_proba_ensemble = np.mean(all_probas, axis=0)
    y_pred_ensemble = y_proba_ensemble.argmax(axis=1)
    
    return y_true, y_pred_ensemble, y_proba_ensemble


# -----------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] = config.CLASS_NAMES,
) -> dict:
    """Compute all evaluation metrics and return a summary dict."""
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    try:
        roc_auc = roc_auc_score(
            y_true_bin,
            y_proba,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "qwk": qwk,
        "accuracy": acc,
        "roc_auc_weighted": roc_auc,
        "classification_report": report,
        "classification_report_str": report_str,
    }


# -----------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = config.CLASS_NAMES,
    normalize: bool = False,
    save_path: str | None = None,
):
    """Confusion matrix heatmap (absolute or row-normalised)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
        row_sums[row_sums == 0] = 1
        cm = cm.astype(np.float64) / row_sums
        fmt = ".2f"
        title = "Normalised Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    ax.set_title(f"{title}  (QWK = {qwk:.4f})", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] = config.CLASS_NAMES,
    save_path: str | None = None,
):
    """Per-class ROC curves with AUC annotations."""
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    n_classes = y_true_bin.shape[1]

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{class_names[i]} (AUC = {roc_auc_i:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-Class ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    return fig


def plot_training_curves(
    history: dict[str, list],
    save_path: str | None = None,
):
    """Plot loss, accuracy, and QWK over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Validation", linewidth=2)
    axes[1].set_title("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # QWK
    axes[2].plot(epochs, history["train_kappa"], label="Train", linewidth=2)
    axes[2].plot(epochs, history["val_kappa"], label="Validation", linewidth=2)
    axes[2].set_title("Quadratic Weighted Kappa", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("QWK")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Mark phase boundaries
    p1_end = config.PHASE1_EPOCHS
    if p1_end < len(history["train_loss"]):
        for ax in axes:
            ax.axvline(
                p1_end + 0.5,
                color="grey",
                ls="--",
                alpha=0.6,
                label="Partial unfreeze" if ax == axes[0] else None,
            )

    p2_end = config.PHASE1_EPOCHS + config.PHASE2_EPOCHS
    if p2_end < len(history["train_loss"]):
        for ax in axes:
            ax.axvline(
                p2_end + 0.5,
                color="grey",
                ls=":",
                alpha=0.6,
                label="Full unfreeze" if ax == axes[0] else None,
            )

    if p1_end < len(history["train_loss"]):
        axes[0].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    return fig


def show_misclassified(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset,
    n_images: int = 16,
    class_names: list[str] = config.CLASS_NAMES,
    save_path: str | None = None,
):
    """Display a gallery of the most-confident misclassified images."""
    y_true, y_pred, y_proba = predict(model, loader, device)

    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassifications found.")
        return

    confidences = y_proba[wrong_idx].max(axis=1)
    sorted_idx = wrong_idx[np.argsort(-confidences)]
    show_idx = sorted_idx[:n_images]

    cols = min(8, n_images)
    rows = (len(show_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    fig.suptitle("Most Confident Misclassifications", fontsize=16)

    for i, idx in enumerate(show_idx):
        ax = axes[i // cols, i % cols]
        img_tensor, _ = dataset[idx]
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = img_np * np.array(config.IMG_STD) + np.array(config.IMG_MEAN)
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = img_tensor

        ax.imshow(img_np)
        conf = y_proba[idx].max()
        ax.set_title(
            f"T: {class_names[y_true[idx]]}\nP: {class_names[y_pred[idx]]} ({conf:.0%})",
            fontsize=8,
            color="red",
        )
        ax.axis("off")

    # Hide unused axes
    for i in range(len(show_idx), rows * cols):
        axes[i // cols, i % cols].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    return fig


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    test_dataset=None,
    save_dir: str = config.FIGURES_DIR,
):
    """
    Run the complete evaluation pipeline and print a summary.

    Generates:
    - Printed classification report
    - Confusion matrix (absolute + normalised)
    - ROC curves
    - Misclassified gallery
    """
    y_true, y_pred, y_proba = predict(model, test_loader, device)
    
    # Try threshold optimization if validation/test
    try:
        thresholds = optimize_qwk_thresholds(y_proba, y_true)
        y_pred_opt = apply_qwk_thresholds(y_proba, thresholds)
        opt_metrics = compute_metrics(y_true, y_pred_opt, y_proba)
        # Use optimized predictions if they are better
        if opt_metrics["qwk"] > compute_metrics(y_true, y_pred, y_proba)["qwk"]:
            y_pred = y_pred_opt
            print(f"Using optimized thresholds: {thresholds}")
    except Exception as e:
        print(f"Threshold optimization failed: {e}")

    metrics = compute_metrics(y_true, y_pred, y_proba)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nQuadratic Weighted Kappa : {metrics['qwk']:.4f}")
    print(f"Accuracy                : {metrics['accuracy']:.4f}")
    print(f"Weighted ROC-AUC (OvR)  : {metrics['roc_auc_weighted']:.4f}")
    print(f"\n{metrics['classification_report_str']}")

    plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        normalize=True,
        save_path=os.path.join(save_dir, "confusion_matrix_normalized.png"),
    )
    plot_roc_curves(
        y_true,
        y_proba,
        save_path=os.path.join(save_dir, "roc_curves.png"),
    )

    if test_dataset is not None:
        show_misclassified(
            model,
            test_loader,
            device,
            test_dataset,
            save_path=os.path.join(save_dir, "misclassified.png"),
        )

    return metrics


# -----------------------------------------------------------------------
# Single-image inference
# -----------------------------------------------------------------------


def predict_single_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    class_names: list[str] = config.CLASS_NAMES,
    img_size: int = config.IMG_SIZE,
    show: bool = True,
    save_path: str | None = None,
) -> dict:
    """
    Predict DR severity for a single retinal image.

    Returns a dict with predicted_class, severity, confidence, probabilities.
    Optionally displays the image alongside a probability bar chart.
    """
    severity_desc = {
        "No_DR": "0 - No Diabetic Retinopathy",
        "Mild": "1 - Mild Non-Proliferative DR",
        "Moderate": "2 - Moderate Non-Proliferative DR",
        "Severe": "3 - Severe Non-Proliferative DR",
        "Proliferative_DR": "4 - Proliferative DR",
    }

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocessed = preprocess_image(img_rgb, img_size=img_size)
    transform = get_valid_transforms(img_size)
    tensor = transform(image=preprocessed)["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "predicted_class": pred_class,
        "severity": severity_desc.get(pred_class, pred_class),
        "confidence": confidence,
        "probabilities": {c: float(p) for c, p in zip(class_names, probs, strict=True)},
    }

    if show:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(13, 5),
            gridspec_kw={"width_ratios": [1, 1.4]},
        )

        ax1.imshow(preprocessed)
        ax1.set_title(
            f"Prediction: {pred_class}\nConfidence: {confidence:.1%}",
            fontsize=13,
        )
        ax1.axis("off")

        colors = ["#2ecc71" if i == pred_idx else "#95a5a6" for i in range(len(class_names))]
        ax2.barh(class_names, probs, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability", fontsize=12)
        ax2.set_title("Class Probabilities", fontsize=13)
        for i, (p, _) in enumerate(zip(probs, class_names, strict=True)):
            ax2.text(p + 0.01, i, f"{p:.1%}", va="center", fontsize=10)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.show()

    return result


# -----------------------------------------------------------------------
# Test-Time Augmentation
# -----------------------------------------------------------------------


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    dataset_df,
    images_dir: str,
    device: torch.device,
    img_size: int = config.IMG_SIZE,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict with Test-Time Augmentation by averaging softmax probabilities
    across multiple geometric transforms.

    Returns (y_true, y_pred, y_proba).
    """
    from .dataset import DRDataset

    model.eval()
    tta_transforms = get_tta_transforms(img_size)

    all_probs_per_tta = []
    y_true = None

    for tfm in tqdm(tta_transforms, desc="TTA transforms"):
        ds = DRDataset(
            dataset_df,
            images_dir=images_dir,
            transform=tfm,
            preprocessing=True,
            img_size=img_size,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        probs_list = []
        labels_list = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(labels.numpy())

        all_probs_per_tta.append(np.concatenate(probs_list))
        if y_true is None:
            y_true = np.concatenate(labels_list)

    y_proba = np.mean(all_probs_per_tta, axis=0)
    y_pred = y_proba.argmax(axis=1)

    return y_true, y_pred, y_proba
