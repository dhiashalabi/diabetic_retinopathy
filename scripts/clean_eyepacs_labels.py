#!/usr/bin/env python3
"""
EyePACS Noisy-Label Cleaning Script.

Uses the trained APTOS fine-tuned model (QWK ~0.91+) to detect and remove
mislabeled images in the EyePACS dataset. EyePACS labels are notoriously
noisy — inter-grader agreement was extremely poor in the original Kaggle
competition. Cleaning the worst offenders means the Stage 1 backbone
learns cleaner foundational features, which cascades to better Stage 2
fine-tuning and higher overall accuracy and QWK.

Strategy:
  1. Load the best APTOS fine-tuned model.
  2. Run inference on ALL EyePACS images.
  3. Flag images where |model_prediction - label| >= threshold (default 2).
  4. Save a cleaned CSV with flagged images removed.

Usage:
  python scripts/clean_eyepacs_labels.py [--checkpoint PATH] [--threshold N]

After running, the cleaned CSV will be auto-detected by
``load_and_split_dataframe()`` — simply retrain Stage 1 from scratch.
"""

import argparse
import os
import sys

# Project root must be on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.augmentations import get_valid_transforms
from src.dataset import DRDataset, load_and_split_dataframe
from src.evaluate import _eval_autocast, _softmax_probs_numpy
from src.model import DRModel, ModelEMA, build_model
from src.utils import get_device, seed_everything


def run_inference_on_eyepacs(
    model: DRModel,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the full EyePACS dataset.

    Returns (image_ids, true_labels, predicted_labels).
    """
    # Load the full EyePACS dataframe (no split — we need ALL images)
    csv_path = config.EYEPACS_LABELS_CSV
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"EyePACS labels CSV missing: {csv_path}")

    df = pd.read_csv(csv_path)
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df["source"] = "eyepacs"
    df["patient_id"] = df["image"].str.extract(r"(\d+)_")[0]
    df["patient_id"] = df["patient_id"].fillna(df["image"]).astype(str)

    print(f"EyePACS dataset: {len(df)} images")
    print(f"Class distribution:\n{df['level'].value_counts().sort_index()}\n")

    # Use Stage 2 (APTOS) resolution for inference — the model was trained at this size
    img_size = config.STAGE2_IMG_SIZE
    transform = get_valid_transforms(img_size)

    dataset = DRDataset(
        df,
        dataset_name="eyepacs",
        transform=transform,
        preprocessing=True,
        img_size=img_size,
        fail_on_missing=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model.eval()
    all_preds = []
    all_labels = []

    print("Running inference on EyePACS...")
    for images, labels in tqdm(loader, desc="EyePACS inference"):
        images = images.to(device, non_blocking=True)
        with _eval_autocast(device):
            logits = model(images)
        probs = _softmax_probs_numpy(logits)
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    image_ids = df["image"].values

    return image_ids, all_labels, all_preds


def clean_labels(
    image_ids: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    threshold: int = config.EYEPACS_CLEANING_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag and remove images where |prediction - label| >= threshold.

    Returns (cleaned_df, flagged_df).
    """
    disagreements = np.abs(pred_labels.astype(int) - true_labels.astype(int))

    flagged_mask = disagreements >= threshold
    clean_mask = ~flagged_mask

    print(f"\n{'=' * 60}")
    print("EYEPACS LABEL CLEANING REPORT")
    print(f"{'=' * 60}")
    print(f"Total images:        {len(image_ids)}")
    print(f"Cleaning threshold:  |pred - label| >= {threshold}")
    print(f"Flagged (removed):   {flagged_mask.sum()} ({100 * flagged_mask.mean():.2f}%)")
    print(f"Clean (kept):        {clean_mask.sum()} ({100 * clean_mask.mean():.2f}%)")

    # Breakdown by disagreement level
    print("\nDisagreement distribution:")
    for d in range(5):
        count = (disagreements == d).sum()
        pct = 100 * count / len(disagreements)
        status = "KEEP" if d < threshold else "REMOVE"
        print(f"  |pred-label|={d}: {count:>6} ({pct:>5.2f}%)  [{status}]")

    # Load original CSV to build cleaned version
    original_df = pd.read_csv(config.EYEPACS_LABELS_CSV)
    cols_to_drop = [c for c in original_df.columns if c.startswith("Unnamed")]
    original_df = original_df.drop(columns=cols_to_drop, errors="ignore")

    flagged_images = set(image_ids[flagged_mask])
    cleaned_df = original_df[~original_df["image"].isin(flagged_images)].reset_index(drop=True)

    flagged_df = pd.DataFrame(
        {
            "image": image_ids[flagged_mask],
            "original_label": true_labels[flagged_mask],
            "model_prediction": pred_labels[flagged_mask],
            "disagreement": disagreements[flagged_mask],
        }
    )

    print(f"\nCleaned CSV rows: {len(cleaned_df)} (from {len(original_df)})")
    print("Cleaned class distribution:")
    print(f"{cleaned_df['level'].value_counts().sort_index()}")

    return cleaned_df, flagged_df


def main():
    parser = argparse.ArgumentParser(description="Clean noisy EyePACS labels")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINTS_DIR, "best_model_aptos_finetune.pth"),
        help="Path to APTOS fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config.EYEPACS_CLEANING_THRESHOLD,
        help="Minimum |pred - label| to flag as noisy (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.EYEPACS_CLEANED_LABELS_CSV,
        help="Output path for cleaned CSV",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    seed_everything()
    device = get_device()

    # --- Load model ---
    print(f"Loading model from: {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Train the APTOS fine-tuned model first (notebook 03b).")
        sys.exit(1)

    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Try loading EMA weights first (they generalize better)
    if "ema_state_dict" in ckpt:
        print("Loading EMA model weights (better generalization)...")
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"Checkpoint metrics: {ckpt.get('metrics', 'N/A')}")

    # --- Run inference ---
    image_ids, true_labels, pred_labels = run_inference_on_eyepacs(
        model,
        device,
        batch_size=args.batch_size,
    )

    # --- Clean labels ---
    cleaned_df, flagged_df = clean_labels(
        image_ids,
        true_labels,
        pred_labels,
        threshold=args.threshold,
    )

    # --- Save ---
    cleaned_df.to_csv(args.output, index=False)
    print(f"\n✓ Cleaned CSV saved to: {args.output}")

    flagged_path = args.output.replace(".csv", "_flagged.csv")
    flagged_df.to_csv(flagged_path, index=False)
    print(f"✓ Flagged images saved to: {flagged_path}")

    print("\nNext steps:")
    print("  1. Retrain Stage 1 (notebook 03_training.ipynb)")
    print("     (Cleaned labels will be auto-detected — no config change needed)")
    print("  2. Run K-fold ensemble training (notebook 08_aptos_kfold_ensemble.ipynb)")


if __name__ == "__main__":
    main()
