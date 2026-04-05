"""
Master Execution Script for the 2-Stage Transfer Learning Pipeline.

Stage 1: Pre-train on the massive, noisy EyePACS dataset (3-phase progressive unfreezing).
Stage 2: Fine-tune on the small, highly curated APTOS dataset (single-phase full unfreeze).

Usage:
    python train_pipeline.py               # Full pipeline
    python train_pipeline.py --dry-run     # 1-epoch test run
    python train_pipeline.py --stage2-only # Skip Stage 1 (requires existing checkpoint)
"""

import argparse
import gc
import json
import os
import shutil

import torch

from src import config
from src.augmentations import get_light_train_transforms, get_train_transforms, get_valid_transforms
from src.data_manager import ensure_aptos_dataset, ensure_dataset
from src.dataset import (
    compute_class_weights,
    create_dataloaders,
    load_and_split_dataframe,
    preprocess_images_to_cache,
)
from src.evaluate import full_evaluation
from src.model import ModelEMA, build_model, count_parameters, freeze_backbone, unfreeze_backbone
from src.trainer import Trainer
from src.utils import get_logger, log_gpu_memory, seed_everything

logger = get_logger(__name__)


def run_stage1_eyepacs(device, dry_run: bool = False):
    """Stage 1: Pre-train on EyePACS with 3-phase progressive unfreezing."""
    logger.info("=" * 70)
    logger.info("  STAGE 1: Pre-training on EyePACS (~35K images)")
    logger.info("=" * 70)

    # Check for existing checkpoint (resume support)
    final_best_path = os.path.join(config.CHECKPOINTS_DIR, "best_eyepacs_model.pth")
    if os.path.exists(final_best_path) and not dry_run:
        logger.info(f"Stage 1 checkpoint already exists: {final_best_path}. Skipping Stage 1.")
        return final_best_path

    # Load & split data
    train_df, valid_df, _ = load_and_split_dataframe(dataset_name="eyepacs")
    if dry_run:
        train_df = train_df.head(100)
        valid_df = valid_df.head(100)
    logger.info(f"EyePACS split: train={len(train_df)}, valid={len(valid_df)}")

    # Build preprocessed cache
    logger.info("Building preprocessed image cache...")
    preprocess_images_to_cache(train_df, dataset_name="eyepacs")
    preprocess_images_to_cache(valid_df, dataset_name="eyepacs")

    # Transforms
    train_tfm_light = get_light_train_transforms(config.IMG_SIZE)
    train_tfm_heavy = get_train_transforms(config.IMG_SIZE)
    valid_tfm = get_valid_transforms(config.IMG_SIZE)

    loaders_light = create_dataloaders(
        train_df,
        valid_df,
        dataset_name="eyepacs",
        train_transform=train_tfm_light,
        valid_transform=valid_tfm,
        batch_size=config.STAGE1_BATCH_SIZE,
    )
    loaders_heavy = create_dataloaders(
        train_df,
        valid_df,
        dataset_name="eyepacs",
        train_transform=train_tfm_heavy,
        valid_transform=valid_tfm,
        batch_size=config.STAGE1_BATCH_SIZE,
    )

    # Model & EMA
    model = build_model(pretrained=True)
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Parameters: {count_parameters(model)}")

    ema = ModelEMA(model) if config.USE_EMA else None

    # Class weights
    class_weights = compute_class_weights(train_df["level"].values)
    trainer = Trainer(model, device, class_weights=class_weights, ema=ema)

    log_gpu_memory("After model init")

    # Override epochs for dry run
    p1_epochs = 1 if dry_run else config.STAGE1_PHASE1_EPOCHS
    p2_epochs = 1 if dry_run else config.STAGE1_PHASE2_EPOCHS
    p3_epochs = 1 if dry_run else config.STAGE1_PHASE3_EPOCHS

    # --- Phase 1: Head Only ---
    freeze_backbone(model)
    logger.info(f"Phase 1 trainable params: {count_parameters(model)}")
    trainer.fit_phase(
        phase="s1_phase1",
        train_loader=loaders_light["train"],
        valid_loader=loaders_light["valid"],
        epochs=p1_epochs,
        lr=config.STAGE1_PHASE1_LR,
        weight_decay=config.STAGE1_PHASE1_WD,
        use_mixup=False,
        use_amp=True,
    )

    # --- Phase 2: Partial Unfreeze ---
    trainer.load_best_checkpoint(phase="s1_phase1")
    unfreeze_backbone(model, from_block=config.STAGE1_PHASE2_UNFREEZE)
    logger.info(f"Phase 2 trainable params: {count_parameters(model)}")
    trainer.fit_phase(
        phase="s1_phase2",
        train_loader=loaders_heavy["train"],
        valid_loader=loaders_heavy["valid"],
        epochs=p2_epochs,
        lr=config.STAGE1_PHASE2_LR,
        weight_decay=config.STAGE1_PHASE2_WD,
        warmup_epochs=2,
        use_mixup=config.USE_MIXUP,
        use_amp=True,
    )

    # --- Phase 3: Full Unfreeze ---
    trainer.load_best_checkpoint(phase="s1_phase2")
    unfreeze_backbone(model, from_block=None)
    logger.info(f"Phase 3 trainable params: {count_parameters(model)}")
    trainer.fit_phase(
        phase="s1_phase3",
        train_loader=loaders_heavy["train"],
        valid_loader=loaders_heavy["valid"],
        epochs=p3_epochs,
        lr=config.STAGE1_PHASE3_LR,
        weight_decay=config.STAGE1_PHASE3_WD,
        warmup_epochs=2,
        use_mixup=config.USE_MIXUP,
        use_amp=True,
    )

    # Save the best model as the final Stage 1 output
    phase3_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_model_s1_phase3.pth")
    phase2_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_model_s1_phase2.pth")

    if os.path.exists(phase3_ckpt):
        shutil.copy(phase3_ckpt, final_best_path)
    elif os.path.exists(phase2_ckpt):
        shutil.copy(phase2_ckpt, final_best_path)
        logger.warning("Phase 3 checkpoint missing. Using Phase 2 checkpoint.")
    else:
        logger.error("No Stage 1 checkpoints found!")

    # Save training history
    trainer.save_history(os.path.join(config.LOGS_DIR, "stage1_history.json"))

    logger.info(f"Stage 1 Complete. Best model saved to {final_best_path}")
    return final_best_path


def run_stage2_aptos(device, pretrained_weights_path: str, dry_run: bool = False):
    """Stage 2: Fine-tune on APTOS with single-phase full unfreeze."""
    logger.info("=" * 70)
    logger.info("  STAGE 2: Fine-tuning on APTOS 2019 (~3.6K images)")
    logger.info("=" * 70)

    # Load & split data
    train_df, valid_df, test_df = load_and_split_dataframe(dataset_name="aptos")
    if dry_run:
        train_df = train_df.head(320)
        valid_df = valid_df.head(320)
        test_df = test_df.head(320)
    logger.info(f"APTOS split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Build preprocessed cache
    logger.info("Building preprocessed image cache for APTOS...")
    preprocess_images_to_cache(train_df, dataset_name="aptos")
    preprocess_images_to_cache(valid_df, dataset_name="aptos")
    if test_df is not None:
        preprocess_images_to_cache(test_df, dataset_name="aptos")

    # Transforms
    train_tfm = get_train_transforms(config.IMG_SIZE)
    valid_tfm = get_valid_transforms(config.IMG_SIZE)

    loaders = create_dataloaders(
        train_df,
        valid_df,
        dataset_name="aptos",
        test_df=test_df,
        train_transform=train_tfm,
        valid_transform=valid_tfm,
        batch_size=config.STAGE2_BATCH_SIZE,
    )

    # Model — load from Stage 1 weights
    model = build_model(pretrained=False)

    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(
            f"Stage 1 weights missing at {pretrained_weights_path}. Run Stage 1 first."
        )

    logger.info(f"Loading Stage 1 weights: {pretrained_weights_path}")
    ckpt = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ema = ModelEMA(model) if config.USE_EMA else None

    # Class weights for APTOS
    class_weights = compute_class_weights(train_df["level"].values)
    trainer = Trainer(model, device, class_weights=class_weights, ema=ema)

    # Full unfreeze for fine-tuning
    unfreeze_backbone(model, from_block=None)

    epochs = 2 if dry_run else config.STAGE2_EPOCHS

    trainer.fit_phase(
        phase="aptos_finetune",
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        epochs=epochs,
        lr=config.STAGE2_LR,
        weight_decay=config.STAGE2_WEIGHT_DECAY,
        warmup_epochs=config.STAGE2_WARMUP_EPOCHS,
        use_mixup=config.USE_MIXUP,
        use_amp=True,
    )

    trainer.load_best_checkpoint(phase="aptos_finetune")

    # Apply EMA weights for evaluation
    if ema is not None:
        ema.apply(trainer.model)
        logger.info("Applied EMA weights for evaluation.")

    # Save training history
    trainer.save_history(os.path.join(config.LOGS_DIR, "stage2_history.json"))

    logger.info("Stage 2 Complete. Running Final Evaluation...")

    # Final evaluation
    if test_df is not None and "test" in loaders:
        test_dataset = loaders["test"].dataset
        metrics = full_evaluation(
            trainer.model,
            loaders["test"],
            device,
            test_dataset=test_dataset,
        )

        # Save metrics
        metrics_path = os.path.join(config.OUTPUTS_DIR, "final_aptos_metrics.json")
        dump_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (dict, str)):
                dump_metrics[k] = v
            else:
                try:
                    dump_metrics[k] = float(v)
                except (TypeError, ValueError):
                    dump_metrics[k] = str(v)

        with open(metrics_path, "w") as f:
            json.dump(dump_metrics, f, indent=4)
        logger.info(f"Final metrics saved to {metrics_path}")

        return metrics

    return None


def main():
    parser = argparse.ArgumentParser(description="DR Two-Stage Training Pipeline")
    parser.add_argument(
        "--dry-run", action="store_true", help="Quick test with 1-2 epochs per phase"
    )
    parser.add_argument(
        "--stage2-only", action="store_true", help="Skip Stage 1, go directly to APTOS fine-tuning"
    )
    args = parser.parse_args()

    config.ensure_dirs()
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Pipeline starting. Device: {device}")
    log_gpu_memory("Initial")

    # Ensure datasets exist
    ensure_dataset()
    ensure_aptos_dataset()

    if args.stage2_only:
        best_eyepacs_weights = os.path.join(config.CHECKPOINTS_DIR, "best_eyepacs_model.pth")
        if not os.path.exists(best_eyepacs_weights):
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found at {best_eyepacs_weights}. "
                "Run full pipeline first or remove --stage2-only."
            )
    else:
        # Stage 1
        best_eyepacs_weights = run_stage1_eyepacs(device, dry_run=args.dry_run)

        # Clean up GPU memory between stages
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Stage 2
    run_stage2_aptos(device, best_eyepacs_weights, dry_run=args.dry_run)

    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
