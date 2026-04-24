"""
Master Execution Script for the 2-Stage Transfer Learning Pipeline.

Stage 1: Pre-train on the massive, noisy EyePACS dataset (3-phase progressive unfreezing).
Stage 2: Fine-tune on the small, highly curated APTOS dataset (single-phase full unfreeze).

Usage:
    python train_pipeline.py               # Full pipeline
    python train_pipeline.py --dry-run     # 1-epoch test run
    python train_pipeline.py --stage2-only # Skip Stage 1 (requires existing checkpoint)
    python train_pipeline.py --aptos-only  # APTOS only from ImageNet weights (ablation baseline)
"""

import argparse
import gc
import json
import os
import shutil
from typing import cast

import torch
import torch.nn as nn

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
from src.model import (
    DRModel,
    ModelEMA,
    build_model,
    count_parameters,
    freeze_backbone,
    unfreeze_backbone,
)
from src.trainer import Trainer
from src.utils import get_logger, log_gpu_memory, seed_everything, write_run_manifest

logger = get_logger(__name__)


def _maybe_torch_compile(model: nn.Module, device: torch.device) -> nn.Module:
    """Optional torch.compile on CUDA for throughput (Hopper H100/H200-friendly)."""
    if device.type != "cuda" or not getattr(config, "USE_TORCH_COMPILE", False):
        return model
    try:
        compiled = cast(nn.Module, torch.compile(model))
        logger.info("torch.compile enabled for this stage.")
        return compiled
    except Exception as e:
        logger.warning("torch.compile skipped: %s", e)
        return model


def run_stage1_eyepacs(device, dry_run: bool = False):
    """Stage 1: Pre-train on EyePACS with 3-phase progressive unfreezing."""
    logger.info("=" * 70)
    logger.info("  STAGE 1: Pre-training on EyePACS (~35K images)")
    logger.info("=" * 70)

    # Check for existing checkpoint (resume support)
    final_best_path = os.path.join(config.CHECKPOINTS_DIR, "best_eyepacs_model.pth")
    if os.path.exists(final_best_path) and not dry_run:
        logger.info(f"Stage 1 checkpoint already exists: {final_best_path}. Skipping Stage 1.")
        write_run_manifest(
            config.RUN_MANIFEST_PATH,
            {
                "stage": "eyepacs",
                "stage1_skipped": True,
                "checkpoint_stage1": final_best_path,
                "AMP_DTYPE": config.AMP_DTYPE,
                "MODEL_NAME": config.MODEL_NAME,
            },
            merge=True,
        )
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
    model = cast(DRModel, _maybe_torch_compile(model, device))
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Parameters: {count_parameters(model)}")

    ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

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
    # Fresh loaders so sampler / worker state does not carry over from Phase 2
    loaders_heavy = create_dataloaders(
        train_df,
        valid_df,
        dataset_name="eyepacs",
        train_transform=train_tfm_heavy,
        valid_transform=valid_tfm,
        batch_size=config.STAGE1_BATCH_SIZE,
    )
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
    write_run_manifest(
        config.RUN_MANIFEST_PATH,
        {
            "stage": "eyepacs",
            "dry_run": dry_run,
            "AMP_DTYPE": config.AMP_DTYPE,
            "MODEL_NAME": config.MODEL_NAME,
            "STAGE1_BATCH_SIZE": config.STAGE1_BATCH_SIZE,
            "PREPROCESSING_CACHE_VERSION": config.PREPROCESSING_CACHE_VERSION,
            "checkpoint_stage1": final_best_path,
        },
        merge=True,
    )
    return final_best_path


def run_stage2_aptos(
    device,
    pretrained_weights_path: str | None,
    dry_run: bool = False,
    aptos_only: bool = False,
    use_tta: bool = False,
):
    """Stage 2: Fine-tune on APTOS with single-phase full unfreeze."""
    logger.info("=" * 70)
    if aptos_only:
        logger.info("  STAGE 2 (APTOS-ONLY): Fine-tuning from ImageNet weights (ablation)")
    else:
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

    # Model — Stage 1 weights or ImageNet-only (ablation)
    if aptos_only:
        model = build_model(pretrained=True)
        logger.info("APTOS-only mode: training from ImageNet pretrained weights (no EyePACS).")
    else:
        model = build_model(pretrained=False)
        if not pretrained_weights_path or not os.path.exists(pretrained_weights_path):
            raise FileNotFoundError(
                f"Stage 1 weights missing at {pretrained_weights_path}. Run Stage 1 first."
            )
        logger.info(f"Loading Stage 1 weights: {pretrained_weights_path}")
        ckpt = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model = cast(DRModel, _maybe_torch_compile(model, device))
    ema = ModelEMA(model, decay=config.STAGE2_EMA_DECAY) if config.USE_EMA else None

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

    aptos_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_model_aptos_finetune.pth")
    write_run_manifest(
        config.RUN_MANIFEST_PATH,
        {
            "stage_aptos_finetune": True,
            "aptos_only": aptos_only,
            "dry_run": dry_run,
            "checkpoint_aptos": aptos_ckpt,
            "STAGE2_BATCH_SIZE": config.STAGE2_BATCH_SIZE,
            "AMP_DTYPE": config.AMP_DTYPE,
        },
        merge=True,
    )

    logger.info("Stage 2 Complete. Running Final Evaluation...")

    # Final evaluation
    if test_df is not None and "test" in loaders:
        test_dataset = loaders["test"].dataset
        if use_tta:
            metrics = full_evaluation(
                trainer.model,
                loaders["test"],
                device,
                test_dataset=test_dataset,
                use_tta=True,
                valid_df=valid_df,
                test_df=test_df,
                dataset_name="aptos",
            )
        else:
            metrics = full_evaluation(
                trainer.model,
                loaders["test"],
                device,
                test_dataset=test_dataset,
                valid_loader=loaders["valid"],
            )

        # Save metrics
        metrics_name = "aptos_only_metrics.json" if aptos_only else "final_aptos_metrics.json"
        metrics_path = os.path.join(config.OUTPUTS_DIR, metrics_name)
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

        qwk_val = metrics.get("qwk")
        write_run_manifest(
            config.RUN_MANIFEST_PATH,
            {
                "evaluation_cli": {
                    "qwk": float(qwk_val) if qwk_val is not None else None,
                    "use_tta": use_tta,
                    "metrics_path": metrics_path,
                    "figures_dir": config.FIGURES_DIR,
                },
            },
            merge=True,
        )

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
    parser.add_argument(
        "--aptos-only",
        action="store_true",
        help="Skip Stage 1; fine-tune on APTOS from ImageNet weights only (ablation baseline)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Final eval with test-time augmentation (val/test TTA + QWK thresholds on val TTA)",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    logger.info(f"Pipeline starting. Device: {device}")
    log_gpu_memory("Initial")

    # Ensure datasets exist
    ensure_dataset()
    ensure_aptos_dataset()

    if args.aptos_only and args.stage2_only:
        raise ValueError("Use either --aptos-only or --stage2-only, not both.")

    if args.aptos_only:
        best_eyepacs_weights = None
    elif args.stage2_only:
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
    run_stage2_aptos(
        device,
        best_eyepacs_weights,
        dry_run=args.dry_run,
        aptos_only=args.aptos_only,
        use_tta=args.tta,
    )

    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
