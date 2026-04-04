"""
Master Execution Script for the 2-Stage Transfer Learning Pipeline.

Stage 1: Pre-train on the massive, noisy EyePACS dataset.
Stage 2: Fine-tune on the small, highly curated APTOS dataset.
"""

import os
import gc
import json
import shutil
import torch

from src import config
from src.utils import get_logger, set_seed
from src.data_manager import ensure_aptos_dataset
from src.dataset import load_and_split_dataframe, create_dataloaders, preprocess_images_to_cache
from src.augmentations import get_train_transforms, get_train_transforms_light, get_valid_transforms
from src.model import build_model, count_parameters, freeze_backbone, unfreeze_backbone
from src.trainer import Trainer
from src.evaluate import full_evaluation

logger = get_logger(__name__)


def run_stage1_eyepacs(device):
    logger.info("=== Initializing Stage 1: Pre-training on EyePACS ===")
    
    # Load Data
    train_df, valid_df, _ = load_and_split_dataframe(dataset_name="eyepacs")
    
    # Preprocess
    preprocess_images_to_cache(train_df, dataset_name="eyepacs")
    preprocess_images_to_cache(valid_df, dataset_name="eyepacs")
    
    # Transforms
    train_tfm_light = get_train_transforms_light(config.IMG_SIZE)
    train_tfm_heavy = get_train_transforms(config.IMG_SIZE)
    valid_tfm = get_valid_transforms(config.IMG_SIZE)
    
    loaders_light = create_dataloaders(
        train_df, valid_df, dataset_name="eyepacs",
        train_transform=train_tfm_light, valid_transform=valid_tfm,
        batch_size=config.STAGE1_BATCH_SIZE
    )
    loaders_heavy = create_dataloaders(
        train_df, valid_df, dataset_name="eyepacs",
        train_transform=train_tfm_heavy, valid_transform=valid_tfm,
        batch_size=config.STAGE1_BATCH_SIZE
    )
    
    # Model & Trainer
    model = build_model(model_name=config.MODEL_NAME, pretrained=True, num_classes=config.NUM_CLASSES)
    trainer = Trainer(model, device)
    
    # Phase 1: Head Only
    freeze_backbone(model)
    logger.info("--- Stage 1 Phase 1: Head Only ---")
    trainer.fit_phase(
        phase="s1_phase1", train_loader=loaders_light["train"], valid_loader=loaders_light["valid"],
        epochs=config.STAGE1_PHASE1_EPOCHS, lr=config.STAGE1_PHASE1_LR, weight_decay=config.STAGE1_PHASE1_WD,
        use_mixup=False, use_amp=True
    )
    
    # Phase 2: Partial Unfreeze
    trainer.load_best_checkpoint(phase="s1_phase1")
    unfreeze_backbone(model, from_block=config.STAGE1_PHASE2_UNFREEZE)
    logger.info("--- Stage 1 Phase 2: Partial Unfreeze ---")
    trainer.fit_phase(
        phase="s1_phase2", train_loader=loaders_heavy["train"], valid_loader=loaders_heavy["valid"],
        epochs=config.STAGE1_PHASE2_EPOCHS, lr=config.STAGE1_PHASE2_LR, weight_decay=config.STAGE1_PHASE2_WD,
        use_mixup=getattr(config, "USE_MIXUP", True), use_amp=True
    )
    
    # Phase 3: Full Unfreeze
    trainer.load_best_checkpoint(phase="s1_phase2")
    unfreeze_backbone(model, from_block=None)
    logger.info("--- Stage 1 Phase 3: Full Unfreeze ---")
    trainer.fit_phase(
        phase="s1_phase3", train_loader=loaders_heavy["train"], valid_loader=loaders_heavy["valid"],
        epochs=config.STAGE1_PHASE3_EPOCHS, lr=config.STAGE1_PHASE3_LR, weight_decay=config.STAGE1_PHASE3_WD,
        use_mixup=getattr(config, "USE_MIXUP", True), use_amp=True
    )
    
    # We want the absolute best weights across all Stage 1.
    # We know phase3 is loaded at the end if it crashed, or we can just grab best_model_s1_phase3.pth
    final_best_path = os.path.join(config.CHECKPOINTS_DIR, "best_eyepacs_model.pth")
    phase3_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_model_s1_phase3.pth")
    
    if os.path.exists(phase3_ckpt):
        shutil.copy(phase3_ckpt, final_best_path)
    else:
        # Fallback to phase2 if phase3 failed instantly
        shutil.copy(os.path.join(config.CHECKPOINTS_DIR, "best_model_s1_phase2.pth"), final_best_path)
        
    logger.info(f"=== Stage 1 Complete. Saved master model to {final_best_path} ===")
    return final_best_path


def run_stage2_aptos(device, pretrained_weights_path):
    logger.info("=== Initializing Stage 2: Fine-Tuning on APTOS ===")
    
    # Load Data
    train_df, valid_df, test_df = load_and_split_dataframe(dataset_name="aptos")
    
    # Preprocess
    preprocess_images_to_cache(train_df, dataset_name="aptos")
    preprocess_images_to_cache(valid_df, dataset_name="aptos")
    if test_df is not None:
        preprocess_images_to_cache(test_df, dataset_name="aptos")
    
    # Transforms
    train_tfm = get_train_transforms(config.IMG_SIZE)
    valid_tfm = get_valid_transforms(config.IMG_SIZE)
    
    loaders = create_dataloaders(
        train_df, valid_df, dataset_name="aptos", test_df=test_df,
        train_transform=train_tfm, valid_transform=valid_tfm,
        batch_size=config.STAGE2_BATCH_SIZE
    )
    
    # Model
    model = build_model(model_name=config.MODEL_NAME, pretrained=False, num_classes=config.NUM_CLASSES)
    
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"Cannot run Stage 2. Stage 1 weights missing at {pretrained_weights_path}")
        
    logger.info(f"Loading Stage 1 weights into model: {pretrained_weights_path}")
    ckpt = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    trainer = Trainer(model, device)
    
    # Single Phase Full Fine-Tune (APTOS)
    unfreeze_backbone(model, from_block=None)
    logger.info("--- Stage 2: Full Fine-Tuning ---")
    trainer.fit_phase(
        phase="aptos_finetune", train_loader=loaders["train"], valid_loader=loaders["valid"],
        epochs=config.STAGE2_EPOCHS, lr=config.STAGE2_LR, weight_decay=config.STAGE2_WEIGHT_DECAY,
        warmup_epochs=config.STAGE2_WARMUP_EPOCHS,
        use_mixup=getattr(config, "USE_MIXUP", True), use_amp=True
    )
    
    trainer.load_best_checkpoint(phase="aptos_finetune")
    logger.info("=== Stage 2 Complete. Running Final Evaluation ===")
    
    if test_df is not None:
        # evaluate.py will run the scipy metric parameter optimisations inside full_evaluation automatically
        metrics = full_evaluation(trainer.model, loaders["test"], device)
        
        metrics_path = os.path.join(config.OUTPUTS_DIR, "final_aptos_metrics.json")
        
        # Unpack metrics correctly for JSON dump
        dump_metrics = {}
        for k, v in metrics.items():
            if type(v) is dict:
                dump_metrics[k] = v
            else:
                dump_metrics[k] = float(v) if not isinstance(v, str) else v
                
        with open(metrics_path, "w") as f:
            json.dump(dump_metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_path}")


def main():
    config.ensure_dirs()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Script starting. Target device: {device}")
    
    # 0. Download datasets if necessary
    ensure_aptos_dataset()
    
    # 1. Train on EyePACS
    best_eyepacs_weights = run_stage1_eyepacs(device)
    
    # 2. Reset Context
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    # 3. Fine-Tune on APTOS
    run_stage2_aptos(device, best_eyepacs_weights)


if __name__ == "__main__":
    main()
