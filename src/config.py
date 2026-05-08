"""
Central configuration for the Diabetic Retinopathy detection project.

Two-stage transfer learning pipeline:
  Stage 1 — Pre-train EfficientNetV2-S on EyePACS (~35K images, noisy labels)
  Stage 2 — Fine-tune on APTOS 2019 (~3.6K images, clean labels)

All hyperparameters, paths, and constants are defined here.
"""

import os

import torch

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
try:
    from google.colab import drive  # noqa: F401

    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if IS_COLAB:
    PROJECT_ROOT = "/content/drive/MyDrive/diabetic_retinopathy"
    _LOCAL_DATA = "/content/dataset"
    DATA_DIR = _LOCAL_DATA if os.path.isdir(_LOCAL_DATA) else os.path.join(PROJECT_ROOT, "dataset")
    _LOCAL_APTOS = "/content/dataset_aptos"
    APTOS_DATA_DIR = (
        _LOCAL_APTOS if os.path.isdir(_LOCAL_APTOS) else os.path.join(PROJECT_ROOT, "dataset_aptos")
    )
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    APTOS_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_aptos")

# EyePACS structure (handles nested Kaggle zip extraction)
_images_candidate = os.path.join(DATA_DIR, "resized_train_cropped")
_images_nested = os.path.join(_images_candidate, "resized_train_cropped")
EYEPACS_IMAGES_DIR = _images_nested if os.path.isdir(_images_nested) else _images_candidate
EYEPACS_LABELS_CSV = os.path.join(DATA_DIR, "trainLabels_cropped.csv")
EYEPACS_CLEANED_LABELS_CSV = os.path.join(DATA_DIR, "trainLabels_cropped_cleaned.csv")

# APTOS structure
APTOS_IMAGES_DIR = os.path.join(APTOS_DATA_DIR, "train_images")
APTOS_LABELS_CSV = os.path.join(APTOS_DATA_DIR, "train.csv")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
KFOLD_CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "kfold_checkpoints")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
RUN_MANIFEST_PATH = os.path.join(LOGS_DIR, "run_manifest.json")

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# ---------------------------------------------------------------------------
# Reproducibility & Model
# ---------------------------------------------------------------------------
# Training defaults below are sized for: NVIDIA H200 (~141 GB VRAM, ~1979 TFLOPS
# peak tensor spec), 16 CPUs. Adjust batch sizes if your machine differs.
SEED = 42

# Per-stage input resolution (EyePACS pre-train vs APTOS fine-tune / deploy).
STAGE1_IMG_SIZE = 384
STAGE2_IMG_SIZE = 512
# Backward-compatible alias: same as Stage 2 (APTOS / ONNX / Gradio).
IMG_SIZE = STAGE2_IMG_SIZE


def resolve_img_size(dataset_name: str) -> int:
    """EyePACS uses Stage-1 resolution; APTOS uses Stage-2 (higher detail)."""
    if dataset_name == "aptos":
        return STAGE2_IMG_SIZE
    if dataset_name == "eyepacs":
        return STAGE1_IMG_SIZE
    raise ValueError(f"Unknown dataset_name for resolve_img_size: {dataset_name!r}")


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
STOCHASTIC_DEPTH_RATE = 0.2

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------
USE_BEN_GRAHAM = False  # Avoid stacking with CLAHE; keeps pixels closer to ImageNet stats
USE_CIRCULAR_MASK = True

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Preprocessing cache
# ---------------------------------------------------------------------------
USE_PREPROCESSED_CACHE = True
# Bump when cache layout, STAGE1/STAGE2 sizes, or preprocessing pipeline changes.
PREPROCESSING_CACHE_VERSION = "v8"
CACHE_IMAGE_FORMAT = "png"
CACHE_IMAGE_QUALITY = 95
CACHE_PNG_COMPRESSION = 3
CACHE_FAIL_ON_MISSING = False  # Graceful degradation during download

# ---------------------------------------------------------------------------
# TRANSFER LEARNING — STAGE 1 (EyePACS pre-training)
# ---------------------------------------------------------------------------
# Three-phase progressive unfreezing on the noisy, massive dataset.
STAGE1_PHASE1_EPOCHS = 8
STAGE1_PHASE1_LR = 1e-3
STAGE1_PHASE1_WD = 1e-4

STAGE1_PHASE2_EPOCHS = 12
STAGE1_PHASE2_LR = 5e-5
STAGE1_PHASE2_WD = 1e-4
STAGE1_PHASE2_UNFREEZE = "blocks[-1:]"

STAGE1_PHASE3_EPOCHS = 15
STAGE1_PHASE3_LR = 5e-5
STAGE1_PHASE3_WD = 5e-4

# Batch sizes tuned for NVIDIA H200 (~141 GB VRAM), EfficientNetV2-S @ 384×384, AMP.
# If you OOM (e.g. consistency reg + mixup spikes), lower STAGE1_BATCH_SIZE / STAGE2_BATCH_SIZE.
STAGE1_BATCH_SIZE = 512

# ---------------------------------------------------------------------------
# TRANSFER LEARNING — STAGE 2 (APTOS fine-tuning)
# ---------------------------------------------------------------------------
# Small, clean dataset. Very low LR, single-phase full unfreeze.
STAGE2_EPOCHS = 25
STAGE2_LR = 3e-5
STAGE2_WEIGHT_DECAY = 1e-3
# 512² inputs use more VRAM than 384²; reduce batch if you OOM (H200 can often raise this).
STAGE2_BATCH_SIZE = 128
STAGE2_WARMUP_EPOCHS = 3

# ---------------------------------------------------------------------------
# Advanced training strategy
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 7

# Discriminative learning rates
USE_DISCRIMINATIVE_LR = True
HEAD_LR_MULTIPLIER = 10.0

# Class balancing
USE_WEIGHTED_SAMPLER = True
SAMPLER_POWER = 1.0  # Inverse frequency rebalancing for massive mini-batches
USE_CLASS_WEIGHTED_LOSS = False  # DO NOT apply class weights to loss (prevents gradient explosion when combined with WeightedSampler)

# Loss configuration
USE_ORDINAL_LOSS = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# Mixup
USE_MIXUP = True
MIXUP_ALPHA = 0.4
USE_CLASS_CONDITIONAL_MIXUP = True
MIXUP_SKIP_CLASSES = [3, 4]  # Don't mix Severe/Proliferative

# Consistency regularization
USE_CONSISTENCY_REG = True
CONSISTENCY_WEIGHT = 0.5

# Exponential Moving Average
USE_EMA = True
EMA_DECAY = 0.999  # Lowered from 0.9999 to account for massive 512 batch size
STAGE2_EMA_DECAY = 0.99

# ---------------------------------------------------------------------------
# K-Fold Ensemble (APTOS fine-tuning)
# ---------------------------------------------------------------------------
KFOLD_N_SPLITS = 5
KFOLD_SEED = SEED  # Reproducible fold assignments

# ---------------------------------------------------------------------------
# EyePACS noisy-label cleaning
# ---------------------------------------------------------------------------
# Remove EyePACS samples where |model_prediction - label| >= threshold.
# Conservative threshold of 2 avoids removing legitimate borderline cases.
EYEPACS_CLEANING_THRESHOLD = 2
# Auto-detection: if EYEPACS_CLEANED_LABELS_CSV exists on disk, it is used
# automatically by load_and_split_dataframe(). No manual flag needed.

# ---------------------------------------------------------------------------
# Multi-scale TTA
# ---------------------------------------------------------------------------
# Additional resolutions to average alongside the primary STAGE2_IMG_SIZE.
# Retinal lesions appear at different scales; multi-resolution averaging helps.
MULTISCALE_TTA_SIZES = [480, STAGE2_IMG_SIZE, 544]

# ---------------------------------------------------------------------------
# Model comparison (notebook 06)
# ---------------------------------------------------------------------------
COMPARISON_MODELS = {
    "EfficientNetV2-S": "tf_efficientnetv2_s.in21k_ft_in1k",
    "ConvNeXt-Base": "convnext_base.fb_in22k_ft_in1k",
    "EfficientNetV2-M": "tf_efficientnetv2_m.in21k_ft_in1k",
}

# ---------------------------------------------------------------------------
# System (defaults assume a dedicated training box: H200-class GPU, ~16 CPUs)
# ---------------------------------------------------------------------------
if IS_COLAB:
    NUM_WORKERS = 2
else:
    _cpu_count = os.cpu_count() or 16
    # Leave 2+ cores for the main process / OS; cap to avoid excessive IPC churn
    NUM_WORKERS = min(14, _cpu_count - 2) if _cpu_count >= 16 else min(8, max(2, _cpu_count // 2))

PREFETCH_FACTOR = 2
PIN_MEMORY = True

# Optional: torch.compile on CUDA (Hopper H100/H200, Ada, Ampere+); first epoch may compile
USE_TORCH_COMPILE = True

# AMP: "bfloat16" on CUDA when supported (H200/Hopper; no GradScaler). Else float16 + GradScaler.
# Override: DR_AMP_DTYPE=float16
AMP_DTYPE = os.environ.get("DR_AMP_DTYPE", "bfloat16").lower()


def resolve_amp(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """
    Returns (autocast_dtype, use_grad_scaler).

    CPU: (None, False). CUDA bf16 when supported: (bfloat16, False).
    CUDA otherwise: (float16, True).
    """
    if device.type != "cuda":
        return None, False
    want_bf16 = AMP_DTYPE in ("bf16", "bfloat16")
    want_fp16 = AMP_DTYPE in ("fp16", "float16")
    if want_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    if want_fp16 or want_bf16:
        return torch.float16, True
    return torch.float16, True


def ensure_dirs():
    """Create all required output directories."""
    for d in [
        DATA_DIR,
        APTOS_DATA_DIR,
        OUTPUTS_DIR,
        CHECKPOINTS_DIR,
        KFOLD_CHECKPOINTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
