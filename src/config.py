"""
Central configuration for the Diabetic Retinopathy detection project.
All hyperparameters, paths, and constants are defined here.
"""

import os

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
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")

# Handle nested folder structure from Kaggle zip (resized_train_cropped/resized_train_cropped/)
_images_candidate = os.path.join(DATA_DIR, "resized_train_cropped")
_images_nested = os.path.join(_images_candidate, "resized_train_cropped")
IMAGES_DIR = _images_nested if os.path.isdir(_images_nested) else _images_candidate

# Outputs always on Drive (Colab) or project root so they persist
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

LABELS_CSV = os.path.join(DATA_DIR, "trainLabels_cropped.csv")
LABELS_CSV_FULL = os.path.join(DATA_DIR, "trainLabels.csv")

RESUME_STATE_PATH = os.path.join(CHECKPOINTS_DIR, "resume_state.pth")

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
CLASS_CODE = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Image & Model
# ---------------------------------------------------------------------------
IMG_SIZE = 384  # Larger than B3 native (300) for better small-lesion detection
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "efficientnet_b3"

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Training — Phase 1 (head only, frozen backbone)
# ---------------------------------------------------------------------------
PHASE1_EPOCHS = 8
PHASE1_LR = 3e-4
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Training — Phase 2 (partial unfreeze: last two timm blocks, see PHASE2_UNFREEZE_FROM)
# ---------------------------------------------------------------------------
PHASE2_EPOCHS = 15
PHASE2_LR = 5e-5
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_BATCH_SIZE = 32
PHASE2_WARMUP_EPOCHS = 2
PHASE2_UNFREEZE_FROM = "blocks[-2:]"  # timm EfficientNet uses blocks.N, not features.N

# ---------------------------------------------------------------------------
# Training — Phase 3 (full unfreeze)
# ---------------------------------------------------------------------------
PHASE3_EPOCHS = 25
# Slightly lower LR than Phase 2 — full backbone + AMP is prone to NaNs if too aggressive
PHASE3_LR = 1e-5
PHASE3_WEIGHT_DECAY = 1e-4
PHASE3_BATCH_SIZE = 32
PHASE3_WARMUP_EPOCHS = 2

# Mixup + full unfreeze can destabilize training; Phase 2 already used Mixup
USE_MIXUP_PHASE3 = False

# Full-backbone fine-tune in fp16 often overflows; Phase 3 in FP32 is slower but stable
USE_AMP_PHASE3 = False

# ---------------------------------------------------------------------------
# Shared training settings
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.05
DROPOUT_RATE = 0.3
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 7

# ---------------------------------------------------------------------------
# Class balancing strategy
# ---------------------------------------------------------------------------
# CRITICAL: use only ONE rebalancing mechanism to avoid over-weighting
# minority classes. WeightedRandomSampler is preferred because it balances
# the batches without distorting the loss magnitude.
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTED_LOSS = False

# ---------------------------------------------------------------------------
# Ordinal-aware training
# ---------------------------------------------------------------------------
USE_ORDINAL_LOSS = True

# ---------------------------------------------------------------------------
# Mixup augmentation (applied during Phase 2 and Phase 3 only)
# ---------------------------------------------------------------------------
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# DataLoader worker defaults differ by platform.
if IS_COLAB:
    NUM_WORKERS = 2
else:
    _cpu_count = os.cpu_count() or 4
    NUM_WORKERS = min(8, max(2, _cpu_count // 2)) if os.name == "nt" else min(16, _cpu_count)

PREFETCH_FACTOR = 4
PIN_MEMORY = True

# ---------------------------------------------------------------------------
# Preprocessing cache
# ---------------------------------------------------------------------------
PREPROCESSING_CACHE_VERSION = "v3"
CACHE_IMAGE_FORMAT = "png"  # supported: jpg, jpeg, png, webp
CACHE_IMAGE_QUALITY = 95
CACHE_PNG_COMPRESSION = 3
CACHE_FAIL_ON_MISSING = True


# ---------------------------------------------------------------------------
# Directory creation (call explicitly at runtime, not on import)
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required output directories. Call once at the start of each notebook."""
    for d in [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
