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
# Paths — EyePACS (primary dataset)
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

# ---------------------------------------------------------------------------
# Paths — APTOS 2019 (secondary dataset, merged with EyePACS)
# ---------------------------------------------------------------------------
APTOS_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_aptos")
APTOS_IMAGES_DIR = os.path.join(APTOS_DATA_DIR, "train_images")
APTOS_LABELS_CSV = os.path.join(APTOS_DATA_DIR, "train.csv")
USE_APTOS_DATASET = True  # set False to train on EyePACS only

# ---------------------------------------------------------------------------
# Paths — Outputs
# ---------------------------------------------------------------------------
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
IMG_SIZE = 384  # Matches EfficientNetV2-S native resolution
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"  # EfficientNetV2-S pretrained on ImageNet-21K

# Stochastic depth (drop-path) regularisation applied inside the backbone
STOCHASTIC_DEPTH_RATE = 0.2

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------
USE_BEN_GRAHAM = True       # Ben Graham colour normalisation (Kaggle DR 2015 winner technique)
USE_CIRCULAR_MASK = True    # Removes edge artefacts from fundus cropping

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Training — Phase 1 (head only, frozen backbone)
# ---------------------------------------------------------------------------
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Training — Phase 2 (partial unfreeze: last backbone block)
# ---------------------------------------------------------------------------
PHASE2_EPOCHS = 25
PHASE2_LR = 3e-5
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_BATCH_SIZE = 16
PHASE2_WARMUP_EPOCHS = 3
PHASE2_UNFREEZE_FROM = "blocks[-1:]"  # only last block — reduces transition shock

# ---------------------------------------------------------------------------
# Training — Phase 3 (full unfreeze)
# ---------------------------------------------------------------------------
PHASE3_EPOCHS = 40
PHASE3_LR = 1e-5
PHASE3_WEIGHT_DECAY = 5e-4  # stronger L2 when all backbone params are trainable
PHASE3_BATCH_SIZE = 16
PHASE3_WARMUP_EPOCHS = 3

# Mixup enabled in Phase 3 — acts as regulariser during full unfreeze
USE_MIXUP_PHASE3 = True

# Full-backbone fine-tune in fp16 often overflows; Phase 3 in FP32 is slower but stable
USE_AMP_PHASE3 = False

# ---------------------------------------------------------------------------
# Shared training settings
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 10

# ---------------------------------------------------------------------------
# Discriminative learning rates
# ---------------------------------------------------------------------------
USE_DISCRIMINATIVE_LR = True  # backbone gets base LR, head gets LR * multiplier
HEAD_LR_MULTIPLIER = 10.0

# ---------------------------------------------------------------------------
# Class balancing strategy
# ---------------------------------------------------------------------------
# CRITICAL: use only ONE rebalancing mechanism to avoid over-weighting
# minority classes.  WeightedRandomSampler is preferred because it balances
# the batches without distorting the loss magnitude.
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTED_LOSS = False
SAMPLER_POWER = 0.5  # sqrt rebalancing — gentler than full inverse-frequency (1.0)

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
USE_ORDINAL_LOSS = True
USE_FOCAL_LOSS = True       # focal loss for hard example mining
FOCAL_GAMMA = 2.0           # focusing parameter — down-weights easy No_DR examples
FOCAL_ALPHA = None          # None = auto-compute from class frequencies

# ---------------------------------------------------------------------------
# Mixup augmentation
# ---------------------------------------------------------------------------
USE_MIXUP = True
MIXUP_ALPHA = 0.4

# Class-conditional Mixup — protects rare severe classes from blurring
USE_CLASS_CONDITIONAL_MIXUP = True
MIXUP_SKIP_CLASSES = [3, 4]  # Severe + Proliferative skip Mixup (too few samples)

# ---------------------------------------------------------------------------
# Consistency regularisation (mitigates label noise)
# ---------------------------------------------------------------------------
USE_CONSISTENCY_REG = True
CONSISTENCY_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
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
PREPROCESSING_CACHE_VERSION = "v4"  # bumped: new preprocessing defaults + APTOS
CACHE_IMAGE_FORMAT = "png"
CACHE_IMAGE_QUALITY = 95
CACHE_PNG_COMPRESSION = 3
CACHE_FAIL_ON_MISSING = True


# ---------------------------------------------------------------------------
# Directory creation (call explicitly at runtime, not on import)
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required output directories. Call once at the start of each notebook."""
    for d in [DATA_DIR, APTOS_DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
