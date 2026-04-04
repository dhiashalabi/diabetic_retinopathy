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

# EyePACS structure
_images_candidate = os.path.join(DATA_DIR, "resized_train_cropped")
_images_nested = os.path.join(_images_candidate, "resized_train_cropped")
EYEPACS_IMAGES_DIR = _images_nested if os.path.isdir(_images_nested) else _images_candidate
EYEPACS_LABELS_CSV = os.path.join(DATA_DIR, "trainLabels_cropped.csv")

# APTOS structure
APTOS_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_aptos")
APTOS_IMAGES_DIR = os.path.join(APTOS_DATA_DIR, "train_images")
APTOS_LABELS_CSV = os.path.join(APTOS_DATA_DIR, "train.csv")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# ---------------------------------------------------------------------------
# Reproducibility & Model
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 384
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
STOCHASTIC_DEPTH_RATE = 0.2

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------
USE_BEN_GRAHAM = True
USE_CIRCULAR_MASK = True

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# TRANSFER LEARNING - STAGE 1 (EyePACS Pre-training)
# ---------------------------------------------------------------------------
# We train progressively to handle the massive volume and noise.
STAGE1_PHASE1_EPOCHS = 10
STAGE1_PHASE1_LR = 1e-3
STAGE1_PHASE1_WD = 1e-4

STAGE1_PHASE2_EPOCHS = 15
STAGE1_PHASE2_LR = 1e-4
STAGE1_PHASE2_WD = 1e-4
STAGE1_PHASE2_UNFREEZE = "blocks[-2:]"

STAGE1_PHASE3_EPOCHS = 20
STAGE1_PHASE3_LR = 1e-5
STAGE1_PHASE3_WD = 5e-4
STAGE1_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# TRANSFER LEARNING - STAGE 2 (APTOS Fine-tuning)
# ---------------------------------------------------------------------------
# Tiny, highly accurate dataset. Very low LR, single phase full unfreeze.
STAGE2_EPOCHS = 25
STAGE2_LR = 1e-5           # Extremely low learning rate to not destroy Stage 1 features
STAGE2_WEIGHT_DECAY = 1e-3 # Strong regularization
STAGE2_BATCH_SIZE = 16
STAGE2_WARMUP_EPOCHS = 3

# ---------------------------------------------------------------------------
# Shared Advanced Strategy (Applies to both)
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 7

USE_DISCRIMINATIVE_LR = True
HEAD_LR_MULTIPLIER = 10.0

USE_WEIGHTED_SAMPLER = True
SAMPLER_POWER = 0.5  # Square-root rebalancing

USE_ORDINAL_LOSS = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

USE_MIXUP = True
MIXUP_ALPHA = 0.4
USE_CLASS_CONDITIONAL_MIXUP = True
MIXUP_SKIP_CLASSES = [3, 4]

USE_CONSISTENCY_REG = True
CONSISTENCY_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
if IS_COLAB:
    NUM_WORKERS = 2
else:
    _cpu_count = os.cpu_count() or 4
    NUM_WORKERS = min(8, max(2, _cpu_count // 2))

PREFETCH_FACTOR = 4
PIN_MEMORY = True

PREPROCESSING_CACHE_VERSION = "v5" # Force rebuild for unmerged dataset structure
CACHE_IMAGE_FORMAT = "png"
CACHE_IMAGE_QUALITY = 95
CACHE_PNG_COMPRESSION = 3

def ensure_dirs():
    for d in [DATA_DIR, APTOS_DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
