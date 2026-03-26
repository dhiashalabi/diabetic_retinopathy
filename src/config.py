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
IMG_SIZE = 380  # EfficientNet-B4 native resolution
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "efficientnet_b4"

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Training — Phase 1 (head only)
# ---------------------------------------------------------------------------
PHASE1_EPOCHS = 10
PHASE1_LR = 4e-3
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Training — Phase 2 (fine-tuning)
# ---------------------------------------------------------------------------
PHASE2_EPOCHS = 25
PHASE2_LR = 4e-5
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_BATCH_SIZE = 64
PHASE2_WARMUP_EPOCHS = 3
PHASE2_UNFREEZE_FROM = "block5"

# ---------------------------------------------------------------------------
# Shared training settings
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 2 if IS_COLAB else min(16, os.cpu_count() or 4)
PREFETCH_FACTOR = 4
PIN_MEMORY = True


# ---------------------------------------------------------------------------
# Directory creation (call explicitly at runtime, not on import)
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required output directories. Call once at the start of each notebook."""
    for d in [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
