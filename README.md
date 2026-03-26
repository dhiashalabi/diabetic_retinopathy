# Diabetic Retinopathy Detection using Deep Learning

Automated detection and grading of Diabetic Retinopathy (DR) from retinal fundus images using EfficientNet-B4 with PyTorch.

## Problem

Diabetic Retinopathy is the leading cause of preventable blindness. Early detection through automated screening can significantly reduce vision loss. This project classifies retinal fundus images into 5 severity levels:

| Grade | Name | Description |
|-------|------|-------------|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than just microaneurysms |
| 3 | Severe | Extensive intraretinal haemorrhages |
| 4 | Proliferative DR | Neovascularisation or vitreous/preretinal haemorrhage |

## Dataset

- ~35,000 retinal fundus images from the [Kaggle Diabetic Retinopathy (resized)](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) dataset
- Two CSV label files: `trainLabels_cropped.csv` (primary, for cropped images) and `trainLabels.csv` (full set)
- **Severe class imbalance**: ~73% No DR, ~5% Mild, ~15% Moderate, ~2% Severe, ~5% Proliferative

## Key Design Decisions

### High Accuracy Strategies
- **EfficientNet-B4** backbone (ImageNet top-1 ~83%) with custom two-layer head
- **Two-phase training**: head-only warm-up, then backbone fine-tuning with cosine warmup
- **Ben Graham preprocessing**: colour normalisation that won Kaggle DR 2015
- **CLAHE enhancement**: boosts local contrast to reveal microaneurysms and haemorrhages
- **Test-Time Augmentation (TTA)**: average predictions over 4 geometric transforms
- **Label smoothing** (0.1): prevents overconfident predictions

### Professional Dataset Balancing
- **Patient-level stratified splitting**: prevents data leakage (both eyes stay in the same split)
- **WeightedRandomSampler**: produces balanced mini-batches despite 73%/5%/15%/2%/5% imbalance
- **Class-weighted loss**: higher penalty for misclassifying rare severe/proliferative cases
- **Aggressive augmentation**: heavy geometric and photometric transforms for regularisation

### Clinical-Grade Evaluation
- **Quadratic Weighted Kappa (QWK)**: primary metric, penalises distant misclassifications more heavily
- **Per-class ROC curves** with AUC
- **Grad-CAM explainability**: visual explanations for clinical trust

## Project Structure

```
diabetic_retinopathy/
├── dataset/                           # CSV labels + image directories
│   ├── trainLabels_cropped.csv        # Primary labels (cropped images)
│   ├── trainLabels.csv                # Full labels
│   └── resized_train_cropped/         # Cropped fundus images (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_preprocessing.ipynb         # Preprocessing pipeline demo
│   ├── 03_training.ipynb              # Model training (Phase 1 + 2)
│   ├── 04_evaluation.ipynb            # Full evaluation suite
│   ├── 05_gradcam.ipynb               # Grad-CAM explainability
│   ├── 06_model_comparison.ipynb      # EfficientNet-B4 vs B3 vs ResNet-50
│   └── 07_demo.ipynb                  # Gradio web demo
├── src/
│   ├── __init__.py                    # Package init
│   ├── config.py                      # Hyperparameters and paths
│   ├── utils.py                       # Seed, device, logging
│   ├── preprocessing.py               # CLAHE, Ben Graham, crop
│   ├── augmentations.py               # Albumentations pipelines
│   ├── dataset.py                     # Dataset, splits, balancing
│   ├── model.py                       # EfficientNet-B4 builder
│   ├── trainer.py                     # Training engine with AMP + resume
│   ├── evaluate.py                    # Metrics and visualisation
│   ├── gradcam.py                     # Grad-CAM implementation
│   ├── export.py                      # ONNX export for production
│   ├── data_manager.py                # Unified dataset download/setup
│   └── app.py                         # Gradio web demo module
├── scripts/
│   └── download_data.py               # CLI dataset downloader
├── outputs/
│   ├── checkpoints/                   # Model weights (.pth + .onnx)
│   ├── figures/                       # Plots and visualisations
│   └── logs/                          # Training history
├── pyproject.toml                     # Project metadata, deps, ruff config
├── .python-version                    # Python version pin
├── .gitignore
└── README.md
```

## Setup

### Option 1: Local / GPU Server (with uv — recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download dataset from Kaggle
# Option A: Set env vars (recommended for servers)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
uv run scripts/download_data.py

# Option B: Use ~/.kaggle/kaggle.json
uv run scripts/download_data.py

# Option C: Custom data directory
uv run scripts/download_data.py --data-dir /data/dr_dataset

# Run notebooks
uv run jupyter notebook notebooks/
```

### Option 2: Local / GPU Server (with pip)

```bash
pip install -e .
python scripts/download_data.py
jupyter notebook notebooks/
```

### Option 3: Google Colab

1. Upload this project to Google Drive at `MyDrive/diabetic_retinopathy/`
2. Open any notebook — each has a setup cell that automatically:
   - Mounts Google Drive
   - Installs dependencies
   - Downloads the dataset from Kaggle (you'll be prompted for `kaggle.json`)

### GPU Server Quick Start (RunPod / DigitalOcean / Lambda)

```bash
git clone <your-repo-url> && cd diabetic_retinopathy
curl -LsSf https://astral.sh/uv/install.sh | sh
export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy
uv sync
uv run scripts/download_data.py
uv run jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
```

## Usage

Run the notebooks in order:

1. **`01_eda.ipynb`** — Explore the dataset, understand class distribution
2. **`02_preprocessing.ipynb`** — Visualise the preprocessing pipeline
3. **`03_training.ipynb`** — Train the model (Phase 1 + Phase 2), exports to ONNX
4. **`04_evaluation.ipynb`** — Evaluate on the held-out test set
5. **`05_gradcam.ipynb`** — Generate Grad-CAM explanations
6. **`06_model_comparison.ipynb`** — Benchmark EfficientNet-B4 vs B3 vs ResNet-50
7. **`07_demo.ipynb`** — Launch the Gradio web demo

## Development

### Linting and Formatting (ruff)

```bash
# Check for issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/
```

## Technology Stack

- **PyTorch** — deep learning framework
- **timm** — pretrained EfficientNet-B4
- **Albumentations** — fast image augmentation
- **OpenCV** — image preprocessing (CLAHE, Ben Graham)
- **scikit-learn** — metrics, splitting, class weighting
- **Matplotlib / Seaborn** — visualisation
- **Gradio** — interactive web demo
- **ONNX / ONNXRuntime** — production model export
- **uv** — fast Python package manager
- **ruff** — Python linter and formatter
