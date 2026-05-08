# Diabetic Retinopathy Detection using Deep Learning

Automated detection and grading of Diabetic Retinopathy (DR) from retinal fundus images using EfficientNetV2-S with two-stage transfer learning in PyTorch.

## Problem

Diabetic Retinopathy is the leading cause of preventable blindness. Early detection through automated screening can significantly reduce vision loss. This project classifies retinal fundus images into 5 severity levels:

| Grade | Name             | Description                                           |
| ----- | ---------------- | ----------------------------------------------------- |
| 0     | No DR            | No signs of diabetic retinopathy                      |
| 1     | Mild             | Microaneurysms only                                   |
| 2     | Moderate         | More than just microaneurysms                         |
| 3     | Severe           | Extensive intraretinal haemorrhages                   |
| 4     | Proliferative DR | Neovascularisation or vitreous/preretinal haemorrhage |

## Dataset

- **Stage 1 (EyePACS)**: ~35,000 retinal fundus images from [Kaggle Diabetic Retinopathy (resized)](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) — noisy labels, massive scale
- **Stage 2 (APTOS 2019)**: ~3,600 images from [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) — clean labels, high quality
- **Severe class imbalance**: ~73% No DR, ~5% Mild, ~15% Moderate, ~2% Severe, ~5% Proliferative

## Key Design Decisions

### High Accuracy Strategies

- **EfficientNetV2-S** backbone (ImageNet-21K pre-trained) with GeM pooling and custom two-layer head
- **Two-stage transfer learning**: pre-train on EyePACS (3-phase progressive unfreezing), fine-tune on APTOS
- **CLAHE enhancement**: boosts local contrast to reveal microaneurysms and haemorrhages
- **Combined loss**: Ordinal cross-entropy + Focal loss (aligned with QWK, focuses on hard cases)
- **Exponential Moving Average (EMA)**: stable model weights for better generalisation
- **Test-Time Augmentation (TTA)**: average predictions over geometric transforms
- **Label smoothing** (0.1): prevents overconfident predictions

### Professional Dataset Balancing

- **Patient-level stratified splitting**: prevents data leakage (both eyes stay in the same split)
- **WeightedRandomSampler**: produces balanced mini-batches despite 73%/5%/15%/2%/5% imbalance
- **Class-conditional Mixup**: linear interpolation while protecting rare/severe class features
- **Consistency regularisation**: KL-divergence between differently augmented views
- **Phase-aware augmentation**: light transforms for head-only training, full augmentation for fine-tuning

### Clinical-Grade Evaluation

- **Quadratic Weighted Kappa (QWK)**: primary metric, penalises distant misclassifications more heavily
- **Threshold optimisation**: learn optimal thresholds on validation set for improved grading
- **Per-class ROC curves** with AUC
- **Grad-CAM explainability**: visual explanations for clinical trust

## Project Structure

```
diabetic_retinopathy/
├── dataset/                           # EyePACS: CSV labels + image directories
│   ├── trainLabels_cropped.csv        # Primary labels (cropped images)
│   ├── trainLabels.csv                # Full labels
│   └── resized_train_cropped/         # Cropped fundus images (gitignored)
├── dataset_aptos/                     # APTOS 2019: CSV labels + images
│   ├── train.csv                      # Labels (id_code, diagnosis)
│   └── train_images/                  # Fundus images (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis (EyePACS + APTOS)
│   ├── 02_preprocessing.ipynb         # Preprocessing pipeline demo
│   ├── 03_training.ipynb              # Stage 1: EyePACS pre-training (3-phase)
│   ├── 03b_aptos_finetuning.ipynb     # Stage 2: APTOS fine-tuning (single model)
│   ├── 04_evaluation.ipynb            # Full evaluation suite + TTA + thresholds
│   ├── 05_gradcam.ipynb               # Grad-CAM explainability (APTOS images)
│   ├── 06_model_comparison.ipynb      # EfficientNetV2-S vs ConvNeXt-Base vs V2-M
│   ├── 07_demo.ipynb                  # Gradio web demo
│   └── 08_aptos_kfold_ensemble.ipynb  # 5-fold ensemble fine-tuning + evaluation
├── src/
│   ├── __init__.py                    # Package init
│   ├── config.py                      # Hyperparameters and paths
│   ├── utils.py                       # Seed, device, logging, GPU monitoring
│   ├── preprocessing.py               # CLAHE, crop, Ben Graham
│   ├── augmentations.py               # Albumentations pipelines + Mixup
│   ├── dataset.py                     # Dataset, splits, balancing, cache
│   ├── losses.py                      # Ordinal CE, Focal, Combined loss
│   ├── model.py                       # EfficientNetV2-S builder + EMA
│   ├── trainer.py                     # Training engine with AMP + EMA
│   ├── evaluate.py                    # Metrics, threshold opt, TTA
│   ├── gradcam.py                     # Grad-CAM implementation
│   ├── export.py                      # ONNX export for production
│   ├── data_manager.py                # Unified dataset download/setup
│   └── app.py                         # Gradio web demo module
├── scripts/
│   ├── download_data.py               # CLI dataset downloader
│   ├── clean_eyepacs_labels.py        # Noisy EyePACS label cleaning
│   └── train_kfold.py                 # CLI K-fold ensemble training
├── outputs/
│   ├── checkpoints/                   # Model weights (.pth + .onnx)
│   ├── figures/                       # Plots and visualisations
│   └── logs/                          # Training history + `run_manifest.json`
├── train_pipeline.py                  # Master 2-stage training script
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

# Download datasets from Kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
uv run scripts/download_data.py

# Run the full training pipeline
Run the notebooks/03_training.ipynb followed by 03b_aptos_finetuning.ipynb in Jupyter.

# Or quick test (1-2 epochs per phase)
uv run python train_pipeline.py --dry-run

# Final evaluation with test-time augmentation (slower)
uv run python train_pipeline.py --tta

# Run notebooks (recommended primary workflow)
uv run jupyter notebook notebooks/
```

### Notebook-first workflow and run manifest

The intended path is **`notebooks/03_training.ipynb` → `03b_aptos_finetuning.ipynb` → `04_evaluation.ipynb`**. `train_pipeline.py` stays available for a single CLI run and writes the same **`outputs/logs/run_manifest.json`** (merged JSON: stages, `AMP_DTYPE`, checkpoints, optional eval summary).

- **`config.AMP_DTYPE`**: default **bfloat16** on CUDA when supported (Trainer uses autocast with that dtype and skips GradScaler). Override with **`DR_AMP_DTYPE=float16`** for FP16 + GradScaler.
- **`full_evaluation(..., use_tta=True, valid_df=..., test_df=..., dataset_name="aptos")`**: validation and test predictions use TTA; QWK thresholds are fit on **validation TTA** probabilities only, then applied to **test TTA** (no test-label leakage). When `use_tta=False`, behaviour matches the previous API (pass `test_loader`; optional `valid_loader` for val-fit thresholds on single-view predictions).
- **`write_run_manifest(path, payload, merge=False)`** in `src/utils.py`: notebooks and the CLI merge stage/eval fields into the manifest path above.

### Option 2: Local / GPU Server (with pip)

```bash
pip install -e .
python scripts/download_data.py
Run the notebooks/03_training.ipynb followed by 03b_aptos_finetuning.ipynb in Jupyter.
jupyter notebook notebooks/
```

### Option 3: Google Colab

1. Upload this project to Google Drive at `MyDrive/diabetic_retinopathy/`
2. Open any notebook — each has a setup cell that automatically:
   - Mounts Google Drive
   - Installs dependencies
   - Downloads the dataset from Kaggle (you'll be prompted for `kaggle.json`)

### GPU Server Quick Start (RunPod / DigitalOcean / Lambda)

Defaults in `src/config.py` assume a **single NVIDIA H200** (~141 GB VRAM, Hopper-class) and **~16 CPU cores** (`STAGE1_BATCH_SIZE=96`, `STAGE2_BATCH_SIZE=64`, up to 14 dataloader workers). On smaller GPUs (for example 24 GB), reduce those batch sizes before training.

```bash
git clone <your-repo-url> && cd diabetic_retinopathy
curl -LsSf https://astral.sh/uv/install.sh | sh
export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy
uv sync
uv run scripts/download_data.py
Run the notebooks/03_training.ipynb followed by 03b_aptos_finetuning.ipynb in Jupyter.
```

## Usage

### Full Training Pipeline

```bash
python train_pipeline.py               # Full 2-stage pipeline
python train_pipeline.py --dry-run     # Quick test (1-2 epochs)
python train_pipeline.py --stage2-only # Skip Stage 1 (needs existing checkpoint)
```

### Complete Training Workflow

Follow these steps in order for optimal results:

#### Phase 1: Exploration & Initial Training

1. **`01_eda.ipynb`** — Explore both datasets (EyePACS + APTOS)
2. **`02_preprocessing.ipynb`** — Visualise the preprocessing pipeline
3. **`03_training.ipynb`** — Stage 1: EyePACS pre-training (3-phase progressive unfreezing)
4. **`03b_aptos_finetuning.ipynb`** — Stage 2: APTOS fine-tuning (single model) + threshold optimization
5. **`04_evaluation.ipynb`** — Full evaluation + TTA comparison + threshold optimization

#### Phase 2: Data Cleaning (optional but recommended)

6. **`scripts/clean_eyepacs_labels.py`** — Clean noisy EyePACS labels using the trained model

   ```bash
   python scripts/clean_eyepacs_labels.py
   ```

   The cleaned CSV is **auto-detected** — no config change needed.

7. **Re-run `03_training.ipynb`** — Retrain Stage 1 on cleaned EyePACS data

#### Phase 3: K-Fold Ensemble (for maximum QWK)

8. **`08_aptos_kfold_ensemble.ipynb`** — 5-fold stratified ensemble fine-tuning + evaluation
   - Trains 5 models on 5 different APTOS splits, starting from the Stage 1 checkpoint
   - Averages predictions across folds with multi-scale TTA + threshold optimization

#### Phase 4: Analysis & Demo

9. **`05_gradcam.ipynb`** — Grad-CAM explanations on APTOS images
10. **`06_model_comparison.ipynb`** — Benchmark EfficientNetV2-S vs ConvNeXt-Base vs V2-M
11. **`07_demo.ipynb`** — Launch the Gradio web demo

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
- **timm** — pretrained EfficientNetV2-S / ConvNeXt-Base / V2-M
- **Albumentations** — fast image augmentation
- **OpenCV** — image preprocessing (CLAHE, Ben Graham)
- **scikit-learn** — metrics, splitting, class weighting
- **scipy** — threshold optimisation
- **Matplotlib / Seaborn** — visualisation
- **Gradio** — interactive web demo
- **ONNX / ONNXRuntime** — production model export
- **uv** — fast Python package manager
- **ruff** — Python linter and formatter
