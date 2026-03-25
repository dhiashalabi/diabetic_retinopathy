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
│   └── resized_train_cropped/         # Cropped fundus images
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
│   ├── trainer.py                     # Training engine with AMP
│   ├── evaluate.py                    # Metrics and visualisation
│   ├── gradcam.py                     # Grad-CAM implementation
│   └── app.py                         # Gradio web demo module
├── outputs/
│   ├── checkpoints/                   # Model weights
│   ├── figures/                       # Plots and visualisations
│   └── logs/                          # Training history
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### Google Colab (recommended)

1. Upload this project to Google Drive at `MyDrive/diabetic_retinopathy/`
2. Place the dataset inside `MyDrive/diabetic_retinopathy/dataset/`
3. Open any notebook -- each has a setup cell with a `DATA_SOURCE` toggle:
   - `"kaggle"` -- downloads directly from Kaggle to Colab's local SSD (fastest, recommended)
   - `"drive"` -- copies from Google Drive to local SSD
   - `"local"` -- data already in place, no copy needed

### Local

```bash
pip install -r requirements.txt
```

Place the dataset images under `dataset/resized_train_cropped/` relative to the project root.

## Usage

Run the notebooks in order:

1. **`01_eda.ipynb`** -- Explore the dataset, understand class distribution
2. **`02_preprocessing.ipynb`** -- Visualise the preprocessing pipeline
3. **`03_training.ipynb`** -- Train the model (Phase 1 + Phase 2)
4. **`04_evaluation.ipynb`** -- Evaluate on the held-out test set
5. **`05_gradcam.ipynb`** -- Generate Grad-CAM explanations
6. **`06_model_comparison.ipynb`** -- Benchmark EfficientNet-B4 vs B3 vs ResNet-50
7. **`07_demo.ipynb`** -- Launch the Gradio web demo

## Technology Stack

- **PyTorch** -- deep learning framework
- **timm** -- pretrained EfficientNet-B4
- **Albumentations** -- fast image augmentation
- **OpenCV** -- image preprocessing (CLAHE, Ben Graham)
- **scikit-learn** -- metrics, splitting, class weighting
- **Matplotlib / Seaborn** -- visualisation
- **Gradio** -- interactive web demo
