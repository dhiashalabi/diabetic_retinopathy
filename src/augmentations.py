"""
Albumentations transform pipelines for training and evaluation.

Training uses heavy, ophthalmology-appropriate augmentations.
Validation / test uses only deterministic resize + normalise.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config


def get_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """
    Aggressive augmentation for training.

    Retinal images are rotation-invariant (camera orientation varies),
    so flips and large rotations are safe.  Brightness / contrast jitter
    simulates different imaging devices.  CoarseDropout acts as CutOut
    regularisation to prevent overfitting on small lesion patterns.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15,
            rotate_limit=30, border_mode=0, p=0.5,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=5),
            A.GaussNoise(std_range=(0.04, 0.2)),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
        ], p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
        ], p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 24, img_size // 12),
            hole_width_range=(img_size // 24, img_size // 12),
            fill=0, p=0.3,
        ),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """Deterministic pipeline for validation and test sets."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = config.IMG_SIZE) -> list[A.Compose]:
    """
    Test-Time Augmentation (TTA) transforms for boosting accuracy.

    Returns a list of transform pipelines.  Predictions from all
    pipelines are averaged to produce the final output.
    """
    base_norm = [
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ]

    return [
        # Original
        A.Compose([A.Resize(img_size, img_size)] + base_norm),
        # Horizontal flip
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0)] + base_norm),
        # Vertical flip
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0)] + base_norm),
        # 90-degree rotation
        A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=(90, 90), p=1.0, border_mode=0),
        ] + base_norm),
    ]
