"""
Albumentations transform pipelines for training and evaluation.

Provides three training pipelines:
- Light transforms (Phase 1, head-only): flips + mild colour jitter
- Full transforms (Phase 2/3, fine-tuning): aggressive geometric + photometric
- Validation / test: deterministic resize + normalise

Also includes Mixup utility for batch-level augmentation.
"""

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from . import config


def get_light_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """
    Mild augmentation for Phase 1 (head-only training).

    When the backbone is frozen, heavy distortions add noise to the
    feature vectors without benefit. Light augmentations let the head
    learn clean decision boundaries first.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3,
            ),
            A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
            ToTensorV2(),
        ]
    )


# Backward-compatible alias (train_pipeline.py may import either name)
get_train_transforms_light = get_light_train_transforms


def get_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """
    Aggressive augmentation for Phase 2/3 fine-tuning.

    Retinal images are rotation-invariant (camera orientation varies),
    so flips and large rotations are safe.  Brightness / contrast jitter
    simulates different imaging devices.  CoarseDropout acts as CutOut
    regularisation to prevent overfitting on small lesion patterns.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-30, 30),
                border_mode=0,
                fill=0,
                keep_ratio=True,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MedianBlur(blur_limit=5),
                    A.GaussNoise(std_range=(0.04, 0.2)),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.3, p=1.0),
                ],
                p=0.15,
            ),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(img_size // 24, img_size // 12),
                hole_width_range=(img_size // 24, img_size // 12),
                fill=0,
                p=0.5,
            ),
            A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """Deterministic pipeline for validation and test sets."""
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
            ToTensorV2(),
        ]
    )


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
        A.Compose([A.Resize(img_size, img_size)] + base_norm),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0)] + base_norm),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0)] + base_norm),
        A.Compose(
            [A.Resize(img_size, img_size), A.Rotate(limit=(90, 90), p=1.0, border_mode=0)]
            + base_norm
        ),
        A.Compose(
            [A.Resize(img_size, img_size), A.Rotate(limit=(180, 180), p=1.0, border_mode=0)]
            + base_norm
        ),
        A.Compose(
            [A.Resize(img_size, img_size), A.Rotate(limit=(270, 270), p=1.0, border_mode=0)]
            + base_norm
        ),
        A.Compose(
            [A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
            + base_norm
        ),
    ]


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = config.MIXUP_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to a batch.

    Blends pairs of images and returns both label sets so the loss can
    be computed as: loss = lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)

    Returns (mixed_images, labels_a, labels_b, lam).
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def mixup_data_class_conditional(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = config.MIXUP_ALPHA,
    skip_classes: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | torch.Tensor]:
    """
    Apply Mixup but SKIP samples from rare/severe classes to avoid blurring
    subtle disease features that distinguish those grades.
    """
    if skip_classes is None:
        skip_classes = config.MIXUP_SKIP_CLASSES

    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1 - lam) * images[index]

    # Create a mask for samples that should NOT be mixed
    mask = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
    for c in skip_classes:
        mask |= labels == c
        mask |= labels[index] == c

    # For masked samples, use original image and target
    mixed[mask] = images[mask]
    lam_tensor = torch.full((batch_size,), lam, device=images.device)
    lam_tensor[mask] = 1.0

    return mixed, labels, labels[index], lam_tensor


def consistency_augment(images: torch.Tensor) -> torch.Tensor:
    """
    Generate a weakly-augmented view of a batch directly on the GPU.
    Used for consistency regularization (KL-divergence penalty).
    Applies mild random crops and horizontal flips.

    NOTE: Returns a new tensor, does NOT modify the input.
    """
    aug_images = images.clone()

    # Random Horizontal Flip (p=0.5)
    b = images.shape[0]
    flip_mask = torch.rand(b, device=images.device) < 0.5
    if flip_mask.any():
        aug_images[flip_mask] = torch.flip(aug_images[flip_mask], [-1])

    # Mild intensity shift
    noise = torch.randn_like(aug_images) * 0.05
    aug_images = aug_images + noise

    return aug_images
