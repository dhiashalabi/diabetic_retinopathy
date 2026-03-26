"""
PyTorch Dataset, patient-level splitting, and DataLoader factory
with professional-grade class balancing.

Balancing strategy (multi-pronged):
  1. Patient-level stratified split — no data leakage
  2. WeightedRandomSampler — balanced mini-batches
  3. Class-weighted loss — higher penalty for minority classes
  4. Heavy augmentation on minority classes (via augmentations.py)
"""

import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .preprocessing import preprocess_image
from .utils import get_logger

_logger = get_logger(__name__)
_EXTENSIONS = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG")


# -----------------------------------------------------------------------
# Patient-level stratified split
# -----------------------------------------------------------------------

def load_and_split_dataframe(
    csv_path: str = config.LABELS_CSV,
    train_ratio: float = config.TRAIN_RATIO,
    valid_ratio: float = config.VALID_RATIO,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load labels CSV, extract patient IDs, and split at the patient level.

    Both eyes of the same patient always land in the same split, preventing
    data leakage.  Stratification is on the patient's *worst* diagnosis
    so that rare severe cases are represented in every split.

    Returns (train_df, valid_df, test_df) with columns:
        image, level, patient_id
    """
    df = pd.read_csv(csv_path)

    # Clean up extra index columns if present
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Standardise column names
    if "image" in df.columns and "level" in df.columns:
        pass  # already correct
    elif "id_code" in df.columns and "diagnosis" in df.columns:
        df = df.rename(columns={"id_code": "image", "diagnosis": "level"})

    df["patient_id"] = df["image"].str.extract(r"(\d+)_")[0]
    bad_rows = df["patient_id"].isna()
    if bad_rows.any():
        _logger.warning(f"Dropped {bad_rows.sum()} rows with unparseable image names")
        df = df[~bad_rows].reset_index(drop=True)
    df["patient_id"] = df["patient_id"].astype(int)

    # Worst diagnosis per patient for stratification
    patient_worst = (
        df.groupby("patient_id")["level"]
        .max()
        .reset_index()
        .rename(columns={"level": "worst_level"})
    )

    test_size = 1.0 - train_ratio
    val_fraction_of_remainder = valid_ratio / (valid_ratio + (1.0 - train_ratio - valid_ratio))

    train_patients, temp_patients = train_test_split(
        patient_worst,
        test_size=test_size,
        random_state=seed,
        stratify=patient_worst["worst_level"],
    )
    valid_patients, test_patients = train_test_split(
        temp_patients,
        test_size=1.0 - val_fraction_of_remainder,
        random_state=seed,
        stratify=temp_patients["worst_level"],
    )

    train_df = df[df["patient_id"].isin(train_patients["patient_id"])].reset_index(drop=True)
    valid_df = df[df["patient_id"].isin(valid_patients["patient_id"])].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients["patient_id"])].reset_index(drop=True)

    # Integrity checks
    assert set(train_df["patient_id"]) & set(valid_df["patient_id"]) == set()
    assert set(train_df["patient_id"]) & set(test_df["patient_id"]) == set()
    assert set(valid_df["patient_id"]) & set(test_df["patient_id"]) == set()

    return train_df, valid_df, test_df


# -----------------------------------------------------------------------
# Class balancing utilities
# -----------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute balanced class weights using sklearn's 'balanced' mode.

    Weights are inversely proportional to class frequencies:
        w_c = n_samples / (n_classes * n_samples_c)

    Returns a float32 tensor suitable for nn.CrossEntropyLoss(weight=...).
    """
    classes = np.arange(config.NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """
    Per-sample weights for WeightedRandomSampler.

    Each sample gets a weight equal to the inverse frequency of its class,
    so the sampler draws roughly equal numbers from every class per epoch.
    """
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    sample_weights = class_weights[labels]
    return sample_weights


# -----------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------

class DRDataset(Dataset):
    """
    Diabetic Retinopathy fundus image dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``image`` (filename stem) and ``level`` (0-4).
    images_dir : str
        Directory containing the .jpeg images.
    transform : callable, optional
        Albumentations transform pipeline.
    preprocessing : bool
        Whether to apply the CLAHE / Ben Graham preprocessing.
    img_size : int
        Target image size (used when preprocessing is enabled).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str = config.IMAGES_DIR,
        transform=None,
        preprocessing: bool = True,
        img_size: int = config.IMG_SIZE,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_size = img_size

        self.image_ids = self.df["image"].values
        self.labels = self.df["level"].values.astype(np.int64)
        self._path_cache = self._build_image_path_cache()

    def _build_image_path_cache(self) -> dict[str, str]:
        """Resolve and cache image paths once to avoid per-item filesystem probing."""
        cache: dict[str, str] = {}
        for img_id in self.image_ids:
            if img_id in cache:
                continue
            for ext in _EXTENSIONS:
                p = os.path.join(self.images_dir, img_id + ext)
                if os.path.isfile(p):
                    cache[img_id] = p
                    break
        return cache

    def _find_image_path(self, img_id: str) -> str:
        """Try common extensions and return the first path that exists."""
        p = self._path_cache.get(img_id)
        if p is not None:
            return p
        raise FileNotFoundError(
            f"Image not found for '{img_id}' in {self.images_dir} "
            f"(tried .jpeg/.jpg/.png)"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        try:
            img_path = self._find_image_path(img_id)
            img = cv2.imread(img_path)
        except FileNotFoundError:
            img = None

        if img is None:
            _logger.warning(f"Missing/unreadable image: '{img_id}' — using black placeholder")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.preprocessing:
            img = preprocess_image(img, img_size=self.img_size)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label


# -----------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------

def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    train_transform=None,
    valid_transform=None,
    images_dir: str = config.IMAGES_DIR,
    batch_size: int = config.PHASE1_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    preprocessing: bool = True,
    use_sampler: bool = True,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders with balanced sampling for training.

    Parameters
    ----------
    use_sampler : bool
        If True, use WeightedRandomSampler on the training set so each
        mini-batch is approximately class-balanced.  This is the primary
        mechanism for handling the severe 73% / 5% / 15% / 2% / 5%
        imbalance in the DR dataset.

    Returns
    -------
    dict with keys "train", "valid", and optionally "test".
    """
    train_ds = DRDataset(
        train_df, images_dir, transform=train_transform,
        preprocessing=preprocessing,
    )
    valid_ds = DRDataset(
        valid_df, images_dir, transform=valid_transform,
        preprocessing=preprocessing,
    )

    # --- Balanced sampling ---
    sampler = None
    shuffle_train = True
    if use_sampler:
        sample_weights = compute_sample_weights(train_ds.labels)
        class_counts = np.bincount(train_ds.labels, minlength=config.NUM_CLASSES)
        min_class_count = int(class_counts[class_counts > 0].min())
        balanced_num_samples = max(
            config.NUM_CLASSES * min_class_count * 2,
            len(train_ds),
        )
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=balanced_num_samples,
            replacement=True,
        )
        shuffle_train = False  # mutually exclusive with sampler

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": config.PIN_MEMORY,
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True
        common_loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            drop_last=True,
            **common_loader_kwargs,
        ),
        "valid": DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
    }

    if test_df is not None:
        test_ds = DRDataset(
            test_df, images_dir, transform=valid_transform,
            preprocessing=preprocessing,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            **common_loader_kwargs,
        )

    return loaders


def preprocess_images_to_cache(
    df: pd.DataFrame,
    source_images_dir: str = config.IMAGES_DIR,
    cache_dir: Optional[str] = None,
    img_size: int = config.IMG_SIZE,
    overwrite: bool = False,
) -> str:
    """
    Preprocess images once and store as JPEG cache for fast training reads.
    """
    cache_dir = cache_dir or os.path.join(config.DATA_DIR, f"preprocessed_{img_size}")
    os.makedirs(cache_dir, exist_ok=True)

    unique_ids = pd.unique(df["image"])
    written = 0
    skipped = 0
    missing = 0

    for img_id in unique_ids:
        out_path = os.path.join(cache_dir, f"{img_id}.jpg")
        if os.path.isfile(out_path) and not overwrite:
            skipped += 1
            continue

        src_path = None
        for ext in _EXTENSIONS:
            candidate = os.path.join(source_images_dir, img_id + ext)
            if os.path.isfile(candidate):
                src_path = candidate
                break
        if src_path is None:
            missing += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            missing += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img, img_size=img_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        written += 1

    _logger.info(
        "Preprocessing cache ready at %s | written=%d skipped=%d missing=%d",
        cache_dir, written, skipped, missing
    )
    return cache_dir
