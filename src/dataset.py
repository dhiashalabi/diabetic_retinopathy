"""
PyTorch Dataset, patient-level splitting, and DataLoader factory.

Balancing strategy:
  1. Patient-level stratified split — no data leakage
  2. WeightedRandomSampler — balanced mini-batches (controlled by config.USE_WEIGHTED_SAMPLER)
  3. Class-weighted loss — available via config.USE_CLASS_WEIGHTED_LOSS (off by default
     to avoid double-rebalancing when sampler is active)
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config
from .preprocessing import preprocess_image
from .utils import get_logger

_logger = get_logger(__name__)
_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp", ".JPEG", ".JPG", ".PNG", ".WEBP")


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
    if "id_code" in df.columns and "diagnosis" in df.columns:
        df = df.rename(columns={"id_code": "image", "diagnosis": "level"})
    
    df["source"] = "eyepacs"

    # Add APTOS dataset if enabled
    if getattr(config, "USE_APTOS_DATASET", False) and os.path.isfile(config.APTOS_LABELS_CSV):
        df_aptos = pd.read_csv(config.APTOS_LABELS_CSV)
        df_aptos = df_aptos.rename(columns={"id_code": "image", "diagnosis": "level"})
        df_aptos["source"] = "aptos"
        df = pd.concat([df, df_aptos], ignore_index=True)

    df["patient_id"] = df["image"].str.extract(r"(\d+)_")[0]
    # For APTOS, patient_id won't have \d+_, so we just use the image ID
    df["patient_id"] = df["patient_id"].fillna(df["image"])
    df["patient_id"] = df["patient_id"].astype(str)

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


def compute_sample_weights(labels: np.ndarray, power: float = 0.5) -> np.ndarray:
    """
    Per-sample weights for WeightedRandomSampler.

    Using power=0.5 applies square-root rebalancing by default (moderates
    the effect on extremely rare classes to prevent over-duplication).
    power=1.0 is full inverse frequency.
    """
    if hasattr(config, "SAMPLER_POWER"):
        power = config.SAMPLER_POWER
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = (1.0 / np.maximum(class_counts, 1).astype(np.float64)) ** power
    class_weights /= class_weights.sum()  # normalize
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
        fail_on_missing: bool = config.CACHE_FAIL_ON_MISSING,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_size = img_size
        self.fail_on_missing = fail_on_missing

        self.image_ids = self.df["image"].values
        self.labels = self.df["level"].values.astype(np.int64)
        self._path_cache = self._build_image_path_cache()

    def _build_image_path_cache(self) -> dict[str, str]:
        """Resolve and cache image paths once to avoid per-item filesystem probing."""
        cache: dict[str, str] = {}
        for idx, row in self.df.iterrows():
            img_id = row["image"]
            source = row.get("source", "eyepacs")
            if img_id in cache:
                continue
                
            search_dir = config.APTOS_IMAGES_DIR if source == "aptos" else self.images_dir
            for ext in _EXTENSIONS:
                p = os.path.join(search_dir, img_id + ext)
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
            f"Image '{img_id}' not found. Please ensure it exists in the expected directory."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        try:
            img_path = self._find_image_path(img_id)
            img = cv2.imread(img_path)
        except FileNotFoundError as exc:
            if self.fail_on_missing:
                raise FileNotFoundError(
                    f"Missing image '{img_id}' in {self.images_dir}. "
                    "Run cache preprocessing/audit before training."
                ) from exc
            img = None

        if img is None:
            msg = (
                f"Missing/unreadable image: '{img_id}' in {self.images_dir}. "
                "Run cache preprocessing/audit before training."
            )
            if self.fail_on_missing:
                raise RuntimeError(msg)
            _logger.warning("%s Using black placeholder due to fail_on_missing=False.", msg)
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
    test_df: pd.DataFrame | None = None,
    train_transform=None,
    valid_transform=None,
    images_dir: str = config.IMAGES_DIR,
    batch_size: int = config.PHASE1_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    preprocessing: bool = True,
    use_sampler: bool = config.USE_WEIGHTED_SAMPLER,
    sampler_seed: int | None = config.SEED,
    fail_on_missing: bool = config.CACHE_FAIL_ON_MISSING,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders with optional balanced sampling for training.

    Parameters
    ----------
    use_sampler : bool
        If True, use WeightedRandomSampler on the training set so each
        mini-batch is approximately class-balanced.  Defaults to
        ``config.USE_WEIGHTED_SAMPLER``.

    Returns
    -------
    dict with keys "train", "valid", and optionally "test".
    """
    train_ds = DRDataset(
        train_df,
        images_dir,
        transform=train_transform,
        preprocessing=preprocessing,
        fail_on_missing=fail_on_missing,
    )
    valid_ds = DRDataset(
        valid_df,
        images_dir,
        transform=valid_transform,
        preprocessing=preprocessing,
        fail_on_missing=fail_on_missing,
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
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.SEED if sampler_seed is None else sampler_seed)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=balanced_num_samples,
            replacement=True,
            generator=sampler_generator,
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
            test_df,
            images_dir,
            transform=valid_transform,
            preprocessing=preprocessing,
            fail_on_missing=fail_on_missing,
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
    cache_dir: str | None = None,
    img_size: int = config.IMG_SIZE,
    overwrite: bool = False,
    cache_image_format: str = config.CACHE_IMAGE_FORMAT,
    cache_image_quality: int = config.CACHE_IMAGE_QUALITY,
    fail_on_missing: bool = config.CACHE_FAIL_ON_MISSING,
) -> str:
    """
    Preprocess images once and store to cache for fast training reads.
    """
    cache_image_format = cache_image_format.lower().lstrip(".")
    if cache_image_format not in {"jpg", "jpeg", "png", "webp"}:
        raise ValueError("Unsupported cache_image_format. Use one of: jpg, jpeg, png, webp")

    cache_dir = cache_dir or os.path.join(config.DATA_DIR, f"preprocessed_{img_size}")
    os.makedirs(cache_dir, exist_ok=True)

    unique_imgs = df.drop_duplicates(subset=["image"])
    cache_ext = f".{cache_image_format}"
    meta_path = os.path.join(cache_dir, "cache_meta.json")

    source_images_dir_abs = os.path.abspath(source_images_dir)
    current_meta = {
        "version": config.PREPROCESSING_CACHE_VERSION,
        "img_size": int(img_size),
        "source_images_dir": source_images_dir_abs,
        "cache_image_format": cache_image_format,
        "cache_image_quality": int(cache_image_quality),
        "num_unique_images": int(len(unique_imgs)),
    }

    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                existing_meta = json.load(f)
            if existing_meta != current_meta and not overwrite:
                _logger.info(
                    "Cache metadata mismatch detected in %s; forcing refresh.",
                    cache_dir,
                )
                overwrite = True
        except (OSError, json.JSONDecodeError):
            _logger.warning(
                "Could not read cache metadata at %s; forcing refresh.",
                meta_path,
            )
            overwrite = True

    written = 0
    skipped = 0
    missing = 0

    for _, row in unique_imgs.iterrows():
        img_id = row["image"]
        source = row.get("source", "eyepacs")
        out_path = os.path.join(cache_dir, f"{img_id}{cache_ext}")
        if os.path.isfile(out_path) and not overwrite:
            skipped += 1
            continue

        src_path = None
        search_dir = config.APTOS_IMAGES_DIR if source == "aptos" else source_images_dir
        for ext in _EXTENSIONS:
            candidate = os.path.join(search_dir, img_id + ext)
            if os.path.isfile(candidate):
                src_path = candidate
                break
        if src_path is None:
            missing += 1
            _logger.warning("Source image missing for '%s' while building cache", img_id)
            continue

        img = cv2.imread(src_path)
        if img is None:
            missing += 1
            _logger.warning("Unreadable source image for '%s' while building cache", img_id)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img, img_size=img_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if cache_image_format in {"jpg", "jpeg"}:
            write_ok = cv2.imwrite(
                out_path,
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(cache_image_quality)],
            )
        elif cache_image_format == "png":
            write_ok = cv2.imwrite(
                out_path,
                img,
                [int(cv2.IMWRITE_PNG_COMPRESSION), int(config.CACHE_PNG_COMPRESSION)],
            )
        else:  # webp
            write_ok = cv2.imwrite(
                out_path,
                img,
                [int(cv2.IMWRITE_WEBP_QUALITY), int(cache_image_quality)],
            )

        if not write_ok:
            _logger.warning("Failed writing cached image: %s", out_path)
            missing += 1
            continue
        written += 1

    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(current_meta, f, indent=2, sort_keys=True)
    except OSError:
        _logger.warning("Failed to write cache metadata at %s", meta_path)

    _logger.info(
        "Preprocessing cache ready at %s | written=%d skipped=%d missing=%d",
        cache_dir,
        written,
        skipped,
        missing,
    )

    if fail_on_missing and missing > 0:
        raise RuntimeError(
            f"Cache build failed integrity check: missing/unreadable count={missing}. "
            "Fix data issues and rerun preprocessing."
        )

    return cache_dir


def validate_cache_for_dataframe(
    df: pd.DataFrame,
    cache_dir: str,
    cache_image_format: str = config.CACHE_IMAGE_FORMAT,
) -> dict[str, int]:
    """Validate that every expected image ID exists in cache with the target extension."""
    cache_image_format = cache_image_format.lower().lstrip(".")
    expected_ids = pd.unique(df["image"])
    cache_ext = f".{cache_image_format}"

    missing_ids: list[str] = []
    for img_id in expected_ids:
        out_path = os.path.join(cache_dir, f"{img_id}{cache_ext}")
        if not os.path.isfile(out_path):
            missing_ids.append(img_id)

    summary = {
        "expected": int(len(expected_ids)),
        "missing": int(len(missing_ids)),
        "present": int(len(expected_ids) - len(missing_ids)),
    }
    if missing_ids:
        sample = ", ".join(missing_ids[:5])
        raise RuntimeError(
            f"Cache validation failed in {cache_dir}: {summary['missing']} missing files. "
            f"Examples: {sample}"
        )

    _logger.info(
        "Cache validation passed for %s | expected=%d present=%d missing=%d",
        cache_dir,
        summary["expected"],
        summary["present"],
        summary["missing"],
    )
    return summary
