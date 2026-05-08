"""
PyTorch Dataset, patient-level splitting, and DataLoader factory.

Balancing strategy:
  1. Patient-level stratified split — no data leakage
  2. WeightedRandomSampler with sqrt rebalancing — balanced mini-batches

Cache strategy:
  - preprocess_images_to_cache() builds preprocessed PNGs once
  - DRDataset auto-detects and reads from cache when available
  - Eliminates redundant CPU preprocessing (~6x speedup)
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
    dataset_name: str,
    train_ratio: float = config.TRAIN_RATIO,
    valid_ratio: float = config.VALID_RATIO,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load labels CSV, extract patient IDs, and split at the patient level.

    dataset_name must be 'eyepacs' or 'aptos'.
    Both eyes of the same patient always land in the same split (EyePACS).

    Returns (train_df, valid_df, test_df) with columns:
        image, level, patient_id, source
    """
    if dataset_name == "eyepacs":
        csv_path = config.EYEPACS_LABELS_CSV
    elif dataset_name == "aptos":
        csv_path = config.APTOS_LABELS_CSV
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Labels CSV missing: {csv_path}. Please download the dataset.")

    df = pd.read_csv(csv_path)

    # Clean up extra index columns
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Standardize column names (APTOS uses id_code/diagnosis)
    if "id_code" in df.columns and "diagnosis" in df.columns:
        df = df.rename(columns={"id_code": "image", "diagnosis": "level"})

    df["source"] = dataset_name

    # Extract patient clustering logic
    if dataset_name == "eyepacs":
        # EyePACS has syntax like 10_left, 10_right — group by patient number
        df["patient_id"] = df["image"].str.extract(r"(\d+)_")[0]
        df["patient_id"] = df["patient_id"].fillna(df["image"])
    else:
        # APTOS images are independent (no paired eyes)
        df["patient_id"] = df["image"]

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
    """Compute sklearn balanced class weights."""
    classes = np.arange(config.NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(labels: np.ndarray, power: float | None = None) -> np.ndarray:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Uses sqrt rebalancing by default: weight = 1/count^power.
    Does NOT normalize — WeightedRandomSampler handles this internally.
    """
    if power is None:
        power = getattr(config, "SAMPLER_POWER", 0.5)
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = (1.0 / np.maximum(class_counts, 1).astype(np.float64)) ** power
    # NOTE: Do NOT normalize here. WeightedRandomSampler expects raw weights.
    sample_weights = class_weights[labels]
    return sample_weights


# -----------------------------------------------------------------------
# Cache directory resolution
# -----------------------------------------------------------------------


def _get_cache_dir(dataset_name: str, img_size: int | None = None) -> str:
    """Get the path to the preprocessed image cache directory."""
    if img_size is None:
        img_size = config.resolve_img_size(dataset_name)
    base = config.DATA_DIR if dataset_name == "eyepacs" else config.APTOS_DATA_DIR
    return os.path.join(base, f"preprocessed_{img_size}_{dataset_name}")


def _get_raw_images_dir(dataset_name: str) -> str:
    """Get the raw images directory for a dataset."""
    if dataset_name == "eyepacs":
        return config.EYEPACS_IMAGES_DIR
    elif dataset_name == "aptos":
        return config.APTOS_IMAGES_DIR
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


# -----------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------


class DRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        transform=None,
        preprocessing: bool = True,
        img_size: int | None = None,
        fail_on_missing: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.dataset_name = dataset_name
        self.transform = transform
        self.img_size = img_size if img_size is not None else config.resolve_img_size(dataset_name)
        self.fail_on_missing = fail_on_missing

        # --- CACHE FIX: Auto-detect and use preprocessed cache ---
        cache_dir = _get_cache_dir(dataset_name, img_size)
        if config.USE_PREPROCESSED_CACHE and os.path.isdir(cache_dir):
            self.images_dir = cache_dir
            self.preprocessing = False  # Images already preprocessed!
            self._use_cache = True
            _logger.info(f"Using preprocessed cache: {cache_dir}")
        else:
            self.images_dir = _get_raw_images_dir(dataset_name)
            self.preprocessing = preprocessing
            self._use_cache = False
            _logger.info(f"Using raw images (no cache): {self.images_dir}")

        self.image_ids = self.df["image"].values
        self.labels = self.df["level"].values.astype(np.int64)
        self._path_cache = self._build_image_path_cache()

    def _build_image_path_cache(self) -> dict[str, str]:
        """Resolve and cache image paths once to avoid per-item filesystem probing."""
        cache: dict[str, str] = {}
        # Use .values for speed instead of iterrows() (~100x faster on 35K rows)
        unique_ids = pd.unique(self.df["image"])
        for img_id in unique_ids:
            for ext in _EXTENSIONS:
                p = os.path.join(self.images_dir, img_id + ext)
                if os.path.isfile(p):
                    cache[img_id] = p
                    break
        return cache

    def _find_image_path(self, img_id: str) -> str:
        p = self._path_cache.get(img_id)
        if p is not None:
            return p
        raise FileNotFoundError(
            f"Image '{img_id}' not found. Please ensure it exists in {self.images_dir}."
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
            msg = f"Missing/unreadable image: '{img_id}' in {self.images_dir}."
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
    dataset_name: str,
    test_df: pd.DataFrame | None = None,
    train_transform=None,
    valid_transform=None,
    batch_size: int = config.STAGE1_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    preprocessing: bool = True,
    use_sampler: bool = config.USE_WEIGHTED_SAMPLER,
    sampler_seed: int | None = config.SEED,
    img_size: int | None = None,
) -> dict[str, DataLoader]:
    resolved_img = img_size if img_size is not None else config.resolve_img_size(dataset_name)
    train_ds = DRDataset(
        train_df,
        dataset_name=dataset_name,
        transform=train_transform,
        preprocessing=preprocessing,
        img_size=resolved_img,
    )
    valid_ds = DRDataset(
        valid_df,
        dataset_name=dataset_name,
        transform=valid_transform,
        preprocessing=preprocessing,
        img_size=resolved_img,
    )

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
        shuffle_train = False

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
            dataset_name=dataset_name,
            transform=valid_transform,
            preprocessing=preprocessing,
            img_size=resolved_img,
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
    dataset_name: str,
    cache_dir: str | None = None,
    img_size: int | None = None,
    overwrite: bool = False,
    cache_image_format: str = config.CACHE_IMAGE_FORMAT,
    cache_image_quality: int = config.CACHE_IMAGE_QUALITY,
) -> str:
    """Preprocess images and save to cache directory for fast loading."""
    if img_size is None:
        img_size = config.resolve_img_size(dataset_name)
    cache_image_format = cache_image_format.lower().lstrip(".")
    if cache_image_format not in {"jpg", "jpeg", "png", "webp"}:
        raise ValueError("Unsupported cache_image_format. Use one of: jpg, jpeg, png, webp")

    source_images_dir = _get_raw_images_dir(dataset_name)
    cache_dir = cache_dir or _get_cache_dir(dataset_name, img_size)
    os.makedirs(cache_dir, exist_ok=True)

    unique_imgs = df.drop_duplicates(subset=["image"])
    cache_ext = f".{cache_image_format}"
    meta_path = os.path.join(cache_dir, "cache_meta.json")

    source_images_dir_abs = os.path.abspath(source_images_dir)
    current_meta = {
        "version": config.PREPROCESSING_CACHE_VERSION,
        "dataset_name": dataset_name,
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
                _logger.info("Cache metadata mismatch detected in %s; forcing refresh.", cache_dir)
                overwrite = True
        except (OSError, json.JSONDecodeError):
            _logger.warning("Could not read cache metadata at %s; forcing refresh.", meta_path)
            overwrite = True

    written = 0
    skipped = 0
    missing = 0

    for _, row in unique_imgs.iterrows():
        img_id = row["image"]
        out_path = os.path.join(cache_dir, f"{img_id}{cache_ext}")
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
            _logger.warning("Source image missing for '%s'", img_id)
            continue

        img = cv2.imread(src_path)
        if img is None:
            missing += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img, img_size=img_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if cache_image_format in {"jpg", "jpeg"}:
            write_ok = cv2.imwrite(
                out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(cache_image_quality)]
            )
        elif cache_image_format == "png":
            write_ok = cv2.imwrite(
                out_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), int(config.CACHE_PNG_COMPRESSION)]
            )
        else:
            write_ok = cv2.imwrite(
                out_path, img, [int(cv2.IMWRITE_WEBP_QUALITY), int(cache_image_quality)]
            )

        if not write_ok:
            missing += 1
            continue
        written += 1

    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(current_meta, f, indent=2, sort_keys=True)
    except OSError:
        pass

    _logger.info(
        f"Cache build complete: {written} written, {skipped} skipped, {missing} missing. "
        f"Cache dir: {cache_dir}"
    )

    if config.CACHE_FAIL_ON_MISSING and missing > 0:
        raise RuntimeError(f"Cache build failed: {missing} missing.")

    return cache_dir


def validate_cache_for_dataframe(
    df: pd.DataFrame,
    cache_dir: str,
    cache_image_format: str = config.CACHE_IMAGE_FORMAT,
) -> dict[str, int]:
    """Validate that all images in a DataFrame exist in the cache."""
    cache_image_format = cache_image_format.lower().lstrip(".")
    expected_ids = pd.unique(df["image"])
    cache_ext = f".{cache_image_format}"

    missing_ids = []
    for img_id in expected_ids:
        if not os.path.isfile(os.path.join(cache_dir, f"{img_id}{cache_ext}")):
            missing_ids.append(img_id)

    summary = {
        "expected": int(len(expected_ids)),
        "missing": int(len(missing_ids)),
        "present": int(len(expected_ids) - len(missing_ids)),
    }
    if missing_ids:
        _logger.warning(
            f"Cache validation: {summary['missing']} missing out of {summary['expected']}. "
            f"Examples: {', '.join(missing_ids[:5])}"
        )

    return summary
