"""
Unified dataset download and setup for all environments.

Handles three scenarios:
- Local / GPU server: dataset already exists or download via Kaggle API
- Google Colab: download via Kaggle to local SSD for fast I/O
- Kaggle credentials: env vars (KAGGLE_USERNAME + KAGGLE_KEY),
  ~/.kaggle/kaggle.json, or interactive Colab upload
"""

import logging
import os
import shutil
import zipfile

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "tanlikesmath/diabetic-retinopathy-resized"
KAGGLE_COMPETITION_APTOS = "aptos2019-blindness-detection"


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    return os.path.isfile(kaggle_json)


def _setup_colab_kaggle_credentials() -> None:
    """On Colab, prompt user to upload kaggle.json if no credentials found."""
    if _has_kaggle_credentials():
        return
    from google.colab import files

    print("No Kaggle credentials found. Upload your kaggle.json:")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("kaggle.json was not uploaded.")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dest = os.path.join(kaggle_dir, "kaggle.json")
    shutil.move("kaggle.json", dest)
    os.chmod(dest, 0o600)
    logger.info("Kaggle credentials installed.")


def _download_from_kaggle(data_dir: str) -> None:
    """Download and extract the dataset using the Kaggle API."""
    if not _has_kaggle_credentials() and _is_colab():
        _setup_colab_kaggle_credentials()

    if not _has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables, or place kaggle.json in ~/.kaggle/"
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    zip_dir = os.path.dirname(data_dir) or "."
    os.makedirs(zip_dir, exist_ok=True)

    logger.info(f"Downloading {KAGGLE_DATASET} to {zip_dir} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=zip_dir, quiet=False)

    zip_name = KAGGLE_DATASET.split("/")[-1] + ".zip"
    zip_path = os.path.join(zip_dir, zip_name)

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Expected zip file not found: {zip_path}")

    logger.info(f"Extracting {zip_path} to {data_dir} ...")
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    logger.info("Download and extraction complete.")


def _resolve_images_dir(data_dir: str) -> str:
    """Handle the nested folder structure from Kaggle zip."""
    candidate = os.path.join(data_dir, "resized_train_cropped")
    nested = os.path.join(candidate, "resized_train_cropped")
    if os.path.isdir(nested):
        return nested
    return candidate


def _dataset_exists(data_dir: str) -> bool:
    """Check if the dataset images are present."""
    images_dir = _resolve_images_dir(data_dir)
    if not os.path.isdir(images_dir):
        return False
    try:
        entries = os.listdir(images_dir)
    except OSError:
        return False
    return len(entries) > 100


def _download_aptos_from_kaggle(data_dir: str) -> None:
    """Download APTOS 2019 competition dataset."""
    if not _has_kaggle_credentials() and _is_colab():
        _setup_colab_kaggle_credentials()

    if not _has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables, or place kaggle.json in ~/.kaggle/"
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Downloading {KAGGLE_COMPETITION_APTOS} to {data_dir} ...")
    api.competition_download_files(KAGGLE_COMPETITION_APTOS, path=data_dir, quiet=False)

    zip_path = os.path.join(data_dir, f"{KAGGLE_COMPETITION_APTOS}.zip")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Expected zip file not found: {zip_path}")

    logger.info(f"Extracting {zip_path} to {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    logger.info("APTOS download and extraction complete.")


def ensure_aptos_dataset(data_dir: str | None = None) -> str:
    """
    Ensure the APTOS 2019 dataset is available.
    Downloads from Kaggle if not present.
    """
    if data_dir is None:
        from . import config
        data_dir = config.APTOS_DATA_DIR

    train_csv = os.path.join(data_dir, "train.csv")
    if os.path.isfile(train_csv):
        logger.info(f"APTOS dataset found at {data_dir}")
        return data_dir

    logger.info(f"APTOS dataset not found at {data_dir}. Downloading from Kaggle ...")
    _download_aptos_from_kaggle(data_dir)

    if not os.path.isfile(train_csv):
        raise RuntimeError(
            f"Download completed but APTOS dataset not found at {data_dir}. "
            "Check the Kaggle dataset structure."
        )

    logger.info(f"APTOS dataset ready at {data_dir}")
    return data_dir


def ensure_dataset(data_dir: str | None = None) -> str:
    """
    Ensure the dataset is available at data_dir.
    Downloads from Kaggle if not present.

    Parameters
    ----------
    data_dir : str, optional
        Target directory. Defaults based on environment:
        - Colab: /content/dataset
        - Otherwise: <project_root>/dataset

    Returns
    -------
    str
        The resolved data directory path.
    """
    if data_dir is None:
        from . import config

        data_dir = config.DATA_DIR

    if _dataset_exists(data_dir):
        images_dir = _resolve_images_dir(data_dir)
        logger.info(f"Dataset found at {images_dir}")
        return data_dir

    logger.info(f"Dataset not found at {data_dir}. Downloading from Kaggle ...")
    _download_from_kaggle(data_dir)

    if not _dataset_exists(data_dir):
        raise RuntimeError(
            f"Download completed but dataset not found at {data_dir}. "
            "Check the Kaggle dataset structure."
        )

    images_dir = _resolve_images_dir(data_dir)
    logger.info(f"Dataset ready at {images_dir}")
    return data_dir
