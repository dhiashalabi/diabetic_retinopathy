"""
Image preprocessing pipeline for retinal fundus images.

Implements:
- Automatic fundus cropping (remove black borders)
- CLAHE enhancement in LAB colour space
- Ben Graham preprocessing (local colour subtraction)
- Circular mask to remove edge artefacts
"""

import cv2
import numpy as np

from . import config


def crop_fundus(img: np.ndarray, tolerance: int = 15) -> np.ndarray:
    """
    Auto-crop the retinal disc from a dark background.

    Converts to grayscale, thresholds to find the bright fundus region,
    and returns the tightly cropped bounding box.
    """
    if img is None or img.size == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Minimal area check — avoid tiny crops from noise
    if w * h < 100:
        return img

    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    return img[y : y + h, x : x + w]


def apply_clahe(
    img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Enhance contrast via CLAHE on the L channel of LAB colour space.

    Preserves colour information while boosting local contrast — critical
    for revealing microaneurysms, haemorrhages, and exudates.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_ch)

    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def ben_graham_preprocessing(img: np.ndarray, sigma_ratio: float = 0.1) -> np.ndarray:
    """
    Ben Graham's colour-normalisation trick (1st place, Kaggle DR 2015).

    Subtracts the local average colour to highlight fine vascular detail
    while suppressing illumination variation across the fundus.

    Formula: result = addWeighted(img, 4, GaussianBlur(img), -4, 128)
    """
    h, w = img.shape[:2]
    sigma = int(min(h, w) * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1

    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, 4, blurred, -4, 128)

    return result


def apply_circular_mask(img: np.ndarray) -> np.ndarray:
    """
    Apply a circular mask to remove edge artefacts from cropping.

    The mask is centred on the image; radius is ~49% of the smaller
    dimension to cover the disc while clipping less peripheral retina.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    centre = (w // 2, h // 2)
    radius = int(min(h, w) * 0.49)
    cv2.circle(mask, centre, radius, 255, -1)

    result = img.copy()
    result[mask == 0] = 0
    return result


def preprocess_image(
    img: np.ndarray,
    img_size: int = config.STAGE2_IMG_SIZE,
    use_ben_graham: bool = config.USE_BEN_GRAHAM,
    use_clahe: bool = True,
    use_crop: bool = True,
    use_circular_mask: bool = config.USE_CIRCULAR_MASK,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single retinal image.

    Parameters
    ----------
    img : np.ndarray
        RGB image (H, W, 3), uint8.
    img_size : int
        Target square size.
    use_ben_graham : bool
        Apply Ben Graham colour normalisation.
    use_clahe : bool
        Apply CLAHE enhancement.
    use_crop : bool
        Auto-crop the fundus disc.
    use_circular_mask : bool
        Mask corners after cropping.

    Returns
    -------
    np.ndarray
        Preprocessed RGB image (img_size, img_size, 3), uint8.
    """
    if img is None or img.size == 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if use_crop:
        img = crop_fundus(img)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if use_circular_mask:
        img = apply_circular_mask(img)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if use_ben_graham:
        img = ben_graham_preprocessing(img)
        img = np.clip(img, 0, 255).astype(np.uint8)

    if use_clahe:
        img = apply_clahe(img)
    return img


def load_and_preprocess(image_path: str, img_size: int = config.STAGE2_IMG_SIZE, **kwargs) -> np.ndarray:
    """Convenience: read an image file from disk and preprocess it."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img, img_size=img_size, **kwargs)
