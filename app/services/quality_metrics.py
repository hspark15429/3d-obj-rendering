"""Quality metrics computation service for reconstruction quality assessment.

This module provides functions for computing image quality metrics (PSNR, SSIM)
and depth error metrics (MAE, RMSE) to assess 3D reconstruction quality.

Quality Thresholds (from Phase 4 CONTEXT.md):
--------------------------------------------
PSNR (Peak Signal-to-Noise Ratio):
  - Normal:  >= 25 dB (good quality reconstruction)
  - Warning: >= 20 dB (acceptable quality, may have visible artifacts)
  - Failure: <  20 dB (poor quality, significant reconstruction errors)

SSIM (Structural Similarity Index):
  - Normal:  >= 0.85 (high perceptual similarity)
  - Warning: >= 0.75 (acceptable similarity)
  - Failure: <  0.75 (low similarity, structural issues)

Status Classification:
  - "normal": PSNR >= 25 dB AND SSIM >= 0.85
  - "warning": PSNR >= 20 dB AND SSIM >= 0.75 (but not normal)
  - "failure": Below warning thresholds

Usage:
    from app.services.quality_metrics import (
        compute_psnr, compute_ssim, compute_depth_error, classify_quality_status
    )

    # Load images as float32 arrays in [0, 1] range
    img_true = load_image_float(Path("input/view_00.png"))
    img_test = load_image_float(Path("output/render_00.png"))

    # Compute metrics
    psnr = compute_psnr(img_true, img_test)
    ssim = compute_ssim(img_true, img_test)

    # Classify quality
    status = classify_quality_status(psnr, ssim)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)

# Quality thresholds (hardcoded per user decision - no runtime config)
PSNR_NORMAL_THRESHOLD = 25.0  # dB
PSNR_WARNING_THRESHOLD = 20.0  # dB
SSIM_NORMAL_THRESHOLD = 0.85
SSIM_WARNING_THRESHOLD = 0.75


def load_image_float(path: Path) -> np.ndarray:
    """Load image as float32 array in [0, 1] range.

    Args:
        path: Path to image file (PNG, JPEG, etc.)

    Returns:
        np.ndarray: Image array with shape (H, W, 3) and dtype float32,
                    values normalized to [0, 1] range.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be loaded or converted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        logger.debug(f"Loaded image {path.name}: shape={img_array.shape}")
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def compute_psnr(img_true: np.ndarray, img_test: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    PSNR measures the ratio between the maximum possible signal value
    and the noise (distortion) introduced by reconstruction.

    CRITICAL: Always specify data_range=1.0 for float images in [0, 1].
    Scikit-image defaults assume [-1, 1] range for floats which gives
    incorrect results.

    Args:
        img_true: Reference image (H, W, C) in [0, 1] range.
        img_test: Test/reconstructed image (H, W, C) in [0, 1] range.

    Returns:
        float: PSNR value in decibels (dB). Higher is better.
               Returns infinity for identical images.
               Typical range: 20-40 dB for reconstruction tasks.

    Raises:
        ValueError: If images have different shapes.
    """
    if img_true.shape != img_test.shape:
        raise ValueError(
            f"Image shape mismatch: {img_true.shape} vs {img_test.shape}"
        )

    psnr = peak_signal_noise_ratio(img_true, img_test, data_range=1.0)
    logger.debug(f"Computed PSNR: {psnr:.2f} dB")
    return float(psnr)


def compute_ssim(img_true: np.ndarray, img_test: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    SSIM measures perceptual similarity based on luminance, contrast,
    and structural information. More aligned with human perception
    than pixel-wise metrics like MSE.

    CRITICAL: Always specify data_range=1.0 and channel_axis=-1 for RGB.
    Scikit-image defaults assume [-1, 1] range for floats.

    Args:
        img_true: Reference image (H, W, C) in [0, 1] range.
        img_test: Test/reconstructed image (H, W, C) in [0, 1] range.

    Returns:
        float: SSIM score in [0, 1] range. 1.0 means identical.
               Typical range: 0.7-0.95 for reconstruction tasks.

    Raises:
        ValueError: If images have different shapes.
    """
    if img_true.shape != img_test.shape:
        raise ValueError(
            f"Image shape mismatch: {img_true.shape} vs {img_test.shape}"
        )

    ssim = structural_similarity(
        img_true,
        img_test,
        data_range=1.0,
        channel_axis=-1,  # Last axis is RGB channels
    )
    logger.debug(f"Computed SSIM: {ssim:.4f}")
    return float(ssim)


def compute_depth_error(
    depth_true: np.ndarray, depth_pred: np.ndarray
) -> Dict[str, float]:
    """Compute depth error metrics between depth maps.

    Computes Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
    between two depth maps, considering only valid pixels where both
    maps have non-zero depth values.

    Args:
        depth_true: Ground truth depth map (H, W), depth > 0 for valid pixels.
        depth_pred: Predicted depth map (H, W), depth > 0 for valid pixels.

    Returns:
        dict with keys:
            - "mae": Mean Absolute Error (more interpretable, average error)
            - "rmse": Root Mean Square Error (penalizes large errors)
            - "valid_pixels": Number of pixels compared

    Raises:
        ValueError: If depth maps have different shapes or no valid pixels.
    """
    if depth_true.shape != depth_pred.shape:
        raise ValueError(
            f"Depth map shape mismatch: {depth_true.shape} vs {depth_pred.shape}"
        )

    # Create valid mask for pixels with non-zero depth in both maps
    valid_mask = (depth_true > 0) & (depth_pred > 0)
    valid_pixels = int(valid_mask.sum())

    if valid_pixels == 0:
        raise ValueError("No valid depth pixels for comparison (all zeros)")

    gt_valid = depth_true[valid_mask]
    pred_valid = depth_pred[valid_mask]

    # Mean Absolute Error
    diff = np.abs(gt_valid - pred_valid)
    mae = float(np.mean(diff))

    # Root Mean Square Error
    rmse = float(np.sqrt(np.mean((gt_valid - pred_valid) ** 2)))

    logger.debug(
        f"Computed depth metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, "
        f"valid_pixels={valid_pixels}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "valid_pixels": valid_pixels,
    }


def compute_overall_metrics(
    image_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, float]:
    """Compute averaged PSNR and SSIM across multiple image pairs.

    Per user decision, only overall averaged metrics are needed
    (no per-view breakdown).

    Args:
        image_pairs: List of (reference, test) image array tuples.
                     Each image should be (H, W, C) in [0, 1] range.

    Returns:
        dict with keys:
            - "psnr": Mean PSNR across all pairs (dB)
            - "ssim": Mean SSIM across all pairs

    Raises:
        ValueError: If image_pairs is empty.
    """
    if not image_pairs:
        raise ValueError("No image pairs provided for metric computation")

    psnr_scores = []
    ssim_scores = []

    for i, (img_true, img_test) in enumerate(image_pairs):
        psnr = compute_psnr(img_true, img_test)
        ssim = compute_ssim(img_true, img_test)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        logger.debug(f"Pair {i}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))

    logger.info(
        f"Overall metrics ({len(image_pairs)} pairs): "
        f"PSNR={mean_psnr:.2f} dB, SSIM={mean_ssim:.4f}"
    )

    return {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
    }


def classify_quality_status(psnr: float, ssim: float) -> str:
    """Classify reconstruction quality status based on thresholds.

    Applies hardcoded thresholds from Phase 4 CONTEXT.md:
    - Normal:  PSNR >= 25 dB AND SSIM >= 0.85
    - Warning: PSNR >= 20 dB AND SSIM >= 0.75 (but not normal)
    - Failure: Below warning thresholds

    Args:
        psnr: PSNR score in decibels.
        ssim: SSIM score in [0, 1] range.

    Returns:
        str: One of "normal", "warning", or "failure".
    """
    # Normal: both metrics meet high thresholds
    if psnr >= PSNR_NORMAL_THRESHOLD and ssim >= SSIM_NORMAL_THRESHOLD:
        status = "normal"
    # Warning: both metrics meet minimum thresholds
    elif psnr >= PSNR_WARNING_THRESHOLD and ssim >= SSIM_WARNING_THRESHOLD:
        status = "warning"
    # Failure: at least one metric below warning threshold
    else:
        status = "failure"

    logger.info(
        f"Quality status: {status} "
        f"(PSNR={psnr:.2f} dB, SSIM={ssim:.4f})"
    )

    return status
