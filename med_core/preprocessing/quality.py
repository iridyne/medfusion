"""
Image quality assessment utilities for medical images.

Provides functions and classes for:
- Automated image quality metrics (sharpness, contrast, noise)
- Artifact detection (motion blur, compression artifacts)
- Quality-based filtering for dataset curation
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for image quality metrics."""

    # Sharpness metrics
    laplacian_variance: float = 0.0  # Higher = sharper
    gradient_magnitude: float = 0.0  # Higher = more edges

    # Contrast metrics
    contrast_rms: float = 0.0  # RMS contrast
    dynamic_range: float = 0.0  # Range of intensity values

    # Noise metrics
    noise_estimate: float = 0.0  # Estimated noise level
    snr_estimate: float = 0.0  # Signal-to-noise ratio estimate

    # Artifact indicators
    has_motion_blur: bool = False
    has_compression_artifacts: bool = False
    has_watermark: bool = False

    # Overall quality score (0-1)
    overall_score: float = 0.0

    # Additional info
    warnings: list[str] = field(default_factory=list)

    def is_acceptable(self, min_score: float = 0.5) -> bool:
        """Check if image quality is acceptable."""
        return self.overall_score >= min_score

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "laplacian_variance": self.laplacian_variance,
            "gradient_magnitude": self.gradient_magnitude,
            "contrast_rms": self.contrast_rms,
            "dynamic_range": self.dynamic_range,
            "noise_estimate": self.noise_estimate,
            "snr_estimate": self.snr_estimate,
            "has_motion_blur": self.has_motion_blur,
            "has_compression_artifacts": self.has_compression_artifacts,
            "has_watermark": self.has_watermark,
            "overall_score": self.overall_score,
            "warnings": self.warnings,
        }


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute Laplacian variance as a sharpness metric.

    Higher values indicate sharper images.

    Args:
        image: Grayscale image as numpy array

    Returns:
        Laplacian variance value
    """
    # Simple Laplacian kernel
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Manual convolution (avoiding scipy dependency)
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    image = image.astype(np.float32)

    # Pad image
    padded = np.pad(image, 1, mode='reflect')

    # Apply Laplacian
    laplacian = np.zeros_like(image)
    for i in range(3):
        for j in range(3):
            laplacian += laplacian_kernel[i, j] * padded[i:i+h, j:j+w]

    return float(laplacian.var())


def compute_gradient_magnitude(image: np.ndarray) -> float:
    """
    Compute average gradient magnitude.

    Args:
        image: Input image as numpy array

    Returns:
        Average gradient magnitude
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    image = image.astype(np.float32)

    # Compute gradients using simple differences
    grad_x = np.diff(image, axis=1)
    grad_y = np.diff(image, axis=0)

    # Compute magnitude (approximate)
    mag_x = np.abs(grad_x).mean()
    mag_y = np.abs(grad_y).mean()

    return float(np.sqrt(mag_x**2 + mag_y**2))


def compute_contrast_rms(image: np.ndarray) -> float:
    """
    Compute RMS contrast.

    Args:
        image: Input image as numpy array (0-255 or 0-1)

    Returns:
        RMS contrast value (0-1)
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Normalize to 0-1
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0

    # RMS contrast is the standard deviation
    return float(image.std())


def compute_dynamic_range(image: np.ndarray) -> float:
    """
    Compute dynamic range of the image.

    Args:
        image: Input image as numpy array

    Returns:
        Dynamic range (0-1 normalized)
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0

    return float(image.max() - image.min())


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate noise level using median absolute deviation.

    Uses the difference between the image and a median-filtered version.

    Args:
        image: Input image as numpy array

    Returns:
        Estimated noise level (0-1)
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0

    # Simple noise estimation using local variance
    # Compute variance in small windows
    h, w = image.shape
    window_size = 3

    local_vars = []
    for i in range(0, h - window_size, window_size):
        for j in range(0, w - window_size, window_size):
            window = image[i:i+window_size, j:j+window_size]
            local_vars.append(window.var())

    if local_vars:
        # Use median of local variances as noise estimate
        noise = float(np.sqrt(np.median(local_vars)))
    else:
        noise = 0.0

    return noise


def detect_motion_blur(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Detect if image has significant motion blur.

    Uses Laplacian variance - low values indicate blur.

    Args:
        image: Input image as numpy array
        threshold: Laplacian variance threshold below which blur is detected

    Returns:
        True if motion blur is detected
    """
    lap_var = compute_laplacian_variance(image)
    return lap_var < threshold


def detect_compression_artifacts(
    image: np.ndarray | Image.Image,
    threshold: float = 0.15,
) -> bool:
    """
    Detect JPEG compression artifacts.

    Looks for block artifacts common in heavily compressed JPEGs.

    Args:
        image: Input image
        threshold: Detection threshold

    Returns:
        True if compression artifacts are detected
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    image = image.astype(np.float32)

    # Check for 8x8 block artifacts (JPEG block size)
    h, w = image.shape
    block_size = 8

    # Compute differences at block boundaries
    boundary_diffs = []

    # Horizontal boundaries
    for i in range(block_size, h - block_size, block_size):
        diff = np.abs(image[i-1, :] - image[i, :]).mean()
        boundary_diffs.append(diff)

    # Vertical boundaries
    for j in range(block_size, w - block_size, block_size):
        diff = np.abs(image[:, j-1] - image[:, j]).mean()
        boundary_diffs.append(diff)

    if boundary_diffs:
        # Compare boundary differences to non-boundary differences
        avg_boundary_diff = np.mean(boundary_diffs)
        overall_diff = compute_gradient_magnitude(image)

        if overall_diff > 0:
            ratio = avg_boundary_diff / overall_diff
            return ratio > threshold

    return False


def detect_watermark(
    image: np.ndarray | Image.Image,
    check_regions: list[tuple[float, float, float, float]] | None = None,
) -> bool:
    """
    Detect presence of watermarks in common locations.

    Args:
        image: Input image
        check_regions: List of (x_start, x_end, y_start, y_end) as fractions.
            Default checks bottom corners and center.

    Returns:
        True if watermark is likely present
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    h, w = image.shape

    if check_regions is None:
        # Default: check bottom 15% of image
        check_regions = [
            (0.0, 1.0, 0.85, 1.0),  # Bottom strip
            (0.0, 0.2, 0.8, 1.0),   # Bottom-left corner
            (0.8, 1.0, 0.8, 1.0),   # Bottom-right corner
        ]

    # Check for unusual patterns in suspected watermark regions
    for x_start, x_end, y_start, y_end in check_regions:
        region = image[
            int(y_start * h):int(y_end * h),
            int(x_start * w):int(x_end * w)
        ]

        if region.size == 0:
            continue

        # Check if region has suspiciously uniform or text-like patterns
        region_std = region.std()
        main_image = image[:int(h * 0.8), :]
        main_std = main_image.std()

        # If region is much more uniform or has very different contrast
        if main_std > 0 and (region_std / main_std < 0.3 or region_std / main_std > 3):
            return True

    return False


def assess_image_quality(
    image: np.ndarray | Image.Image,
    sharpness_weight: float = 0.3,
    contrast_weight: float = 0.25,
    noise_weight: float = 0.25,
    artifact_weight: float = 0.2,
) -> QualityMetrics:
    """
    Comprehensive image quality assessment.

    Computes multiple quality metrics and returns an overall score.

    Args:
        image: Input image (numpy array or PIL Image)
        sharpness_weight: Weight for sharpness in overall score
        contrast_weight: Weight for contrast in overall score
        noise_weight: Weight for noise (inverse) in overall score
        artifact_weight: Weight for artifact absence in overall score

    Returns:
        QualityMetrics object with all computed metrics
    """
    # Convert to numpy
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    metrics = QualityMetrics()
    warnings = []

    # Compute sharpness metrics
    metrics.laplacian_variance = compute_laplacian_variance(img_array)
    metrics.gradient_magnitude = compute_gradient_magnitude(img_array)

    # Compute contrast metrics
    metrics.contrast_rms = compute_contrast_rms(img_array)
    metrics.dynamic_range = compute_dynamic_range(img_array)

    # Compute noise metrics
    metrics.noise_estimate = estimate_noise(img_array)
    if metrics.noise_estimate > 0:
        metrics.snr_estimate = metrics.contrast_rms / metrics.noise_estimate
    else:
        metrics.snr_estimate = float('inf')

    # Detect artifacts
    metrics.has_motion_blur = detect_motion_blur(img_array)
    metrics.has_compression_artifacts = detect_compression_artifacts(img_array)
    metrics.has_watermark = detect_watermark(img_array)

    # Add warnings
    if metrics.has_motion_blur:
        warnings.append("Motion blur detected - image may be too blurry for analysis")
    if metrics.has_compression_artifacts:
        warnings.append("Compression artifacts detected - consider using higher quality images")
    if metrics.has_watermark:
        warnings.append("Possible watermark detected in image")
    if metrics.contrast_rms < 0.1:
        warnings.append("Low contrast image - may affect feature extraction")
    if metrics.laplacian_variance < 50:
        warnings.append("Low sharpness - image may be out of focus")

    metrics.warnings = warnings

    # Compute overall score (0-1)
    # Normalize individual metrics
    sharpness_score = min(metrics.laplacian_variance / 500.0, 1.0)
    contrast_score = min(metrics.contrast_rms / 0.25, 1.0)
    noise_score = max(1.0 - metrics.noise_estimate * 10, 0.0)
    artifact_score = 1.0 - (
        0.4 * metrics.has_motion_blur +
        0.3 * metrics.has_compression_artifacts +
        0.3 * metrics.has_watermark
    )

    # Weighted average
    metrics.overall_score = (
        sharpness_weight * sharpness_score +
        contrast_weight * contrast_score +
        noise_weight * noise_score +
        artifact_weight * artifact_score
    )

    return metrics


def detect_artifacts(
    image: np.ndarray | Image.Image,
) -> dict[str, bool]:
    """
    Detect various types of artifacts in an image.

    Convenience function that returns a dictionary of artifact flags.

    Args:
        image: Input image

    Returns:
        Dictionary with artifact detection results
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    return {
        "motion_blur": detect_motion_blur(img_array),
        "compression_artifacts": detect_compression_artifacts(img_array),
        "watermark": detect_watermark(img_array),
    }


def filter_by_quality(
    image_paths: list[str],
    min_score: float = 0.5,
    return_metrics: bool = False,
) -> list[str] | tuple[list[str], list[QualityMetrics]]:
    """
    Filter a list of images by quality score.

    Args:
        image_paths: List of paths to images
        min_score: Minimum quality score to accept
        return_metrics: Whether to also return quality metrics

    Returns:
        List of paths to acceptable images (and optionally their metrics)
    """

    accepted_paths = []
    accepted_metrics = []

    for path in image_paths:
        try:
            img = Image.open(path)
            metrics = assess_image_quality(img)

            if metrics.overall_score >= min_score:
                accepted_paths.append(path)
                accepted_metrics.append(metrics)
            else:
                logger.debug(f"Rejected {path}: quality score {metrics.overall_score:.2f} < {min_score}")

        except Exception as e:
            logger.warning(f"Failed to assess quality of {path}: {e}")

    logger.info(f"Accepted {len(accepted_paths)}/{len(image_paths)} images based on quality")

    if return_metrics:
        return accepted_paths, accepted_metrics
    return accepted_paths
