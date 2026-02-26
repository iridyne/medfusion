"""
Medical Image Preprocessing Utilities
=====================================
Common preprocessing functions for medical images.

This module provides:
- intensity normalization
- center cropping
- CLAHE
- watermark removal
- a configurable ImagePreprocessor with both callable behavior and a
  `preprocess` method that returns a NumPy array (useful for tests and
  downstream pipelines that expect arrays).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

if TYPE_CHECKING:
    # Helpful aliases for type-checkers only
    from numpy import ndarray as NPArray  # type: ignore
    from PIL.Image import Image as PILImageType  # type: ignore
else:
    # At runtime we rely on concrete runtime types; the TYPE_CHECKING branch
    # above keeps static analyzers happy without affecting runtime behavior.
    NPArray = Any  # type: ignore
    PILImageType = Image  # runtime reference to PIL Image type for annotations

logger = logging.getLogger(__name__)


def normalize_intensity(
    image: NPArray,
    method: Literal["minmax", "zscore", "percentile"] = "minmax",
    percentile_range: tuple[float, float] = (1.0, 99.0),
) -> NPArray:
    """
    Normalize image intensity values.

    Args:
        image: Input image array
        method: Normalization method
        percentile_range: (low, high) percentiles for percentile method

    Returns:
        Normalized image (0-1 range)
    """
    img = np.asarray(image).astype(np.float32)

    if method == "minmax":
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            return (img - min_val) / (max_val - min_val)
        return np.zeros_like(img)

    if method == "zscore":
        mean, std = img.mean(), img.std()
        if std > 0:
            normalized = (img - mean) / std
            normalized = np.clip(normalized, -3, 3)
            return (normalized + 3) / 6
        return img - mean

    if method == "percentile":
        p_low, p_high = percentile_range
        low, high = np.percentile(img, p_low), np.percentile(img, p_high)
        if high > low:
            return np.clip((img - low) / (high - low), 0, 1)
        return np.zeros_like(img)

    raise ValueError(f"Unknown method: {method}")


def crop_center(
    image: NPArray | Image.Image, size: int | tuple[int, int],
) -> Image.Image:
    """
    Crop center region of image.

    Args:
        image: Input image (NumPy array or PIL Image)
        size: Target size (int or (width, height))

    Returns:
        Cropped PIL Image
    """
    # If NumPy array provided, convert to PIL Image first
    if isinstance(image, np.ndarray):
        arr = image
        if getattr(arr, "dtype", None) is not None and arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
                np.uint8,
            )
        image = Image.fromarray(arr)

    if not isinstance(image, Image.Image):
        raise TypeError(
            "`image` must be a PIL Image or numpy array convertible to one.",
        )

    width, height = image.size
    target_w, target_h = (size, size) if isinstance(size, int) else size

    left = max((width - target_w) // 2, 0)
    top = max((height - target_h) // 2, 0)
    right = left + target_w
    bottom = top + target_h

    return image.crop((left, top, right, bottom))


def apply_clahe(
    image: NPArray | Image.Image,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> NPArray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Input image (NumPy array or PIL Image)
        clip_limit: Contrast limit
        tile_size: Grid size for CLAHE

    Returns:
        Enhanced image as NumPy array (dtype uint8)
    """
    # Ensure we have a NumPy array (uint8)
    if isinstance(image, Image.Image):
        arr = np.array(image)
    else:
        arr = np.asarray(image)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    if arr.ndim == 3 and arr.shape[2] == 3:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale
        enhanced = clahe.apply(arr)

    return enhanced


def remove_bottom_watermark(
    image: Image.Image,
    watermark_height_ratio: float = 0.15,
) -> Image.Image:
    """
    Remove bottom watermark by cropping.

    Args:
        image: Input PIL Image
        watermark_height_ratio: Ratio of image height to remove from bottom

    Returns:
        Cropped image
    """
    if not isinstance(image, Image.Image):
        raise TypeError("`image` must be a PIL Image")
    width, height = image.size
    crop_height = int(height * (1 - watermark_height_ratio))
    return image.crop((0, 0, width, crop_height))


class ImagePreprocessor:
    """
    Configurable image preprocessing pipeline.

    Example:
        >>> preprocessor = ImagePreprocessor(
        ...     normalize_method="percentile",
        ...     apply_clahe=True,
        ...     output_size=(224, 224),
        ... )
        >>> processed_array = preprocessor.preprocess(image_array)
    """

    def __init__(
        self,
        normalize_method: Literal[
            "minmax", "zscore", "percentile", "none",
        ] = "percentile",
        apply_clahe: bool = False,
        remove_watermark: bool = False,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        # Validate normalize_method
        valid_methods = ["minmax", "zscore", "percentile", "none"]
        if normalize_method not in valid_methods:
            raise ValueError(
                f"Invalid normalize_method: {normalize_method}. "
                f"Must be one of {valid_methods}",
            )

        self.normalize_method = normalize_method
        self.apply_clahe = apply_clahe
        self.remove_watermark = remove_watermark
        self.output_size = output_size

    def __call__(self, image: PILImageType | NPArray | str | Path) -> Image.Image:
        """
        Apply preprocessing pipeline and return a PIL Image.

        Note: use `preprocess` to get a NumPy array output (tests and some
        downstream code expect arrays).
        """
        # Load from path if necessary
        if isinstance(image, (str, Path)):
            image = Image.open(image)  # type: ignore[return-value]

        # Convert numpy arrays to PIL
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
                    np.uint8,
                )
            image = Image.fromarray(arr)

        if not isinstance(image, Image.Image):
            raise TypeError(
                "`image` must be a PIL Image or a numpy array convertible to one.",
            )

        # Ensure RGB
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")

        # Remove watermark
        if self.remove_watermark:
            image = remove_bottom_watermark(image)

        # Apply CLAHE
        if self.apply_clahe:
            img_array = np.array(image)
            img_array = apply_clahe(img_array)
            image = Image.fromarray(img_array)

        # Resize
        if self.output_size:
            # PIL.Image.Resampling introduced in newer Pillow versions; fallback if needed
            resample = getattr(Image, "Resampling", None)
            if resample is not None:
                image = image.resize(self.output_size, resample.LANCZOS)
            else:
                image = image.resize(self.output_size, Image.LANCZOS)

        # Normalize if requested
        if self.normalize_method != "none":
            img_array = np.array(image, dtype=np.float32)
            normalized = normalize_intensity(img_array, method=self.normalize_method)
            image = Image.fromarray((normalized * 255).astype(np.uint8))

        return image

    def preprocess(self, image: NPArray | PILImageType | str | Path) -> np.ndarray:
        """
        Apply the pipeline and return a NumPy array.

        Behavior:
        - If input was a 2D array, returns a 2D array (grayscale) with the same
          spatial dimensions.
        - If input had 3 channels, returns a HxWx3 array.
        - If input was a PIL Image, behavior follows the same conversion rules.

        Tests in the repo assume `preprocess` exists and returns an array with
        the same shape as the original NumPy input (at least in terms of
        spatial dimensions), so we preserve that contract.
        """
        orig_was_numpy = isinstance(image, np.ndarray)
        orig_shape = None
        if orig_was_numpy:
            orig_shape = image.shape

        pil_img = self(image)  # get a PIL.Image

        arr = np.array(pil_img)

        # If original input was 2D, convert to 2D grayscale by averaging channels
        if orig_was_numpy and orig_shape is not None and (len(orig_shape) == 2):
            # Convert RGB to grayscale preserving spatial dims
            if arr.ndim == 3:
                arr_gray = arr.mean(axis=2).astype(arr.dtype)
            else:
                arr_gray = arr
            # Ensure shape matches original spatial dims
            return arr_gray.reshape(orig_shape)

        return arr

    def process_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path,
        suffix: str = "_processed",
    ) -> list[Path]:
        """Process batch of images and save to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_paths: list[Path] = []
        for img_path in image_paths:
            img_path = Path(img_path)
            try:
                processed = self(img_path)
                output_path = output_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
                processed.save(output_path)
                processed_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

        logger.info(f"Processed {len(processed_paths)}/{len(image_paths)} images")
        return processed_paths
