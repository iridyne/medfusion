"""
Medical Dataset Module

This module implements a medical-grade dataset class for multimodal learning.

Key Features:
- Support for DICOM (.dcm), NumPy (.npy), and standard image formats (.png, .jpg)
- 16-bit grayscale DICOM handling
- Intensity normalization for medical images
- CSV-based clinical data loading
- Transform pipeline for preprocessing
- Type-safe implementation with Python 3.12+ hints
"""

from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Optional DICOM support
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


class MedicalDataset(Dataset):
    """
    Medical multimodal dataset combining imaging and clinical data.

    Supports:
    - DICOM 16-bit grayscale images
    - NumPy arrays
    - Standard image formats (PNG, JPG)
    - CSV-based clinical features
    - Intensity normalization
    - Custom transform pipelines
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        image_column: str = "image_path",
        label_column: str = "label",
        feature_columns: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        intensity_norm: Literal["minmax", "zscore", "percentile", "none"] = "percentile",
        percentile_range: tuple[float, float] = (1.0, 99.0),
        target_size: Optional[tuple[int, int]] = None,
    ):
        """
        Initialize medical dataset.

        Args:
            csv_path: Path to CSV file with metadata
            image_dir: Root directory containing images
            image_column: Column name for image paths
            label_column: Column name for labels
            feature_columns: List of clinical feature columns (None = auto-detect)
            transform: Optional transform pipeline
            intensity_norm: Intensity normalization method
            percentile_range: Percentile range for clipping (if using percentile norm)
            target_size: Target image size (H, W) for resizing
        """
        super().__init__()

        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform
        self.intensity_norm = intensity_norm
        self.percentile_range = percentile_range
        self.target_size = target_size

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            exclude_cols = {image_column, label_column, "patient_id", "study_id"}
            feature_columns = [col for col in self.df.columns if col not in exclude_cols]

        self.feature_columns = feature_columns

        # Validate columns
        required_cols = [image_column, label_column] + feature_columns
        missing_cols = set(required_cols) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        # Extract features and labels
        self.features = self.df[feature_columns].values.astype(np.float32)
        self.labels = self.df[label_column].values.astype(np.int64)
        self.image_paths = self.df[image_column].values

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def _load_image(self, path: str | Path) -> np.ndarray:
        """
        Load image from various formats.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array (H, W) or (H, W, C)
        """
        path = Path(path)

        # Handle relative paths
        if not path.is_absolute():
            path = self.image_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load based on extension
        suffix = path.suffix.lower()

        if suffix == ".dcm":
            if not PYDICOM_AVAILABLE:
                raise ImportError("pydicom is required for DICOM files. Install with: pip install pydicom")
            return self._load_dicom(path)

        elif suffix == ".npy":
            return np.load(path)

        elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            img = Image.open(path)
            return np.array(img)

        else:
            raise ValueError(f"Unsupported image format: {suffix}")

    def _load_dicom(self, path: Path) -> np.ndarray:
        """
        Load DICOM image with proper 16-bit handling.

        Args:
            path: Path to DICOM file

        Returns:
            Image array (H, W)
        """
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Apply rescale slope and intercept if available
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept

        return img

    def _normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity for medical images.

        Args:
            img: Input image array

        Returns:
            Normalized image in [0, 1] range
        """
        if self.intensity_norm == "none":
            return img

        elif self.intensity_norm == "minmax":
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            return img

        elif self.intensity_norm == "zscore":
            mean, std = img.mean(), img.std()
            if std > 0:
                img = (img - mean) / std
                # Clip to reasonable range and rescale to [0, 1]
                img = np.clip(img, -3, 3)
                img = (img + 3) / 6
            return img

        elif self.intensity_norm == "percentile":
            p_low, p_high = self.percentile_range
            v_low = np.percentile(img, p_low)
            v_high = np.percentile(img, p_high)
            img = np.clip(img, v_low, v_high)
            if v_high > v_low:
                img = (img - v_low) / (v_high - v_low)
            return img

        else:
            raise ValueError(f"Unknown normalization method: {self.intensity_norm}")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image: normalize, resize, convert to tensor.

        Args:
            img: Input image array

        Returns:
            Preprocessed image tensor (C, H, W)
        """
        # Normalize intensity
        img = self._normalize_intensity(img)

        # Ensure 2D or 3D
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # Add channel dimension (1, H, W)
        elif img.ndim == 3:
            if img.shape[2] in [1, 3]:  # (H, W, C)
                img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Resize if needed
        if self.target_size is not None:
            img_tensor = torch.from_numpy(img).float()
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = img_tensor.numpy()

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        return img_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, features, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)

        # Preprocess image
        img_tensor = self._preprocess_image(img)

        # Apply custom transform if provided
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # Get clinical features
        features = torch.from_numpy(self.features[idx]).float()

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, features, label

    def get_class_distribution(self) -> dict[int, int]:
        """Get distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_feature_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics for each clinical feature."""
        stats = {}
        for i, col in enumerate(self.feature_columns):
            feature_values = self.features[:, i]
            stats[col] = {
                "mean": float(np.mean(feature_values)),
                "std": float(np.std(feature_values)),
                "min": float(np.min(feature_values)),
                "max": float(np.max(feature_values)),
            }
        return stats
