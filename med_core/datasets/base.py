"""
Base dataset classes for multimodal medical learning.

Defines the interface that all multimodal datasets must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseMultimodalDataset(ABC, Dataset):
    """
    Abstract base class for multimodal datasets.

    All multimodal datasets should inherit from this class and implement
    the required methods for loading images, tabular data, and labels.

    The dataset is designed to handle:
    - Medical images (various formats)
    - Tabular/structured data (from CSV/Excel)
    - Multiple label types (classification, regression)
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        tabular_data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        transform: Any = None,
        target_transform: Any = None,
    ):
        """
        Initialize base multimodal dataset.

        Args:
            image_paths: List of paths to image files
            tabular_data: Tabular features array (N, num_features)
            labels: Target labels array (N,) or (N, num_labels)
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
        """
        super().__init__()

        if len(image_paths) != len(tabular_data):
            raise ValueError(
                f"Mismatch between images ({len(image_paths)}) "
                f"and tabular data ({len(tabular_data)})"
            )

        if len(image_paths) != len(labels):
            raise ValueError(
                f"Mismatch between images ({len(image_paths)}) "
                f"and labels ({len(labels)})"
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.tabular_data = self._ensure_tensor(tabular_data, dtype=torch.float32)
        self.labels = self._ensure_tensor(labels, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _ensure_tensor(
        data: np.ndarray | torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert data to tensor if needed."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype)
        return data.to(dtype)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    @abstractmethod
    def load_image(self, path: Path) -> Image.Image | np.ndarray:
        """
        Load an image from the given path.

        This method should be overridden for specific image formats
        or loading requirements (e.g., DICOM, NIfTI).

        Args:
            path: Path to the image file

        Returns:
            Loaded image
        """
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, tabular_features, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)

        # Apply image transform
        if self.transform is not None:
            image = self.transform(image)

        # Get tabular features
        tabular = self.tabular_data[idx]

        # Get label
        label = self.labels[idx]

        # Apply target transform
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, tabular, label

    def get_tabular_dim(self) -> int:
        """Return the dimension of tabular features."""
        return self.tabular_data.shape[1]

    def get_num_classes(self) -> int:
        """Return the number of unique classes (for classification)."""
        return len(torch.unique(self.labels))

    def get_class_distribution(self) -> dict[int, int]:
        """Return the distribution of samples across classes."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute sample weights for balanced sampling.

        Useful for handling class imbalance in training.

        Returns:
            Tensor of sample weights (N,)
        """
        class_dist = self.get_class_distribution()
        total_samples = len(self)

        # Compute inverse frequency weights
        class_weights = {
            cls: total_samples / (len(class_dist) * count)
            for cls, count in class_dist.items()
        }

        # Assign weight to each sample based on its class
        sample_weights = torch.zeros(total_samples)
        for idx, label in enumerate(self.labels):
            sample_weights[idx] = class_weights[int(label)]

        return sample_weights

    def subset(self, indices: list[int]) -> "BaseMultimodalDataset":
        """
        Create a subset of the dataset.

        Args:
            indices: List of indices to include in subset

        Returns:
            New dataset containing only the specified indices
        """
        return type(self)(
            image_paths=[self.image_paths[i] for i in indices],
            tabular_data=self.tabular_data[indices],
            labels=self.labels[indices],
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        return {
            "num_samples": len(self),
            "tabular_dim": self.get_tabular_dim(),
            "num_classes": self.get_num_classes(),
            "class_distribution": self.get_class_distribution(),
            "tabular_mean": self.tabular_data.mean(dim=0).tolist(),
            "tabular_std": self.tabular_data.std(dim=0).tolist(),
        }
