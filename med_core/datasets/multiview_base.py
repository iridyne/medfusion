"""
Multi-view dataset base classes.

Extends the base dataset classes to support multiple images per patient.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from med_core.datasets.multiview_types import MultiViewConfig, ViewDict, ViewTensor

logger = logging.getLogger(__name__)


class BaseMultiViewDataset(ABC, Dataset):
    """
    Abstract base class for multi-view multimodal datasets.

    Supports multiple images per patient (e.g., different CT views, time series).

    Key features:
    - Handles variable number of views per sample
    - Supports missing view handling strategies
    - Maintains backward compatibility with single-view datasets

    Example:
        >>> config = MultiViewConfig(
        ...     view_names=["axial", "coronal", "sagittal"],
        ...     required_views=["axial"],
        ...     handle_missing="zero",
        ... )
        >>> dataset = MyMultiViewDataset(
        ...     image_paths=[
        ...         {"axial": Path("p1_axial.jpg"), "coronal": Path("p1_coronal.jpg")},
        ...         {"axial": Path("p2_axial.jpg"), "sagittal": Path("p2_sagittal.jpg")},
        ...     ],
        ...     tabular_data=np.array([[1, 2], [3, 4]]),
        ...     labels=np.array([0, 1]),
        ...     view_config=config,
        ... )
    """

    def __init__(
        self,
        image_paths: list[ViewDict],
        tabular_data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        view_config: MultiViewConfig,
        transform: Any = None,
        target_transform: Any = None,
    ):
        """
        Initialize multi-view dataset.

        Args:
            image_paths: List of ViewDict, one per sample
                Example: [{"axial": Path(...), "coronal": Path(...)}, ...]
            tabular_data: Tabular features array (N, num_features)
            labels: Target labels array (N,) or (N, num_labels)
            view_config: Multi-view configuration
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
        """
        super().__init__()

        # Validate data consistency
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

        self.image_paths = image_paths
        self.view_config = view_config
        self.tabular_data = self._ensure_tensor(tabular_data, dtype=torch.float32)
        self.labels = self._ensure_tensor(labels, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

        # Validate samples
        self._validate_samples()

    def _validate_samples(self):
        """Validate that all samples meet the view requirements."""
        invalid_samples = []

        for idx, view_dict in enumerate(self.image_paths):
            if not self.view_config.validate_sample(view_dict):
                invalid_samples.append(idx)

        if invalid_samples:
            logger.warning(
                f"Found {len(invalid_samples)} samples not meeting view requirements. "
                f"First few indices: {invalid_samples[:5]}"
            )

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
        Load a single image from the given path.

        This method should be overridden for specific image formats
        or loading requirements (e.g., DICOM, NIfTI).

        Args:
            path: Path to the image file

        Returns:
            Loaded image
        """
        pass

    def _handle_missing_view(
        self,
        view_name: str,
        idx: int,
        reference_image: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Handle missing view according to the configured strategy.

        Args:
            view_name: Name of the missing view
            idx: Sample index
            reference_image: Optional reference image for duplication

        Returns:
            Tensor to use for missing view, or None if skipping
        """
        strategy = self.view_config.handle_missing

        if strategy == "skip":
            return None

        elif strategy == "zero":
            # Return zero tensor with same shape as reference
            if reference_image is not None:
                return torch.zeros_like(reference_image)
            else:
                # Default shape (will need to be handled by model)
                return torch.zeros(3, 224, 224)

        elif strategy == "duplicate":
            # Duplicate the first available view
            if reference_image is not None:
                return reference_image.clone()
            else:
                # Try to find any available view in this sample
                view_dict = self.image_paths[idx]
                for v_name in self.view_config.view_names:
                    if v_name in view_dict and view_dict[v_name] is not None:
                        img = self.load_image(view_dict[v_name])
                        if self.transform:
                            img = self.transform(img)
                        return img
                # No views available, return zero
                return torch.zeros(3, 224, 224)

        else:
            raise ValueError(f"Unknown missing view strategy: {strategy}")

    def __getitem__(self, idx: int) -> tuple[ViewTensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of:
                - images: Dict of view_name -> image tensor
                - tabular_features: Tabular feature tensor
                - label: Label tensor
        """
        view_dict = self.image_paths[idx]
        images = {}
        reference_image = None

        # Load all views
        for view_name in self.view_config.view_names:
            if view_name in view_dict and view_dict[view_name] is not None:
                # Load and transform image
                img = self.load_image(view_dict[view_name])
                if self.transform is not None:
                    img = self.transform(img)

                images[view_name] = img

                # Keep first image as reference for missing views
                if reference_image is None:
                    reference_image = img
            else:
                # Handle missing view
                missing_img = self._handle_missing_view(view_name, idx, reference_image)
                if missing_img is not None:
                    images[view_name] = missing_img

        # Get tabular features
        tabular = self.tabular_data[idx]

        # Get label
        label = self.labels[idx]

        # Apply target transform
        if self.target_transform is not None:
            label = self.target_transform(label)

        return images, tabular, label

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

    def get_available_views(self, idx: int) -> list[str]:
        """
        Get list of available views for a specific sample.

        Args:
            idx: Sample index

        Returns:
            List of view names that are available (not None)
        """
        return self.view_config.get_available_views(self.image_paths[idx])

    def get_view_statistics(self) -> dict[str, Any]:
        """
        Compute statistics about view availability across the dataset.

        Returns:
            Dictionary with view statistics:
                - view_counts: Number of samples with each view
                - view_percentages: Percentage of samples with each view
                - samples_with_all_views: Number of samples with all views
                - samples_with_required_views: Number of samples with required views
        """
        view_counts = dict.fromkeys(self.view_config.view_names, 0)
        samples_with_all = 0
        samples_with_required = 0

        for view_dict in self.image_paths:
            available = self.view_config.get_available_views(view_dict)

            # Count each view
            for view in available:
                if view in view_counts:
                    view_counts[view] += 1

            # Check if all views present
            if len(available) == len(self.view_config.view_names):
                samples_with_all += 1

            # Check if required views present
            if all(v in available for v in self.view_config.required_views):
                samples_with_required += 1

        total = len(self)
        view_percentages = {
            view: (count / total * 100) if total > 0 else 0
            for view, count in view_counts.items()
        }

        return {
            "view_counts": view_counts,
            "view_percentages": view_percentages,
            "samples_with_all_views": samples_with_all,
            "samples_with_required_views": samples_with_required,
            "total_samples": total,
        }

    def subset(self, indices: list[int]) -> "BaseMultiViewDataset":
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
            view_config=self.view_config,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute comprehensive dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "num_samples": len(self),
            "tabular_dim": self.get_tabular_dim(),
            "num_classes": self.get_num_classes(),
            "class_distribution": self.get_class_distribution(),
            "tabular_mean": self.tabular_data.mean(dim=0).tolist(),
            "tabular_std": self.tabular_data.std(dim=0).tolist(),
        }

        # Add view statistics
        stats["view_statistics"] = self.get_view_statistics()

        return stats
