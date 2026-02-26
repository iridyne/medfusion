"""
Multi-view data types and configurations.

Defines the core data structures and interfaces for multi-view/multi-image support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

# Type aliases for multi-view data
ViewDict = dict[str, Path | None]  # View name -> image path (None if missing)
ViewTensor = dict[str, torch.Tensor]  # View name -> image tensor


@dataclass
class MultiViewConfig:
    """
    Configuration for multi-view datasets.

    Defines how to handle multiple images per patient, including:
    - Which views are expected
    - How to handle missing views
    - Maximum number of views to support

    Example:
        >>> config = MultiViewConfig(
        ...     view_names=["axial", "coronal", "sagittal"],
        ...     required_views=["axial"],  # Only axial is mandatory
        ...     handle_missing="zero",
        ... )
    """

    # View definitions
    view_names: list[str] = field(default_factory=lambda: ["default"])
    """List of expected view names (e.g., ["axial", "coronal", "sagittal"])"""

    required_views: list[str] = field(default_factory=list)
    """Views that must be present. Empty list means all views are optional."""

    # Missing view handling
    handle_missing: Literal["skip", "zero", "duplicate"] = "zero"
    """
    Strategy for handling missing views:
    - "skip": Skip samples with missing required views
    - "zero": Replace missing views with zero tensors
    - "duplicate": Duplicate the first available view
    """

    # Constraints
    max_views: int = 10
    """Maximum number of views per sample"""

    min_views: int = 1
    """Minimum number of views required per sample"""

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Validate required views are in view_names
        for view in self.required_views:
            if view not in self.view_names:
                raise ValueError(
                    f"Required view '{view}' not in view_names: {self.view_names}",
                )

        if self.max_views < self.min_views:
            raise ValueError(
                f"max_views ({self.max_views}) must be >= min_views ({self.min_views})",
            )

        if len(self.view_names) > self.max_views:
            raise ValueError(
                f"Number of view_names ({len(self.view_names)}) exceeds max_views ({self.max_views})",
            )

    def is_view_required(self, view_name: str) -> bool:
        """Check if a view is required."""
        return view_name in self.required_views

    def validate_sample(self, view_dict: ViewDict) -> bool:
        """
        Validate that a sample meets the requirements.

        Args:
            view_dict: Dictionary of view names to paths

        Returns:
            True if sample is valid, False otherwise
        """
        # Check required views
        if not all(view in view_dict and view_dict[view] is not None for view in self.required_views):
            return False

        # Check minimum views
        available_views = sum(1 for v in view_dict.values() if v is not None)
        return available_views >= self.min_views

    def get_available_views(self, view_dict: ViewDict) -> list[str]:
        """Get list of available (non-None) views in a sample."""
        return [name for name, path in view_dict.items() if path is not None]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "view_names": self.view_names,
            "required_views": self.required_views,
            "handle_missing": self.handle_missing,
            "max_views": self.max_views,
            "min_views": self.min_views,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiViewConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def single_view(cls, view_name: str = "default") -> "MultiViewConfig":
        """
        Create a single-view configuration (for backward compatibility).

        Args:
            view_name: Name of the single view

        Returns:
            MultiViewConfig with single view
        """
        return cls(
            view_names=[view_name],
            required_views=[view_name],
            handle_missing="zero",
            min_views=1,
            max_views=1,
        )


def create_single_view_dict(image_path: Path, view_name: str = "default") -> ViewDict:
    """
    Convert a single image path to ViewDict format.

    Helper function for backward compatibility with single-image datasets.

    Args:
        image_path: Path to single image
        view_name: Name to assign to this view

    Returns:
        ViewDict with single entry
    """
    return {view_name: image_path}


def convert_to_multiview_paths(
    image_paths: list[Path],
    view_name: str = "default",
) -> list[ViewDict]:
    """
    Convert list of single image paths to multi-view format.

    Args:
        image_paths: List of single image paths
        view_name: Name to assign to all images

    Returns:
        List of ViewDict, one per image
    """
    return [create_single_view_dict(path, view_name) for path in image_paths]
