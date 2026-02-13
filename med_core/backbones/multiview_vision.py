"""
Multi-view vision backbone wrapper.

Extends vision backbones to support multiple images per sample by:
1. Processing each view through a shared or separate backbone
2. Aggregating view features using configurable strategies
3. Maintaining backward compatibility with single-view inputs
"""

from typing import Any

import torch
import torch.nn as nn

from med_core.backbones.base import BaseVisionBackbone
from med_core.backbones.view_aggregator import (
    BaseViewAggregator,
    create_view_aggregator,
)
from med_core.datasets.multiview_types import ViewTensor


class MultiViewVisionBackbone(nn.Module):
    """
    Wrapper that enables any vision backbone to process multiple views.

    Architecture:
        Input: dict[view_name, Tensor(B, C, H, W)] or Tensor(B, N, C, H, W)
        ↓
        Per-view feature extraction (shared or separate backbone)
        ↓
        View aggregation (attention, pooling, etc.)
        ↓
        Output: Tensor(B, feature_dim)

    Args:
        backbone: Base vision backbone to use for feature extraction
        aggregator: View aggregation strategy
        share_weights: If True, use same backbone for all views (default: True)
        view_names: Optional list of expected view names for validation

    Example:
        >>> from med_core.backbones.vision import ResNetBackbone
        >>> base_backbone = ResNetBackbone(variant="resnet18", feature_dim=128)
        >>> multiview_backbone = MultiViewVisionBackbone(
        ...     backbone=base_backbone,
        ...     aggregator="attention",
        ...     share_weights=True
        ... )
        >>>
        >>> # Multi-view input
        >>> views = {
        ...     "axial": torch.randn(4, 3, 224, 224),
        ...     "coronal": torch.randn(4, 3, 224, 224),
        ...     "sagittal": torch.randn(4, 3, 224, 224),
        ... }
        >>> features = multiview_backbone(views)  # (4, 128)
        >>>
        >>> # Single-view input (backward compatible)
        >>> single_view = torch.randn(4, 3, 224, 224)
        >>> features = multiview_backbone(single_view)  # (4, 128)
    """

    def __init__(
        self,
        backbone: BaseVisionBackbone,
        aggregator: BaseViewAggregator | str = "attention",
        share_weights: bool = True,
        view_names: list[str] | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.share_weights = share_weights
        self.view_names = view_names
        self._output_dim = backbone.output_dim

        # Create aggregator
        if isinstance(aggregator, str):
            self.aggregator = create_view_aggregator(
                aggregator_type=aggregator,
                feature_dim=backbone.output_dim,
            )
        else:
            self.aggregator = aggregator

        # If not sharing weights, create separate backbones for each view
        if not share_weights and view_names is not None:
            self.view_backbones = nn.ModuleDict({
                view_name: self._clone_backbone(backbone)
                for view_name in view_names
            })
        else:
            self.view_backbones = None

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    def _clone_backbone(self, backbone: BaseVisionBackbone) -> BaseVisionBackbone:
        """Create a deep copy of the backbone with separate parameters."""
        import copy
        return copy.deepcopy(backbone)

    def _process_single_view(
        self,
        view_name: str,
        view_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a single view through the backbone.

        Args:
            view_name: Name of the view
            view_tensor: Image tensor (B, C, H, W)

        Returns:
            Feature tensor (B, feature_dim)
        """
        if self.view_backbones is not None and view_name in self.view_backbones:
            # Use view-specific backbone
            return self.view_backbones[view_name](view_tensor)
        else:
            # Use shared backbone
            return self.backbone(view_tensor)

    def forward(
        self,
        x: ViewTensor | torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-view vision backbone.

        Args:
            x: Input can be:
                - dict[str, Tensor]: Dictionary of view tensors, each (B, C, H, W)
                - Tensor(B, N, C, H, W): Stacked view tensor
                - Tensor(B, C, H, W): Single view (backward compatible)
            view_mask: Optional boolean mask (B, N) indicating valid views

        Returns:
            Aggregated feature tensor (B, feature_dim)
        """
        # Handle single-view input (backward compatibility)
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            # Single view: (B, C, H, W)
            return self.backbone(x)

        # Handle multi-view dictionary input
        if isinstance(x, dict):
            batch_size = next(iter(x.values())).size(0)
            num_views = len(x)

            # Validate view names if specified
            if self.view_names is not None:
                unexpected_views = set(x.keys()) - set(self.view_names)
                if unexpected_views:
                    raise ValueError(
                        f"Unexpected view names: {unexpected_views}. "
                        f"Expected: {self.view_names}"
                    )

            # Process each view
            view_features = []
            view_names_ordered = []

            for view_name, view_tensor in x.items():
                if view_tensor.dim() != 4:
                    raise ValueError(
                        f"Expected 4D tensor for view '{view_name}', "
                        f"got shape {view_tensor.shape}"
                    )

                features = self._process_single_view(view_name, view_tensor)
                view_features.append(features)
                view_names_ordered.append(view_name)

            # Stack features: (B, N, feature_dim)
            view_features = torch.stack(view_features, dim=1)

            # Create view mask if not provided (all views valid)
            if view_mask is None:
                view_mask = torch.ones(
                    batch_size, num_views,
                    dtype=torch.bool,
                    device=view_features.device
                )

        # Handle stacked tensor input
        elif isinstance(x, torch.Tensor) and x.dim() == 5:
            # Stacked views: (B, N, C, H, W)
            batch_size, num_views, C, H, W = x.shape

            # Reshape to (B*N, C, H, W) for batch processing
            x_flat = x.view(batch_size * num_views, C, H, W)

            # Process all views through backbone
            features_flat = self.backbone(x_flat)  # (B*N, feature_dim)

            # Reshape back to (B, N, feature_dim)
            view_features = features_flat.view(batch_size, num_views, -1)

            # Create view mask if not provided
            if view_mask is None:
                view_mask = torch.ones(
                    batch_size, num_views,
                    dtype=torch.bool,
                    device=view_features.device
                )

        else:
            raise ValueError(
                f"Unsupported input type or shape. Expected:\n"
                f"  - dict[str, Tensor(B, C, H, W)]\n"
                f"  - Tensor(B, N, C, H, W)\n"
                f"  - Tensor(B, C, H, W) for single view\n"
                f"Got: {type(x)} with shape {x.shape if isinstance(x, torch.Tensor) else 'N/A'}"
            )

        # Aggregate view features
        aggregated_features = self.aggregator(view_features, view_mask)

        return aggregated_features

    def freeze_backbone(self, strategy: str = "full", unfreeze_last_n: int = 2) -> None:
        """
        Freeze backbone parameters.

        Args:
            strategy: Freezing strategy ("full", "partial", "none")
            unfreeze_last_n: Number of last layers to keep trainable
        """
        if self.view_backbones is not None:
            # Freeze all view-specific backbones
            for backbone in self.view_backbones.values():
                backbone.freeze_backbone(strategy, unfreeze_last_n)
        else:
            # Freeze shared backbone
            self.backbone.freeze_backbone(strategy, unfreeze_last_n)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        if self.view_backbones is not None:
            for backbone in self.view_backbones.values():
                backbone.unfreeze_backbone()
        else:
            self.backbone.unfreeze_backbone()

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "backbone_config": self.backbone.get_config(),
            "aggregator_type": self.aggregator.__class__.__name__,
            "share_weights": self.share_weights,
            "view_names": self.view_names,
            "output_dim": self.output_dim,
        }


def create_multiview_vision_backbone(
    backbone_name: str,
    aggregator_type: str = "attention",
    share_weights: bool = True,
    view_names: list[str] | None = None,
    **backbone_kwargs,
) -> MultiViewVisionBackbone:
    """
    Factory function to create a multi-view vision backbone.

    Args:
        backbone_name: Name of the base backbone (e.g., "resnet18", "efficientnet_b0")
        aggregator_type: Type of view aggregator ("max", "mean", "attention", etc.)
        share_weights: Whether to share backbone weights across views
        view_names: Optional list of expected view names
        **backbone_kwargs: Additional arguments for the backbone constructor

    Returns:
        Configured MultiViewVisionBackbone instance

    Example:
        >>> backbone = create_multiview_vision_backbone(
        ...     backbone_name="resnet18",
        ...     aggregator_type="attention",
        ...     share_weights=True,
        ...     view_names=["axial", "coronal", "sagittal"],
        ...     pretrained=True,
        ...     feature_dim=128,
        ... )
    """
    from med_core.backbones.factory import create_vision_backbone

    # Create base backbone
    base_backbone = create_vision_backbone(backbone_name, **backbone_kwargs)

    # Wrap with multi-view support
    multiview_backbone = MultiViewVisionBackbone(
        backbone=base_backbone,
        aggregator=aggregator_type,
        share_weights=share_weights,
        view_names=view_names,
    )

    return multiview_backbone
