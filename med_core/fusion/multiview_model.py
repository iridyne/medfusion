"""
Multi-view multimodal fusion model.

Extends the base MultiModalFusionModel to support multi-view vision inputs
while maintaining backward compatibility with single-view inputs.
"""

from typing import Any

import torch
from torch import nn

from med_core.backbones.multiview_vision import MultiViewVisionBackbone
from med_core.datasets.multiview_types import ViewTensor
from med_core.fusion.base import BaseFusion


class MultiViewMultiModalFusionModel(nn.Module):
    """
    Complete multimodal fusion model with multi-view vision support.

    Architecture:
        Multi-view images → MultiViewVisionBackbone → vision_features (B, vision_dim)
        Tabular data → TabularBackbone → tabular_features (B, tabular_dim)
        Both features → FusionModule → fused_features (B, fusion_dim)
        Fused features → Classifier → logits (B, num_classes)

    Key differences from MultiModalFusionModel:
        - Accepts dict[str, Tensor] or Tensor(B, N, C, H, W) for images
        - Vision backbone must be MultiViewVisionBackbone
        - Returns view-level auxiliary outputs for interpretability

    Args:
        vision_backbone: Multi-view vision feature extractor
        tabular_backbone: Tabular feature extractor
        fusion_module: Fusion strategy
        num_classes: Number of output classes
        dropout: Dropout rate for classifier
        use_auxiliary_heads: Whether to include per-modality classifiers

    Example:
        >>> from med_core.backbones import create_multiview_vision_backbone, create_tabular_backbone
        >>> from med_core.fusion import GatedFusion
        >>>
        >>> vision_backbone = create_multiview_vision_backbone(
        ...     backbone_name="resnet18",
        ...     aggregator_type="attention",
        ...     feature_dim=128,
        ... )
        >>> tabular_backbone = create_tabular_backbone(input_dim=10, output_dim=32)
        >>> fusion_module = GatedFusion(vision_dim=128, tabular_dim=32, output_dim=64)
        >>>
        >>> model = MultiViewMultiModalFusionModel(
        ...     vision_backbone=vision_backbone,
        ...     tabular_backbone=tabular_backbone,
        ...     fusion_module=fusion_module,
        ...     num_classes=2,
        ... )
        >>>
        >>> # Multi-view input
        >>> images = {
        ...     "axial": torch.randn(4, 3, 224, 224),
        ...     "coronal": torch.randn(4, 3, 224, 224),
        ... }
        >>> tabular = torch.randn(4, 10)
        >>> outputs = model(images, tabular)
    """

    def __init__(
        self,
        vision_backbone: MultiViewVisionBackbone | nn.Module,
        tabular_backbone: nn.Module,
        fusion_module: BaseFusion,
        num_classes: int = 2,
        dropout: float = 0.4,
        use_auxiliary_heads: bool = True,
    ):
        super().__init__()

        self.vision_backbone = vision_backbone
        self.tabular_backbone = tabular_backbone
        self.fusion_module = fusion_module
        self.use_auxiliary_heads = use_auxiliary_heads
        self.num_classes = num_classes

        # Check if vision backbone supports multi-view
        self.is_multiview = isinstance(vision_backbone, MultiViewVisionBackbone)

        # Main classifier
        fusion_dim = fusion_module.output_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

        # Auxiliary classifiers (for multi-task learning)
        if use_auxiliary_heads:
            vision_dim = vision_backbone.output_dim
            tabular_dim = tabular_backbone.output_dim

            self.vision_classifier = nn.Linear(vision_dim, num_classes)
            self.tabular_classifier = nn.Linear(tabular_dim, num_classes)

    def forward(
        self,
        images: ViewTensor | torch.Tensor,
        tabular: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the full multimodal model.

        Args:
            images: Image input, can be:
                - dict[str, Tensor]: Multi-view images, each (B, C, H, W)
                - Tensor(B, N, C, H, W): Stacked multi-view images
                - Tensor(B, C, H, W): Single-view images (backward compatible)
            tabular: Tabular input tensor (B, num_features)
            view_mask: Optional boolean mask (B, N) for valid views

        Returns:
            Dictionary containing:
                - logits: Main fusion classifier output (B, num_classes)
                - vision_features: Aggregated vision features (B, vision_dim)
                - tabular_features: Tabular backbone features (B, tabular_dim)
                - fused_features: Fused features (B, fusion_dim)
                - vision_logits: Vision-only classifier output (if auxiliary heads)
                - tabular_logits: Tabular-only classifier output (if auxiliary heads)
                - fusion_aux: Auxiliary fusion outputs (attention weights, etc.)
                - view_aggregation_aux: View aggregation info (if multi-view)
        """
        # Extract features from each modality
        if self.is_multiview:
            # Multi-view vision backbone
            vision_features = self.vision_backbone(images, view_mask=view_mask)

            # Get view aggregation auxiliary outputs if available
            view_aggregation_aux = None
            if hasattr(self.vision_backbone.aggregator, "last_attention_weights"):
                view_aggregation_aux = {
                    "view_attention_weights": self.vision_backbone.aggregator.last_attention_weights,
                }
        else:
            # Single-view vision backbone (backward compatible)
            if isinstance(images, dict):
                raise ValueError(
                    "Multi-view input provided but vision_backbone is not MultiViewVisionBackbone",
                )
            vision_features = self.vision_backbone(images)
            view_aggregation_aux = None

        tabular_features = self.tabular_backbone(tabular)

        # Fuse features
        fused_features, fusion_aux = self.fusion_module(
            vision_features, tabular_features,
        )

        # Main classification
        logits = self.classifier(fused_features)

        # Build output dict
        outputs = {
            "logits": logits,
            "vision_features": vision_features,
            "tabular_features": tabular_features,
            "fused_features": fused_features,
        }

        if fusion_aux is not None:
            outputs["fusion_aux"] = fusion_aux

        if view_aggregation_aux is not None:
            outputs["view_aggregation_aux"] = view_aggregation_aux

        # Auxiliary classifications
        if self.use_auxiliary_heads:
            outputs["vision_logits"] = self.vision_classifier(vision_features)
            outputs["tabular_logits"] = self.tabular_classifier(tabular_features)

        return outputs

    def get_attention_weights(self) -> dict[str, torch.Tensor]:
        """
        Get all attention weights from the model.

        Returns:
            Dictionary containing:
                - fusion_attention: Attention weights from fusion module (if available)
                - view_attention: Attention weights from view aggregation (if multi-view)
        """
        attention_weights = {}

        # Fusion attention
        if hasattr(self.fusion_module, "get_attention_weights"):
            fusion_attn = self.fusion_module.get_attention_weights()
            if fusion_attn is not None:
                attention_weights["fusion_attention"] = fusion_attn

        # View aggregation attention
        if self.is_multiview and hasattr(
            self.vision_backbone.aggregator, "last_attention_weights",
        ):
            attention_weights["view_attention"] = (
                self.vision_backbone.aggregator.last_attention_weights
            )

        return attention_weights or None

    def freeze_vision_backbone(
        self, strategy: str = "full", unfreeze_last_n: int = 2,
    ) -> None:
        """
        Freeze vision backbone parameters.

        Args:
            strategy: Freezing strategy ("full", "partial", "none")
            unfreeze_last_n: Number of last layers to keep trainable
        """
        if hasattr(self.vision_backbone, "freeze_backbone"):
            self.vision_backbone.freeze_backbone(strategy, unfreeze_last_n)

    def unfreeze_vision_backbone(self) -> None:
        """Unfreeze all vision backbone parameters."""
        if hasattr(self.vision_backbone, "unfreeze_backbone"):
            self.vision_backbone.unfreeze_backbone()

    def get_config(self) -> dict[str, Any]:
        """Return model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "use_auxiliary_heads": self.use_auxiliary_heads,
            "is_multiview": self.is_multiview,
            "fusion_config": self.fusion_module.get_config(),
        }

        if hasattr(self.vision_backbone, "get_config"):
            config["vision_config"] = self.vision_backbone.get_config()
        if hasattr(self.tabular_backbone, "get_config"):
            config["tabular_config"] = self.tabular_backbone.get_config()

        return config


def create_multiview_fusion_model(
    vision_backbone_name: str,
    tabular_input_dim: int,
    fusion_type: str = "gated",
    num_classes: int = 2,
    vision_feature_dim: int = 128,
    tabular_feature_dim: int = 32,
    fusion_output_dim: int = 64,
    aggregator_type: str = "attention",
    share_weights: bool = True,
    view_names: list[str] | None = None,
    **kwargs: Any,
) -> MultiViewMultiModalFusionModel:
    """
    Factory function to create a complete multi-view multimodal fusion model.

    Args:
        vision_backbone_name: Name of vision backbone (e.g., "resnet18")
        tabular_input_dim: Input dimension for tabular data
        fusion_type: Type of fusion strategy ("concatenate", "gated", "attention", etc.)
        num_classes: Number of output classes
        vision_feature_dim: Output dimension of vision backbone
        tabular_feature_dim: Output dimension of tabular backbone
        fusion_output_dim: Output dimension of fusion module
        aggregator_type: Type of view aggregator ("attention", "mean", etc.)
        share_weights: Whether to share backbone weights across views
        view_names: Optional list of expected view names
        **kwargs: Additional arguments for backbones and fusion

    Returns:
        Configured MultiViewMultiModalFusionModel instance

    Example:
        >>> model = create_multiview_fusion_model(
        ...     vision_backbone_name="resnet18",
        ...     tabular_input_dim=10,
        ...     fusion_type="gated",
        ...     num_classes=2,
        ...     aggregator_type="attention",
        ...     view_names=["axial", "coronal", "sagittal"],
        ... )
    """
    from med_core.backbones import (
        create_multiview_vision_backbone,
        create_tabular_backbone,
    )
    from med_core.fusion.strategies import create_fusion_module

    # Create multi-view vision backbone
    vision_backbone = create_multiview_vision_backbone(
        backbone_name=vision_backbone_name,
        aggregator_type=aggregator_type,
        share_weights=share_weights,
        view_names=view_names,
        feature_dim=vision_feature_dim,
        **kwargs.get("vision_kwargs", {}),
    )

    # Create tabular backbone
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_input_dim,
        output_dim=tabular_feature_dim,
        **kwargs.get("tabular_kwargs", {}),
    )

    # Create fusion module
    fusion_module = create_fusion_module(
        fusion_type=fusion_type,
        vision_dim=vision_feature_dim,
        tabular_dim=tabular_feature_dim,
        output_dim=fusion_output_dim,
        **kwargs.get("fusion_kwargs", {}),
    )

    # Create complete model
    model = MultiViewMultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=num_classes,
        dropout=kwargs.get("dropout", 0.4),
        use_auxiliary_heads=kwargs.get("use_auxiliary_heads", True),
    )

    return model
