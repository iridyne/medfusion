"""
Base class for multimodal fusion modules.

Defines the interface that all fusion strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for multimodal fusion modules.

    All fusion modules must implement the forward method that takes
    features from multiple modalities and produces a fused representation.

    Attributes:
        output_dim: The dimension of the fused output features
    """

    def __init__(self, output_dim: int):
        """
        Initialize base fusion module.

        Args:
            output_dim: Dimension of the fused output features
        """
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    @abstractmethod
    def forward(
        self,
        vision_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """
        Fuse features from vision and tabular modalities.

        Args:
            vision_features: Features from vision backbone (B, vision_dim)
            tabular_features: Features from tabular backbone (B, tabular_dim)

        Returns:
            Tuple of:
                - Fused features tensor (B, output_dim)
                - Optional dict of auxiliary outputs (attention weights, gates, etc.)
        """

    def get_config(self) -> dict[str, Any]:
        """Return fusion configuration for serialization."""
        return {"output_dim": self._output_dim}

    @staticmethod
    def compute_modality_contribution(
        attention_weights: torch.Tensor | None = None,
        gate_values: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Compute the contribution of each modality to the fusion.

        Useful for interpretability and debugging.

        Args:
            attention_weights: Attention weights if using attention-based fusion
            gate_values: Gate values if using gated fusion

        Returns:
            Dictionary with modality contribution percentages
        """
        if gate_values is not None:
            # For gated fusion, use mean gate value
            mean_gate = gate_values.mean().item()
            return {
                "vision_contribution": mean_gate,
                "tabular_contribution": 1.0 - mean_gate,
            }

        if attention_weights is not None:
            # For attention fusion, analyze attention distribution
            # Assuming first half attends to vision, second half to tabular
            mid = attention_weights.size(-1) // 2
            vision_attn = attention_weights[..., :mid].mean().item()
            tabular_attn = attention_weights[..., mid:].mean().item()
            total = vision_attn + tabular_attn
            return {
                "vision_contribution": vision_attn / total if total > 0 else 0.5,
                "tabular_contribution": tabular_attn / total if total > 0 else 0.5,
            }

        # Default: equal contribution
        return {
            "vision_contribution": 0.5,
            "tabular_contribution": 0.5,
        }


class MultiModalFusionModel(nn.Module):
    """
    Complete multimodal fusion model combining vision, tabular backbones and fusion.

    This is a convenience class that wraps the full multimodal pipeline.
    """

    def __init__(
        self,
        vision_backbone: nn.Module,
        tabular_backbone: nn.Module,
        fusion_module: BaseFusion,
        num_classes: int = 2,
        dropout: float = 0.4,
        use_auxiliary_heads: bool = True,
        return_dict: bool = True,
    ):
        """
        Initialize multimodal fusion model.

        Args:
            vision_backbone: Vision feature extractor
            tabular_backbone: Tabular feature extractor
            fusion_module: Fusion strategy
            num_classes: Number of output classes
            dropout: Dropout rate for classifier
            use_auxiliary_heads: Whether to include per-modality classifiers
            return_dict: If True, return dict with all outputs. If False, return only logits tensor.
        """
        super().__init__()

        self.vision_backbone = vision_backbone
        self.tabular_backbone = tabular_backbone
        self.fusion_module = fusion_module
        self.use_auxiliary_heads = use_auxiliary_heads
        self.return_dict = return_dict

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
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass through the full multimodal model.

        Args:
            images: Image input tensor (B, C, H, W)
            tabular: Tabular input tensor (B, num_features)

        Returns:
            If return_dict=True (default):
                Dictionary containing:
                    - logits: Main fusion classifier output (B, num_classes)
                    - vision_features: Vision backbone features (B, vision_dim)
                    - tabular_features: Tabular backbone features (B, tabular_dim)
                    - fused_features: Fused features (B, fusion_dim)
                    - vision_logits: Vision-only classifier output (if auxiliary heads)
                    - tabular_logits: Tabular-only classifier output (if auxiliary heads)
                    - fusion_aux: Auxiliary fusion outputs (attention weights, etc.)
            If return_dict=False:
                Tensor of logits (B, num_classes)
        """
        # Extract features from each modality
        vision_features = self.vision_backbone(images)
        tabular_features = self.tabular_backbone(tabular)

        # Fuse features
        fused_features, fusion_aux = self.fusion_module(
            vision_features, tabular_features,
        )

        # Main classification
        logits = self.classifier(fused_features)

        # If return_dict=False, just return logits tensor
        if not self.return_dict:
            return logits

        # Build output dict
        outputs = {
            "logits": logits,
            "vision_features": vision_features,
            "tabular_features": tabular_features,
            "fused_features": fused_features,
        }

        if fusion_aux is not None:
            outputs["fusion_aux"] = fusion_aux

        # Auxiliary classifications
        if self.use_auxiliary_heads:
            outputs["vision_logits"] = self.vision_classifier(vision_features)
            outputs["tabular_logits"] = self.tabular_classifier(tabular_features)

        return outputs

    def get_attention_weights(self) -> dict[str, torch.Tensor] | None:
        """Get attention weights from fusion module if available."""
        if hasattr(self.fusion_module, "get_attention_weights"):
            return self.fusion_module.get_attention_weights()
        return None


def create_fusion_model(
    vision_backbone_name: str = "resnet18",
    tabular_input_dim: int = 10,
    num_classes: int = 2,
    fusion_type: str = "attention",
    dropout: float = 0.4,
    use_auxiliary_heads: bool = True,
    pretrained: bool = True,
    **kwargs: Any,
) -> MultiModalFusionModel:
    """
    Factory function to create a complete multimodal fusion model.

    This is a convenience function that creates vision backbone, tabular backbone,
    fusion module, and assembles them into a complete model.

    Args:
        vision_backbone_name: Name of vision backbone (e.g., "resnet18", "efficientnet_b0")
        tabular_input_dim: Number of tabular input features
        num_classes: Number of output classes
        fusion_type: Type of fusion strategy ("concatenate", "gated", "attention",
                     "cross_attention", "bilinear")
        dropout: Dropout rate for classifier
        use_auxiliary_heads: Whether to include per-modality classifiers
        pretrained: Whether to use pretrained vision backbone
        **kwargs: Additional arguments passed to backbone/fusion constructors

    Returns:
        Complete MultiModalFusionModel ready for training

    Example:
        >>> model = create_fusion_model(
        ...     vision_backbone_name="resnet18",
        ...     tabular_input_dim=10,
        ...     num_classes=2,
        ...     fusion_type="attention",
        ... )
    """
    from med_core.backbones import create_tabular_backbone, create_vision_backbone
    from med_core.fusion.strategies import create_fusion_module

    # Filter out 'config' from kwargs as it's not needed by sub-components
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "config"}

    # Create vision backbone
    vision_backbone = create_vision_backbone(
        backbone_name=vision_backbone_name,
        pretrained=pretrained,
        **filtered_kwargs,
    )

    # Create tabular backbone
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_input_dim,
        **filtered_kwargs,
    )

    # Create tabular backbone
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_input_dim,
        **kwargs,
    )

    # Create fusion module
    fusion_module = create_fusion_module(
        fusion_type=fusion_type,
        vision_dim=vision_backbone.output_dim,
        tabular_dim=tabular_backbone.output_dim,
        **filtered_kwargs,
    )

    # Assemble complete model
    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=num_classes,
        dropout=dropout,
        use_auxiliary_heads=use_auxiliary_heads,
    )

    return model
