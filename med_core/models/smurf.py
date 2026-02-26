"""
SMuRF: Survival Multimodal Radiology-Pathology Fusion Model

This module implements the SMuRF architecture using the generic multi-modal
model builder. SMuRF combines 3D radiology (CT) and 2D pathology features
for classification and survival prediction tasks.

Note: This is a refactored implementation that uses the generic MultiModalModelBuilder
internally, eliminating code duplication and automatically benefiting from all
framework features (gradient checkpointing, attention supervision, etc.).
"""

from typing import Any, Literal

import torch
import torch.nn as nn

from med_core.models.builder import MultiModalModelBuilder


class SMuRFModel(nn.Module):
    """
    SMuRF model for multimodal medical imaging.

    Combines 3D radiology imaging (CT scans) with 2D pathology imaging
    (histopathology slides) using advanced fusion strategies.

    This is a convenience wrapper around GenericMultiModalModel that provides
    a domain-specific API for radiology-pathology fusion.

    Args:
        radiology_backbone: Configuration for 3D radiology backbone
        pathology_backbone: Configuration for 2D pathology backbone
        fusion_strategy: Fusion method ('concat', 'kronecker', 'fused_attention')
        num_classes: Number of classification classes
        radiology_feature_dim: Output dimension for radiology features
        pathology_feature_dim: Output dimension for pathology features
        fusion_hidden_dim: Hidden dimension for fusion module
        dropout: Dropout rate

    Example:
        >>> model = SMuRFModel(
        ...     radiology_backbone={'variant': 'small'},
        ...     pathology_backbone={'variant': 'small'},
        ...     fusion_strategy='fused_attention',
        ...     num_classes=4
        ... )
        >>> ct = torch.randn(2, 1, 64, 128, 128)
        >>> pathology = torch.randn(2, 3, 224, 224)
        >>> logits = model(ct, pathology)
    """

    def __init__(
        self,
        radiology_backbone: dict,
        pathology_backbone: dict,
        fusion_strategy: Literal[
            "concat", "kronecker", "fused_attention"
        ] = "fused_attention",
        num_classes: int = 4,
        radiology_feature_dim: int = 512,
        pathology_feature_dim: int = 512,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes

        # Build model using generic builder
        radiology_variant = radiology_backbone.get("variant", "small")
        radiology_in_channels = radiology_backbone.get("in_channels", 1)
        pathology_variant = pathology_backbone.get("variant", "small")
        pathology_in_channels = pathology_backbone.get("in_channels", 3)

        # Map fusion strategy names
        fusion_map = {
            "concat": "concat",
            "kronecker": "kronecker",
            "fused_attention": "fused_attention",
        }

        # Validate fusion strategy
        if fusion_strategy not in fusion_map:
            raise ValueError(
                f"Invalid fusion strategy: {fusion_strategy}. "
                f"Available: {list(fusion_map.keys())}"
            )

        builder = MultiModalModelBuilder()
        builder.add_modality(
            "radiology",
            backbone=f"swin3d_{radiology_variant}",
            modality_type="vision3d",
            feature_dim=radiology_feature_dim,
            in_channels=radiology_in_channels,
        )
        builder.add_modality(
            "pathology",
            backbone=f"swin2d_{pathology_variant}",
            modality_type="vision",
            feature_dim=pathology_feature_dim,
            in_channels=pathology_in_channels,
        )

        # Set fusion
        fusion_kwargs = {}
        if fusion_strategy != "concat":
            fusion_kwargs["output_dim"] = fusion_hidden_dim
        if fusion_strategy == "fused_attention":
            fusion_kwargs["num_heads"] = 8
            fusion_kwargs["use_kronecker"] = True

        builder.set_fusion(fusion_map[fusion_strategy], **fusion_kwargs)

        # Set head
        builder.set_head("classification", num_classes=num_classes, dropout=dropout)

        # Build the model
        self._model = builder.build()

    def forward(
        self,
        radiology: torch.Tensor,
        pathology: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            radiology: 3D CT scan [B, C, D, H, W]
            pathology: 2D histopathology [B, C, H, W]
            return_features: Return intermediate features

        Returns:
            Classification logits [B, num_classes]
            If return_features=True, also returns feature dict
        """
        inputs = {
            "radiology": radiology,
            "pathology": pathology,
        }

        if return_features:
            logits, features = self._model(inputs, return_features=True)
            # Reformat features to match original API
            reformatted_features = {
                "radiology": features["modality_features"]["radiology"],
                "pathology": features["modality_features"]["pathology"],
                "fused": features["fused_features"],
            }
            return logits, reformatted_features
        else:
            return self._model(inputs, return_features=False)


class SMuRFWithMIL(nn.Module):
    """
    SMuRF model with Multiple Instance Learning (MIL) for pathology.

    Handles multiple pathology patches per sample using attention-based
    aggregation before fusion with radiology features.

    Args:
        radiology_backbone: Configuration for 3D radiology backbone
        pathology_backbone: Configuration for 2D pathology backbone
        fusion_strategy: Fusion method
        num_classes: Number of classification classes
        radiology_feature_dim: Output dimension for radiology features
        pathology_feature_dim: Output dimension for pathology features
        fusion_hidden_dim: Hidden dimension for fusion module
        mil_attention_dim: Attention dimension for MIL aggregation
        dropout: Dropout rate

    Example:
        >>> model = SMuRFWithMIL(
        ...     radiology_backbone={'variant': 'small'},
        ...     pathology_backbone={'variant': 'small'},
        ...     num_classes=4
        ... )
        >>> ct = torch.randn(2, 1, 64, 128, 128)
        >>> pathology_patches = torch.randn(2, 10, 3, 224, 224)  # 10 patches
        >>> logits = model(ct, pathology_patches)
    """

    def __init__(
        self,
        radiology_backbone: dict,
        pathology_backbone: dict,
        fusion_strategy: Literal[
            "concat", "kronecker", "fused_attention"
        ] = "fused_attention",
        num_classes: int = 4,
        radiology_feature_dim: int = 512,
        pathology_feature_dim: int = 512,
        fusion_hidden_dim: int = 256,
        mil_attention_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes

        # Build model using generic builder with MIL
        radiology_variant = radiology_backbone.get("variant", "small")
        radiology_in_channels = radiology_backbone.get("in_channels", 1)
        pathology_variant = pathology_backbone.get("variant", "small")
        pathology_in_channels = pathology_backbone.get("in_channels", 3)

        # Map fusion strategy names
        fusion_map = {
            "concat": "concat",
            "kronecker": "kronecker",
            "fused_attention": "fused_attention",
        }

        builder = MultiModalModelBuilder()
        builder.add_modality(
            "radiology",
            backbone=f"swin3d_{radiology_variant}",
            modality_type="vision3d",
            feature_dim=radiology_feature_dim,
            in_channels=radiology_in_channels,
        )
        builder.add_modality(
            "pathology",
            backbone=f"swin2d_{pathology_variant}",
            modality_type="vision",
            feature_dim=pathology_feature_dim,
            in_channels=pathology_in_channels,
        )

        # Add MIL aggregation for pathology
        builder.add_mil_aggregation(
            "pathology",
            strategy="attention",
            attention_dim=mil_attention_dim,
        )

        # Set fusion
        fusion_kwargs = {}
        if fusion_strategy != "concat":
            fusion_kwargs["output_dim"] = fusion_hidden_dim
        if fusion_strategy == "fused_attention":
            fusion_kwargs["num_heads"] = 8
            fusion_kwargs["use_kronecker"] = True

        builder.set_fusion(fusion_map[fusion_strategy], **fusion_kwargs)

        # Set head
        builder.set_head("classification", num_classes=num_classes, dropout=dropout)

        # Build the model
        self._model = builder.build()

    def forward(
        self,
        radiology: torch.Tensor,
        pathology_patches: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass with MIL aggregation.

        Args:
            radiology: 3D CT scan [B, C, D, H, W]
            pathology_patches: Multiple pathology patches [B, N, C, H, W]
            return_features: Return intermediate features

        Returns:
            Classification logits [B, num_classes]
            If return_features=True, also returns feature dict
        """
        inputs = {
            "radiology": radiology,
            "pathology": pathology_patches,
        }

        if return_features:
            logits, features = self._model(inputs, return_features=True)
            # Reformat features to match original API
            reformatted_features = {
                "radiology": features["modality_features"]["radiology"],
                "pathology_patches": features["modality_features"]["pathology_patches"],
                "pathology_aggregated": features["modality_features"]["pathology"],
                "fused": features["fused_features"],
            }
            if "mil_attention_weights" in features:
                reformatted_features["attention_weights"] = features[
                    "mil_attention_weights"
                ].get("pathology")
            return logits, reformatted_features
        else:
            return self._model(inputs, return_features=False)


def smurf_small(
    num_classes: int = 4,
    fusion_strategy: Literal[
        "concat", "kronecker", "fused_attention"
    ] = "fused_attention",
    **kwargs: Any,
) -> SMuRFModel:
    """
    Create a small SMuRF model.

    Args:
        num_classes: Number of classification classes
        fusion_strategy: Fusion method
        **kwargs: Additional arguments

    Returns:
        SMuRF model with small backbones

    Example:
        >>> model = smurf_small(num_classes=4, fusion_strategy='fused_attention')
        >>> ct = torch.randn(2, 1, 64, 128, 128)
        >>> pathology = torch.randn(2, 3, 224, 224)
        >>> logits = model(ct, pathology)
    """
    return SMuRFModel(
        radiology_backbone={"variant": "small"},
        pathology_backbone={"variant": "small"},
        fusion_strategy=fusion_strategy,
        num_classes=num_classes,
        **kwargs,
    )


def smurf_base(
    num_classes: int = 4,
    fusion_strategy: Literal[
        "concat", "kronecker", "fused_attention"
    ] = "fused_attention",
    **kwargs: Any,
) -> SMuRFModel:
    """
    Create a base SMuRF model.

    Args:
        num_classes: Number of classification classes
        fusion_strategy: Fusion method
        **kwargs: Additional arguments

    Returns:
        SMuRF model with base backbones

    Example:
        >>> model = smurf_base(num_classes=4, fusion_strategy='fused_attention')
    """
    return SMuRFModel(
        radiology_backbone={"variant": "base"},
        pathology_backbone={"variant": "base"},
        fusion_strategy=fusion_strategy,
        num_classes=num_classes,
        radiology_feature_dim=768,
        pathology_feature_dim=768,
        fusion_hidden_dim=512,
        **kwargs,
    )


def smurf_with_mil_small(
    num_classes: int = 4,
    fusion_strategy: Literal[
        "concat", "kronecker", "fused_attention"
    ] = "fused_attention",
    **kwargs: Any,
) -> SMuRFWithMIL:
    """
    Create a small SMuRF model with MIL.

    Args:
        num_classes: Number of classification classes
        fusion_strategy: Fusion method
        **kwargs: Additional arguments

    Returns:
        SMuRF model with MIL and small backbones

    Example:
        >>> model = smurf_with_mil_small(num_classes=4)
        >>> ct = torch.randn(2, 1, 64, 128, 128)
        >>> pathology_patches = torch.randn(2, 10, 3, 224, 224)
        >>> logits = model(ct, pathology_patches)
    """
    return SMuRFWithMIL(
        radiology_backbone={"variant": "small"},
        pathology_backbone={"variant": "small"},
        fusion_strategy=fusion_strategy,
        num_classes=num_classes,
        **kwargs,
    )
