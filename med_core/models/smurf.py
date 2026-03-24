"""
SMuRF: Survival Multimodal Radiology-Pathology Fusion Model.

This module implements the SMuRF architecture using the generic multi-modal
model builder. SMuRF combines 3D radiology (CT) and pathology features for
classification and survival-related tasks.

Pathology encoder options:
- patch_mil (default): 2D patch images with a vision backbone
- hipt: offline/precomputed embedding vectors (HIPT-style)
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from med_core.models.builder import MultiModalModelBuilder

PathologyEncoderType = Literal["patch_mil", "hipt"]


def _normalize_pathology_encoder(pathology_encoder: str) -> PathologyEncoderType:
    normalized = str(pathology_encoder).strip().lower()
    if normalized not in {"patch_mil", "hipt"}:
        raise ValueError(
            f"Invalid pathology_encoder: {pathology_encoder}. "
            "Available: ['patch_mil', 'hipt']",
        )
    return normalized  # type: ignore[return-value]


def _add_pathology_modality(
    builder: MultiModalModelBuilder,
    pathology_backbone: dict[str, Any],
    pathology_encoder: PathologyEncoderType,
    pathology_feature_dim: int,
) -> None:
    """Attach pathology modality according to encoder type."""
    if pathology_encoder == "patch_mil":
        pathology_variant = pathology_backbone.get("variant", "small")
        pathology_in_channels = pathology_backbone.get("in_channels", 3)
        builder.add_modality(
            "pathology",
            backbone=f"swin2d_{pathology_variant}",
            modality_type="vision",
            feature_dim=pathology_feature_dim,
            in_channels=pathology_in_channels,
        )
        return

    embedding_dim = int(
        pathology_backbone.get(
            "embedding_dim",
            pathology_backbone.get("input_dim", 192),
        )
    )
    embedding_dropout = float(pathology_backbone.get("dropout", 0.1))

    builder.add_modality(
        "pathology",
        backbone="hipt",
        modality_type="embedding",
        feature_dim=pathology_feature_dim,
        input_dim=embedding_dim,
        dropout=embedding_dropout,
    )


class SMuRFModel(nn.Module):
    """SMuRF model for radiology-pathology fusion."""

    def __init__(
        self,
        radiology_backbone: dict,
        pathology_backbone: dict,
        fusion_strategy: Literal[
            "concat", "kronecker", "fused_attention",
        ] = "fused_attention",
        num_classes: int = 4,
        radiology_feature_dim: int = 512,
        pathology_feature_dim: int = 512,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        pathology_encoder: PathologyEncoderType = "patch_mil",
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.pathology_encoder = _normalize_pathology_encoder(pathology_encoder)

        # Build model using generic builder
        radiology_variant = radiology_backbone.get("variant", "small")
        radiology_in_channels = radiology_backbone.get("in_channels", 1)

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
                f"Available: {list(fusion_map.keys())}",
            )

        builder = MultiModalModelBuilder()
        builder.add_modality(
            "radiology",
            backbone=f"swin3d_{radiology_variant}",
            modality_type="vision3d",
            feature_dim=radiology_feature_dim,
            in_channels=radiology_in_channels,
        )
        _add_pathology_modality(
            builder=builder,
            pathology_backbone=pathology_backbone,
            pathology_encoder=self.pathology_encoder,
            pathology_feature_dim=pathology_feature_dim,
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
        """Forward pass.

        Args:
            radiology: 3D CT scan [B, C, D, H, W]
            pathology:
                - patch_mil: [B, C, H, W]
                - hipt: [B, D]
            return_features: Return intermediate features
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
        return self._model(inputs, return_features=False)


class SMuRFWithMIL(nn.Module):
    """SMuRF model with MIL for pathology."""

    def __init__(
        self,
        radiology_backbone: dict,
        pathology_backbone: dict,
        fusion_strategy: Literal[
            "concat", "kronecker", "fused_attention",
        ] = "fused_attention",
        num_classes: int = 4,
        radiology_feature_dim: int = 512,
        pathology_feature_dim: int = 512,
        fusion_hidden_dim: int = 256,
        mil_attention_dim: int = 128,
        dropout: float = 0.3,
        pathology_encoder: PathologyEncoderType = "patch_mil",
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.pathology_encoder = _normalize_pathology_encoder(pathology_encoder)

        # Build model using generic builder with MIL
        radiology_variant = radiology_backbone.get("variant", "small")
        radiology_in_channels = radiology_backbone.get("in_channels", 1)

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
        _add_pathology_modality(
            builder=builder,
            pathology_backbone=pathology_backbone,
            pathology_encoder=self.pathology_encoder,
            pathology_feature_dim=pathology_feature_dim,
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
        """Forward pass with MIL aggregation.

        Args:
            radiology: 3D CT scan [B, C, D, H, W]
            pathology_patches:
                - patch_mil: [B, N, C, H, W]
                - hipt: [B, N, D]
            return_features: Return intermediate features
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
        return self._model(inputs, return_features=False)


def smurf_small(
    num_classes: int = 4,
    fusion_strategy: Literal[
        "concat", "kronecker", "fused_attention",
    ] = "fused_attention",
    **kwargs: Any,
) -> SMuRFModel:
    """Create a small SMuRF model."""
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
        "concat", "kronecker", "fused_attention",
    ] = "fused_attention",
    **kwargs: Any,
) -> SMuRFModel:
    """Create a base SMuRF model."""
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
        "concat", "kronecker", "fused_attention",
    ] = "fused_attention",
    **kwargs: Any,
) -> SMuRFWithMIL:
    """Create a small SMuRF model with MIL."""
    return SMuRFWithMIL(
        radiology_backbone={"variant": "small"},
        pathology_backbone={"variant": "small"},
        fusion_strategy=fusion_strategy,
        num_classes=num_classes,
        **kwargs,
    )
