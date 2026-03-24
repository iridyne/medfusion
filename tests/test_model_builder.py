"""Tests for GenericMultiModalModel and MultiModalModelBuilder."""

import pytest
import torch
from torch import nn

from med_core.backbones import HIPTEmbeddingBackbone
from med_core.fusion.fused_attention import (
    FusedAttentionFusion,
    MultimodalFusedAttention,
)
from med_core.fusion.kronecker import CompactKroneckerFusion, KroneckerFusion
from med_core.fusion.multimodal import (
    MultimodalConcatenateFusion,
    MultimodalGatedFusion,
)
from med_core.fusion.self_attention import MultimodalSelfAttentionFusion
from med_core.models import MultiModalModelBuilder


class DummyBackbone(nn.Module):
    """Small backbone used to test builder behavior without heavy vision models."""

    def __init__(self, input_dim: int = 4, output_dim: int = 8) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_three_modality_concat_builder_uses_real_multimodal_fusion() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("mod1", backbone=DummyBackbone())
        .add_modality("mod2", backbone=DummyBackbone())
        .add_modality("mod3", backbone=DummyBackbone())
        .set_fusion("concat")
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.fusion_module, MultimodalConcatenateFusion)
    assert model.multimodal_attention is None

    logits, features = model(
        {
            "mod1": torch.randn(2, 4),
            "mod2": torch.randn(2, 4),
            "mod3": torch.randn(2, 4),
        },
        return_features=True,
    )

    assert logits.shape == (2, 2)
    assert features["fused_features"].shape == (2, 24)

    loss = logits.sum()
    loss.backward()

    concat_params = list(model.fusion_module.parameters())
    assert concat_params
    assert all(param.grad is not None for param in concat_params)


def test_three_modality_gated_builder_uses_real_multimodal_fusion() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("mod1", backbone=DummyBackbone())
        .add_modality("mod2", backbone=DummyBackbone())
        .add_modality("mod3", backbone=DummyBackbone())
        .set_fusion("gated", output_dim=10)
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.fusion_module, MultimodalGatedFusion)

    logits, features = model(
        {
            "mod1": torch.randn(2, 4),
            "mod2": torch.randn(2, 4),
            "mod3": torch.randn(2, 4),
        },
        return_features=True,
    )

    assert logits.shape == (2, 2)
    assert features["fused_features"].shape == (2, 10)
    assert "fusion_aux" in features
    assert features["fusion_aux"]["gate_values"].shape == (2, 3)


def test_three_modality_attention_builder_uses_real_multimodal_fusion() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("mod1", backbone=DummyBackbone())
        .add_modality("mod2", backbone=DummyBackbone())
        .add_modality("mod3", backbone=DummyBackbone())
        .set_fusion("attention", output_dim=12, num_heads=4)
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.fusion_module, MultimodalSelfAttentionFusion)

    logits, features = model(
        {
            "mod1": torch.randn(2, 4),
            "mod2": torch.randn(2, 4),
            "mod3": torch.randn(2, 4),
        },
        return_features=True,
    )

    assert logits.shape == (2, 2)
    assert features["fused_features"].shape == (2, 12)
    assert "fusion_aux" in features
    assert features["fusion_aux"]["attention_weights"].shape[0] == 2


def test_three_modality_fused_attention_builder_uses_real_multimodal_fusion() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("mod1", backbone=DummyBackbone())
        .add_modality("mod2", backbone=DummyBackbone())
        .add_modality("mod3", backbone=DummyBackbone())
        .set_fusion("fused_attention", output_dim=12, num_heads=4)
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.fusion_module, MultimodalFusedAttention)

    logits = model(
        {
            "mod1": torch.randn(2, 4),
            "mod2": torch.randn(2, 4),
            "mod3": torch.randn(2, 4),
        }
    )

    assert logits.shape == (2, 2)


def test_three_modality_cross_attention_is_rejected() -> None:
    with pytest.raises(ValueError, match=">2 modalities"):
        (
            MultiModalModelBuilder()
            .add_modality("mod1", backbone=DummyBackbone())
            .add_modality("mod2", backbone=DummyBackbone())
            .add_modality("mod3", backbone=DummyBackbone())
            .set_fusion("cross_attention")
            .set_head("classification", num_classes=2)
            .build()
        )


def test_builder_uses_real_kronecker_fusion_for_small_models() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("image", backbone=DummyBackbone(output_dim=8))
        .add_modality("tabular", backbone=DummyBackbone(output_dim=8))
        .set_fusion("kronecker", output_dim=6)
        .set_head("classification", num_classes=3)
        .build()
    )

    assert isinstance(model.fusion_module, KroneckerFusion)

    logits = model(
        {
            "image": torch.randn(2, 4),
            "tabular": torch.randn(2, 4),
        }
    )

    assert logits.shape == (2, 3)


def test_builder_uses_compact_kronecker_for_large_projections() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("image", backbone=DummyBackbone(output_dim=512))
        .add_modality("tabular", backbone=DummyBackbone(output_dim=512))
        .set_fusion("kronecker", output_dim=64)
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.fusion_module, CompactKroneckerFusion)

    logits = model(
        {
            "image": torch.randn(2, 4),
            "tabular": torch.randn(2, 4),
        }
    )

    assert logits.shape == (2, 2)


def test_builder_uses_real_fused_attention_module() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("image", backbone=DummyBackbone(output_dim=16))
        .add_modality("tabular", backbone=DummyBackbone(output_dim=16))
        .set_fusion("fused_attention", output_dim=12, num_heads=4, use_kronecker=False)
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.fusion_module, FusedAttentionFusion)

    logits = model(
        {
            "image": torch.randn(2, 4),
            "tabular": torch.randn(2, 4),
        }
    )

    assert logits.shape == (2, 2)


def test_builder_supports_embedding_modality_with_mil_aggregation() -> None:
    model = (
        MultiModalModelBuilder()
        .add_modality("image", backbone=DummyBackbone(output_dim=8))
        .add_modality(
            "pathology",
            backbone="hipt",
            modality_type="embedding",
            feature_dim=8,
            embedding_dim=16,
        )
        .add_mil_aggregation("pathology", strategy="attention", attention_dim=4)
        .set_fusion("concat")
        .set_head("classification", num_classes=2)
        .build()
    )

    assert isinstance(model.modality_backbones["pathology"], HIPTEmbeddingBackbone)

    logits, features = model(
        {
            "image": torch.randn(2, 4),
            "pathology": torch.randn(2, 5, 16),
        },
        return_features=True,
    )

    assert logits.shape == (2, 2)
    assert features["modality_features"]["pathology_patches"].shape == (2, 5, 8)
    assert features["modality_features"]["pathology"].shape == (2, 8)
