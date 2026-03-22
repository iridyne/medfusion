"""Tests for GenericMultiModalModel and MultiModalModelBuilder."""

import torch
from torch import nn, optim

from med_core.models import MultiModalModelBuilder


class DummyBackbone(nn.Module):
    """Small backbone used to test builder behavior without heavy vision models."""

    def __init__(self, input_dim: int = 4, output_dim: int = 8) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_three_modality_builder_registers_attention_before_forward() -> None:
    """The >2-modality pooling module should exist before the optimizer is built."""
    model = (
        MultiModalModelBuilder()
        .add_modality("mod1", backbone=DummyBackbone())
        .add_modality("mod2", backbone=DummyBackbone())
        .add_modality("mod3", backbone=DummyBackbone())
        .set_fusion("concat")
        .set_head("classification", num_classes=2)
        .build()
    )

    assert model.multimodal_attention is not None

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    attention_params = list(model.multimodal_attention.parameters())

    assert attention_params
    assert all(id(param) in optimizer_param_ids for param in attention_params)

    logits, features = model(
        {
            "mod1": torch.randn(2, 4),
            "mod2": torch.randn(2, 4),
            "mod3": torch.randn(2, 4),
        },
        return_features=True,
    )

    assert logits.shape == (2, 2)
    assert features["fused_features"].shape == (2, 8)
    assert "fusion_aux" in features
    assert features["fusion_aux"]["multimodal_attention_weights"].shape == (2, 3)

    loss = logits.sum()
    loss.backward()

    assert all(param.grad is not None for param in attention_params)
