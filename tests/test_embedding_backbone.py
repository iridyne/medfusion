"""Tests for embedding-based pathology backbones."""

import torch

from med_core.backbones import HIPTEmbeddingBackbone


def test_hipt_embedding_backbone_supports_2d_inputs() -> None:
    backbone = HIPTEmbeddingBackbone(input_dim=16, feature_dim=8)

    output = backbone(torch.randn(3, 16))

    assert output.shape == (3, 8)


def test_hipt_embedding_backbone_supports_3d_inputs() -> None:
    backbone = HIPTEmbeddingBackbone(input_dim=16, feature_dim=8)

    output = backbone(torch.randn(3, 5, 16))

    assert output.shape == (3, 5, 8)
