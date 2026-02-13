"""
Tests for factory functions to ensure proper object creation.
"""

import pytest
import torch
import torch.nn as nn

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.fusion import create_fusion_module


class TestVisionBackboneFactory:
    """Test vision backbone factory function."""

    @pytest.mark.parametrize(
        "backbone_name",
        [
            "resnet18",
            "resnet34",
            "resnet50",
            "mobilenet_v2",
            "efficientnet_b0",
        ],
    )
    def test_create_vision_backbone_basic(self, backbone_name: str):
        """Test creating vision backbones with different architectures."""
        backbone = create_vision_backbone(
            backbone_name=backbone_name,
            pretrained=False,
            feature_dim=128,
        )

        assert backbone is not None
        assert isinstance(backbone, nn.Module)
        assert hasattr(backbone, "output_dim")
        assert backbone.output_dim == 128

    def test_create_vision_backbone_with_attention(self):
        """Test creating vision backbone with attention mechanism."""
        backbone = create_vision_backbone(
            backbone_name="resnet18",
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
        )

        assert backbone is not None
        assert isinstance(backbone, nn.Module)

    def test_create_vision_backbone_invalid_name(self):
        """Test that invalid backbone name raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_vision_backbone(
                backbone_name="invalid_backbone",
                pretrained=False,
            )

    def test_vision_backbone_forward_pass(self):
        """Test forward pass through vision backbone."""
        backbone = create_vision_backbone(
            backbone_name="resnet18",
            pretrained=False,
            feature_dim=128,
        )

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        output = backbone(x)

        assert output.shape == (batch_size, 128)


class TestTabularBackboneFactory:
    """Test tabular backbone factory function."""

    def test_create_tabular_backbone_basic(self):
        """Test creating basic tabular backbone."""
        backbone = create_tabular_backbone(
            input_dim=10,
            output_dim=32,
            hidden_dims=[64, 64],
        )

        assert backbone is not None
        assert isinstance(backbone, nn.Module)
        assert hasattr(backbone, "output_dim")
        assert backbone.output_dim == 32

    def test_create_tabular_backbone_with_dropout(self):
        """Test creating tabular backbone with dropout."""
        backbone = create_tabular_backbone(
            input_dim=10,
            output_dim=32,
            hidden_dims=[64, 64],
            dropout=0.3,
        )

        assert backbone is not None

    def test_tabular_backbone_forward_pass(self):
        """Test forward pass through tabular backbone."""
        backbone = create_tabular_backbone(
            input_dim=10,
            output_dim=32,
            hidden_dims=[64, 64],
        )

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 10)

        # Forward pass
        output = backbone(x)

        assert output.shape == (batch_size, 32)


class TestFusionModuleFactory:
    """Test fusion module factory function."""

    @pytest.mark.parametrize(
        "fusion_type",
        [
            "concatenate",
            "gated",
            "attention",
            "cross_attention",
            "bilinear",
        ],
    )
    def test_create_fusion_module_basic(self, fusion_type: str):
        """Test creating fusion modules with different strategies."""
        fusion = create_fusion_module(
            fusion_type=fusion_type,
            vision_dim=128,
            tabular_dim=32,
            output_dim=64,
        )

        assert fusion is not None
        assert isinstance(fusion, nn.Module)
        assert hasattr(fusion, "output_dim")

    def test_create_fusion_module_invalid_type(self):
        """Test that invalid fusion type raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_fusion_module(
                fusion_type="invalid_fusion",
                vision_dim=128,
                tabular_dim=32,
                output_dim=64,
            )

    def test_fusion_module_forward_pass(self):
        """Test forward pass through fusion module."""
        fusion = create_fusion_module(
            fusion_type="concatenate",
            vision_dim=128,
            tabular_dim=32,
            output_dim=64,
        )

        # Create dummy inputs
        batch_size = 4
        vision_features = torch.randn(batch_size, 128)
        tabular_features = torch.randn(batch_size, 32)

        # Forward pass
        output, aux = fusion(vision_features, tabular_features)

        assert output.shape == (batch_size, 64)

    def test_fusion_module_with_dropout(self):
        """Test creating fusion module with dropout."""
        fusion = create_fusion_module(
            fusion_type="gated",
            vision_dim=128,
            tabular_dim=32,
            output_dim=64,
            dropout=0.3,
        )

        assert fusion is not None


class TestFactoryIntegration:
    """Test integration of factory functions."""

    def test_end_to_end_model_creation(self):
        """Test creating a complete model using factory functions."""
        # Create components
        vision_backbone = create_vision_backbone(
            backbone_name="resnet18",
            pretrained=False,
            feature_dim=128,
        )

        tabular_backbone = create_tabular_backbone(
            input_dim=10,
            output_dim=32,
            hidden_dims=[64, 64],
        )

        fusion_module = create_fusion_module(
            fusion_type="gated",
            vision_dim=128,
            tabular_dim=32,
            output_dim=64,
        )

        # Verify all components created successfully
        assert vision_backbone is not None
        assert tabular_backbone is not None
        assert fusion_module is not None

    def test_factory_functions_with_different_configs(self):
        """Test factory functions with various configurations."""
        configs = [
            {
                "vision": "resnet18",
                "fusion": "concatenate",
            },
            {
                "vision": "resnet50",
                "fusion": "gated",
            },
            {
                "vision": "efficientnet_b0",
                "fusion": "attention",
            },
        ]

        for config in configs:
            vision = create_vision_backbone(
                backbone_name=config["vision"],
                pretrained=False,
                feature_dim=128,
            )

            fusion = create_fusion_module(
                fusion_type=config["fusion"],
                vision_dim=128,
                tabular_dim=32,
                output_dim=64,
            )

            assert vision is not None
            assert fusion is not None
