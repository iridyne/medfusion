"""
Tests for new backbone architectures (Phase 1).

Tests ConvNeXt, MaxViT, EfficientNetV2, and RegNet backbones.
"""

import pytest
import torch

from med_core.backbones import (
    ConvNeXtBackbone,
    EfficientNetV2Backbone,
    MaxViTBackbone,
    RegNetBackbone,
    create_vision_backbone,
)


class TestConvNeXtBackbone:
    """Test ConvNeXt backbone variants."""

    @pytest.mark.parametrize(
        "variant,expected_backbone_dim",
        [
            ("convnext_tiny", 768),
            ("convnext_small", 768),
            ("convnext_base", 1024),
            ("convnext_large", 1536),
        ],
    )
    def test_convnext_variants(self, variant, expected_backbone_dim):
        """Test all ConvNeXt variants."""
        backbone = ConvNeXtBackbone(
            variant=variant,
            pretrained=False,
            feature_dim=128,
        )

        assert backbone.backbone_output_dim == expected_backbone_dim
        assert backbone.output_dim == 128

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()

    def test_convnext_with_attention(self):
        """Test ConvNeXt with optional attention."""
        backbone = ConvNeXtBackbone(
            variant="convnext_tiny",
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
        )

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)

    def test_convnext_freeze(self):
        """Test ConvNeXt freezing."""
        backbone = ConvNeXtBackbone(
            variant="convnext_tiny",
            pretrained=False,
            freeze=True,
        )

        # Check that backbone parameters are frozen
        for param in backbone._backbone.parameters():
            assert not param.requires_grad

        # Check that projection parameters are trainable
        for param in backbone._projection.parameters():
            assert param.requires_grad

    def test_convnext_factory(self):
        """Test ConvNeXt creation via factory function."""
        backbone = create_vision_backbone(
            "convnext_tiny",
            pretrained=False,
            feature_dim=256,
        )

        assert isinstance(backbone, ConvNeXtBackbone)
        assert backbone.output_dim == 256


class TestMaxViTBackbone:
    """Test MaxViT backbone."""

    def test_maxvit_basic(self):
        """Test basic MaxViT functionality."""
        backbone = MaxViTBackbone(
            variant="maxvit_t",
            pretrained=False,
            feature_dim=128,
        )

        assert backbone.backbone_output_dim == 512
        assert backbone.output_dim == 128

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()

    def test_maxvit_freeze(self):
        """Test MaxViT freezing."""
        backbone = MaxViTBackbone(
            variant="maxvit_t",
            pretrained=False,
            freeze=True,
        )

        # Check that backbone parameters are frozen
        for param in backbone._backbone.parameters():
            assert not param.requires_grad

        # Check that projection parameters are trainable
        for param in backbone._projection.parameters():
            assert param.requires_grad

    def test_maxvit_factory(self):
        """Test MaxViT creation via factory function."""
        backbone = create_vision_backbone(
            "maxvit_t",
            pretrained=False,
            feature_dim=256,
        )

        assert isinstance(backbone, MaxViTBackbone)
        assert backbone.output_dim == 256


class TestEfficientNetV2Backbone:
    """Test EfficientNetV2 backbone variants."""

    @pytest.mark.parametrize(
        "variant,expected_backbone_dim",
        [
            ("efficientnet_v2_s", 1280),
            ("efficientnet_v2_m", 1280),
            ("efficientnet_v2_l", 1280),
        ],
    )
    def test_efficientnetv2_variants(self, variant, expected_backbone_dim):
        """Test all EfficientNetV2 variants."""
        backbone = EfficientNetV2Backbone(
            variant=variant,
            pretrained=False,
            feature_dim=128,
        )

        assert backbone.backbone_output_dim == expected_backbone_dim
        assert backbone.output_dim == 128

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()

    def test_efficientnetv2_with_attention(self):
        """Test EfficientNetV2 with optional attention."""
        backbone = EfficientNetV2Backbone(
            variant="efficientnet_v2_s",
            pretrained=False,
            feature_dim=128,
            attention_type="se",
        )

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)

    def test_efficientnetv2_freeze(self):
        """Test EfficientNetV2 freezing."""
        backbone = EfficientNetV2Backbone(
            variant="efficientnet_v2_s",
            pretrained=False,
            freeze=True,
        )

        # Check that backbone parameters are frozen
        for param in backbone._backbone.parameters():
            assert not param.requires_grad

        # Check that projection parameters are trainable
        for param in backbone._projection.parameters():
            assert param.requires_grad

    def test_efficientnetv2_factory(self):
        """Test EfficientNetV2 creation via factory function."""
        backbone = create_vision_backbone(
            "efficientnet_v2_m",
            pretrained=False,
            feature_dim=256,
        )

        assert isinstance(backbone, EfficientNetV2Backbone)
        assert backbone.output_dim == 256


class TestRegNetBackbone:
    """Test RegNet backbone variants."""

    @pytest.mark.parametrize(
        "variant,expected_backbone_dim",
        [
            ("regnet_y_400mf", 440),
            ("regnet_y_800mf", 784),
            ("regnet_y_1_6gf", 888),
            ("regnet_y_3_2gf", 1512),
            ("regnet_y_8gf", 2016),
            ("regnet_y_16gf", 3024),
            ("regnet_y_32gf", 3712),
        ],
    )
    def test_regnet_variants(self, variant, expected_backbone_dim):
        """Test all RegNet variants."""
        backbone = RegNetBackbone(
            variant=variant,
            pretrained=False,
            feature_dim=128,
        )

        assert backbone.backbone_output_dim == expected_backbone_dim
        assert backbone.output_dim == 128

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()

    def test_regnet_with_attention(self):
        """Test RegNet with optional attention."""
        backbone = RegNetBackbone(
            variant="regnet_y_400mf",
            pretrained=False,
            feature_dim=128,
            attention_type="eca",
        )

        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 128)

    def test_regnet_freeze(self):
        """Test RegNet freezing."""
        backbone = RegNetBackbone(
            variant="regnet_y_400mf",
            pretrained=False,
            freeze=True,
        )

        # Check that backbone parameters are frozen
        for param in backbone._backbone.parameters():
            assert not param.requires_grad

        # Check that projection parameters are trainable
        for param in backbone._projection.parameters():
            assert param.requires_grad

    def test_regnet_factory(self):
        """Test RegNet creation via factory function."""
        backbone = create_vision_backbone(
            "regnet_y_1_6gf",
            pretrained=False,
            feature_dim=256,
        )

        assert isinstance(backbone, RegNetBackbone)
        assert backbone.output_dim == 256


class TestBackboneRegistry:
    """Test that all new backbones are properly registered."""

    def test_all_new_backbones_registered(self):
        """Test that all new backbone variants are in the registry."""
        from med_core.backbones.vision import BACKBONE_REGISTRY

        # ConvNeXt variants
        assert "convnext_tiny" in BACKBONE_REGISTRY
        assert "convnext_small" in BACKBONE_REGISTRY
        assert "convnext_base" in BACKBONE_REGISTRY
        assert "convnext_large" in BACKBONE_REGISTRY

        # MaxViT variants
        assert "maxvit_t" in BACKBONE_REGISTRY

        # EfficientNetV2 variants
        assert "efficientnet_v2_s" in BACKBONE_REGISTRY
        assert "efficientnet_v2_m" in BACKBONE_REGISTRY
        assert "efficientnet_v2_l" in BACKBONE_REGISTRY

        # RegNet variants
        assert "regnet_y_400mf" in BACKBONE_REGISTRY
        assert "regnet_y_800mf" in BACKBONE_REGISTRY
        assert "regnet_y_1_6gf" in BACKBONE_REGISTRY
        assert "regnet_y_3_2gf" in BACKBONE_REGISTRY
        assert "regnet_y_8gf" in BACKBONE_REGISTRY
        assert "regnet_y_16gf" in BACKBONE_REGISTRY
        assert "regnet_y_32gf" in BACKBONE_REGISTRY

    def test_total_backbone_count(self):
        """Test that we now have 26 total backbones."""
        from med_core.backbones.vision import BACKBONE_REGISTRY

        # Original: 14 backbones
        # New: 4 ConvNeXt + 1 MaxViT + 3 EfficientNetV2 + 7 RegNet = 15
        # Total: 14 + 15 = 29 (wait, let me recount)

        # Original backbones:
        # - ResNet: 4 (18, 34, 50, 101)
        # - MobileNet: 3 (v2, v3_small, v3_large)
        # - EfficientNet: 3 (b0, b1, b2)
        # - ViT: 2 (b_16, b_32)
        # - Swin: 2 (t, s)
        # Total original: 14

        # New backbones:
        # - ConvNeXt: 4 (tiny, small, base, large)
        # - MaxViT: 1 (t)
        # - EfficientNetV2: 3 (s, m, l)
        # - RegNet: 7 (400mf, 800mf, 1.6gf, 3.2gf, 8gf, 16gf, 32gf)
        # Total new: 15

        # Grand total: 29
        assert len(BACKBONE_REGISTRY) == 29

    @pytest.mark.parametrize(
        "backbone_name",
        [
            "convnext_tiny",
            "convnext_small",
            "maxvit_t",
            "efficientnet_v2_s",
            "regnet_y_400mf",
            "regnet_y_1_6gf",
        ],
    )
    def test_create_new_backbones_via_factory(self, backbone_name):
        """Test creating new backbones via factory function."""
        backbone = create_vision_backbone(
            backbone_name,
            pretrained=False,
            feature_dim=128,
        )

        assert backbone is not None
        assert backbone.output_dim == 128

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)
        assert output.shape == (2, 128)


class TestBackboneComparison:
    """Compare new backbones with existing ones."""

    def test_convnext_vs_resnet(self):
        """Compare ConvNeXt with ResNet."""
        convnext = create_vision_backbone(
            "convnext_tiny", pretrained=False, feature_dim=128
        )
        resnet = create_vision_backbone("resnet50", pretrained=False, feature_dim=128)

        x = torch.randn(2, 3, 224, 224)

        convnext_out = convnext(x)
        resnet_out = resnet(x)

        # Both should produce same output shape
        assert convnext_out.shape == resnet_out.shape == (2, 128)

    def test_efficientnetv2_vs_efficientnet(self):
        """Compare EfficientNetV2 with original EfficientNet."""
        v2 = create_vision_backbone(
            "efficientnet_v2_s", pretrained=False, feature_dim=128
        )
        v1 = create_vision_backbone(
            "efficientnet_b0", pretrained=False, feature_dim=128
        )

        x = torch.randn(2, 3, 224, 224)

        v2_out = v2(x)
        v1_out = v1(x)

        # Both should produce same output shape
        assert v2_out.shape == v1_out.shape == (2, 128)

    def test_regnet_scaling(self):
        """Test that RegNet variants scale properly."""
        small = create_vision_backbone(
            "regnet_y_400mf", pretrained=False, feature_dim=128
        )
        medium = create_vision_backbone(
            "regnet_y_1_6gf", pretrained=False, feature_dim=128
        )
        large = create_vision_backbone(
            "regnet_y_8gf", pretrained=False, feature_dim=128
        )

        # Count parameters
        small_params = sum(p.numel() for p in small.parameters())
        medium_params = sum(p.numel() for p in medium.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        # Larger variants should have more parameters
        assert small_params < medium_params < large_params


class TestMultiViewCompatibility:
    """Test that new backbones work with multi-view wrapper."""

    @pytest.mark.parametrize(
        "backbone_name",
        [
            "convnext_tiny",
            "maxvit_t",
            "efficientnet_v2_s",
            "regnet_y_400mf",
        ],
    )
    def test_new_backbones_with_multiview(self, backbone_name):
        """Test that new backbones work with MultiViewVisionBackbone."""
        from med_core.backbones import create_multiview_vision_backbone

        multiview_backbone = create_multiview_vision_backbone(
            backbone_name=backbone_name,
            view_names=["axial", "coronal", "sagittal"],
            aggregation_strategy="attention",
            pretrained=False,
            feature_dim=128,
        )

        # Test forward pass with multi-view input
        views = {
            "axial": torch.randn(2, 3, 224, 224),
            "coronal": torch.randn(2, 3, 224, 224),
            "sagittal": torch.randn(2, 3, 224, 224),
        }

        output = multiview_backbone(views)

        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
