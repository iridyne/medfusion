"""
Unit tests for Multi-Region Extractors.
"""

import pytest
import torch
import torch.nn as nn

from med_core.extractors.multi_region import (
    AdaptiveRegionExtractor,
    HierarchicalRegionExtractor,
    MultiRegionExtractor,
    MultiScaleRegionExtractor,
)


class DummyBackbone(nn.Module):
    """Dummy backbone for testing."""

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone_output_dim = feature_dim
        self.conv = nn.Conv3d(1, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestMultiRegionExtractor:
    """Test suite for MultiRegionExtractor."""

    def test_basic_extraction(self):
        """Test basic multi-region extraction."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiRegionExtractor(backbone, num_regions=3)

        image = torch.randn(2, 1, 32, 64, 64)
        features = extractor(image)

        assert features.shape == (2, 3, 512)

    def test_with_masks(self):
        """Test extraction with region masks."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiRegionExtractor(backbone, num_regions=3)

        image = torch.randn(2, 1, 32, 64, 64)
        masks = torch.randn(2, 3, 32, 64, 64)

        features = extractor(image, masks)

        assert features.shape == (2, 3, 512)

    def test_different_num_regions(self):
        """Test with different number of regions."""
        backbone = DummyBackbone(feature_dim=512)

        for num_regions in [1, 3, 5]:
            extractor = MultiRegionExtractor(backbone, num_regions=num_regions)
            image = torch.randn(2, 1, 32, 64, 64)

            features = extractor(image)

            assert features.shape == (2, num_regions, 512)

    def test_gradient_flow(self):
        """Test gradient flow."""
        backbone = DummyBackbone(feature_dim=128)
        extractor = MultiRegionExtractor(backbone, num_regions=3)

        image = torch.randn(2, 1, 32, 64, 64, requires_grad=True)
        features = extractor(image)

        loss = features.sum()
        loss.backward()

        assert image.grad is not None

    def test_2d_input(self):
        """Test with 2D input."""

        class DummyBackbone2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 512)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        backbone = DummyBackbone2D()
        extractor = MultiRegionExtractor(backbone, num_regions=3)

        image = torch.randn(2, 1, 64, 64)
        features = extractor(image)

        assert features.shape == (2, 3, 512)


class TestHierarchicalRegionExtractor:
    """Test suite for HierarchicalRegionExtractor."""

    def test_basic_extraction(self):
        """Test basic hierarchical extraction."""
        backbone = DummyBackbone(feature_dim=512)
        region_names = ["tumor", "peritumoral", "background"]
        extractor = HierarchicalRegionExtractor(
            backbone, region_names=region_names, aggregation="concat"
        )

        image = torch.randn(2, 1, 32, 64, 64)
        masks = {
            "tumor": torch.randn(2, 1, 32, 64, 64),
            "peritumoral": torch.randn(2, 1, 32, 64, 64),
            "background": torch.randn(2, 1, 32, 64, 64),
        }

        features = extractor(image, masks)

        assert features.shape == (2, 512 * 3)  # Concatenated

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        backbone = DummyBackbone(feature_dim=512)
        region_names = ["tumor", "peritumoral"]
        extractor = HierarchicalRegionExtractor(
            backbone, region_names=region_names, aggregation="mean"
        )

        image = torch.randn(2, 1, 32, 64, 64)
        masks = {
            "tumor": torch.randn(2, 1, 32, 64, 64),
            "peritumoral": torch.randn(2, 1, 32, 64, 64),
        }

        features = extractor(image, masks)

        assert features.shape == (2, 512)

    def test_attention_aggregation(self):
        """Test attention aggregation."""
        backbone = DummyBackbone(feature_dim=512)
        region_names = ["tumor", "peritumoral", "background"]
        extractor = HierarchicalRegionExtractor(
            backbone, region_names=region_names, aggregation="attention"
        )

        image = torch.randn(2, 1, 32, 64, 64)
        masks = {
            "tumor": torch.randn(2, 1, 32, 64, 64),
            "peritumoral": torch.randn(2, 1, 32, 64, 64),
            "background": torch.randn(2, 1, 32, 64, 64),
        }

        features = extractor(image, masks)

        assert features.shape == (2, 512)

    def test_missing_mask_error(self):
        """Test error handling for missing masks."""
        backbone = DummyBackbone(feature_dim=512)
        region_names = ["tumor", "peritumoral"]
        extractor = HierarchicalRegionExtractor(backbone, region_names=region_names)

        image = torch.randn(2, 1, 32, 64, 64)
        masks = {"tumor": torch.randn(2, 1, 32, 64, 64)}  # Missing peritumoral

        with pytest.raises(ValueError):
            extractor(image, masks)

    def test_gradient_flow(self):
        """Test gradient flow."""
        backbone = DummyBackbone(feature_dim=128)
        region_names = ["tumor", "background"]
        extractor = HierarchicalRegionExtractor(
            backbone, region_names=region_names, aggregation="attention"
        )

        image = torch.randn(2, 1, 32, 64, 64, requires_grad=True)
        masks = {
            "tumor": torch.randn(2, 1, 32, 64, 64),
            "background": torch.randn(2, 1, 32, 64, 64),
        }

        features = extractor(image, masks)
        loss = features.sum()
        loss.backward()

        assert image.grad is not None


class TestAdaptiveRegionExtractor:
    """Test suite for AdaptiveRegionExtractor."""

    def test_basic_extraction(self):
        """Test basic adaptive extraction."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = AdaptiveRegionExtractor(backbone, num_regions=5, feature_dim=512)

        image = torch.randn(2, 1, 32, 64, 64)
        features = extractor(image)

        assert features.shape == (2, 5, 512)

    def test_return_coordinates(self):
        """Test returning region coordinates."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = AdaptiveRegionExtractor(backbone, num_regions=5, feature_dim=512)

        image = torch.randn(2, 1, 32, 64, 64)
        features, coords = extractor(image, return_coords=True)

        assert features.shape == (2, 5, 512)
        assert coords.shape == (2, 5, 3)
        # Coordinates should be in [0, 1]
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_different_num_regions(self):
        """Test with different number of regions."""
        backbone = DummyBackbone(feature_dim=512)

        for num_regions in [1, 3, 10]:
            extractor = AdaptiveRegionExtractor(
                backbone, num_regions=num_regions, feature_dim=512
            )
            image = torch.randn(2, 1, 32, 64, 64)

            features = extractor(image)

            assert features.shape == (2, num_regions, 512)

    def test_gradient_flow(self):
        """Test gradient flow."""
        backbone = DummyBackbone(feature_dim=128)
        extractor = AdaptiveRegionExtractor(backbone, num_regions=3, feature_dim=128)

        image = torch.randn(2, 1, 32, 64, 64, requires_grad=True)
        features = extractor(image)

        loss = features.sum()
        loss.backward()

        assert image.grad is not None


class TestMultiScaleRegionExtractor:
    """Test suite for MultiScaleRegionExtractor."""

    def test_basic_extraction(self):
        """Test basic multi-scale extraction."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiScaleRegionExtractor(
            backbone, scales=[1.0, 0.5], num_regions_per_scale=3
        )

        image = torch.randn(2, 1, 32, 64, 64)
        features = extractor(image)

        # 2 scales × 3 regions = 6 total regions
        assert features.shape == (2, 6, 512)

    def test_single_scale(self):
        """Test with single scale."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiScaleRegionExtractor(
            backbone, scales=[1.0], num_regions_per_scale=3
        )

        image = torch.randn(2, 1, 32, 64, 64)
        features = extractor(image)

        assert features.shape == (2, 3, 512)

    def test_multiple_scales(self):
        """Test with multiple scales."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiScaleRegionExtractor(
            backbone, scales=[1.0, 0.75, 0.5, 0.25], num_regions_per_scale=2
        )

        image = torch.randn(2, 1, 32, 64, 64)
        features = extractor(image)

        # 4 scales × 2 regions = 8 total regions
        assert features.shape == (2, 8, 512)

    def test_gradient_flow(self):
        """Test gradient flow."""
        backbone = DummyBackbone(feature_dim=128)
        extractor = MultiScaleRegionExtractor(
            backbone, scales=[1.0, 0.5], num_regions_per_scale=2
        )

        image = torch.randn(2, 1, 32, 64, 64, requires_grad=True)
        features = extractor(image)

        loss = features.sum()
        loss.backward()

        assert image.grad is not None


class TestIntegration:
    """Integration tests for multi-region extractors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        backbone = DummyBackbone(feature_dim=512).cuda()
        extractor = MultiRegionExtractor(backbone, num_regions=3).cuda()

        image = torch.randn(2, 1, 32, 64, 64).cuda()
        features = extractor(image)

        assert features.shape == (2, 3, 512)
        assert features.device.type == "cuda"

    def test_training_vs_eval_mode(self):
        """Test behavior in training vs eval mode."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiRegionExtractor(backbone, num_regions=3)

        image = torch.randn(2, 1, 32, 64, 64)

        # Training mode
        extractor.train()
        features_train = extractor(image)

        # Eval mode
        extractor.eval()
        with torch.no_grad():
            features_eval = extractor(image)

        assert features_train.shape == features_eval.shape

    def test_combine_with_aggregator(self):
        """Test combining region extractor with aggregator."""
        from med_core.aggregators import MILAggregator

        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiRegionExtractor(backbone, num_regions=5)
        aggregator = MILAggregator(input_dim=512, strategy="attention")

        image = torch.randn(2, 1, 32, 64, 64)

        # Extract region features
        region_features = extractor(image)  # [2, 5, 512]

        # Aggregate
        aggregated = aggregator(region_features)  # [2, 512]

        assert aggregated.shape == (2, 512)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        backbone = DummyBackbone(feature_dim=512)
        extractor = MultiRegionExtractor(backbone, num_regions=3)

        image = torch.randn(1, 1, 32, 64, 64)
        features = extractor(image)

        assert features.shape == (1, 3, 512)
