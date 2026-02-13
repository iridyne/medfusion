"""
Unit tests for Kronecker Product Fusion.
"""

import pytest
import torch

from med_core.fusion.kronecker import (
    CompactKroneckerFusion,
    KroneckerFusion,
    MultimodalKroneckerFusion,
)


class TestKroneckerFusion:
    """Test suite for KroneckerFusion."""

    def test_basic_fusion(self):
        """Test basic Kronecker fusion."""
        fusion = KroneckerFusion(dim1=128, dim2=128, output_dim=256)
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        output = fusion(x1, x2)

        assert output.shape == (4, 256)

    def test_different_input_dims(self):
        """Test with different input dimensions."""
        fusion = KroneckerFusion(dim1=256, dim2=512, output_dim=128)
        x1 = torch.randn(2, 256)
        x2 = torch.randn(2, 512)

        output = fusion(x1, x2)

        assert output.shape == (2, 128)

    def test_bilinear_projection(self):
        """Test with bilinear projection."""
        fusion = KroneckerFusion(
            dim1=128, dim2=128, output_dim=256, use_bilinear=True
        )
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        output = fusion(x1, x2)

        assert output.shape == (4, 256)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = KroneckerFusion(dim1=64, dim2=64, output_dim=128)
        x1 = torch.randn(2, 64, requires_grad=True)
        x2 = torch.randn(2, 64, requires_grad=True)

        output = fusion(x1, x2)
        loss = output.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None
        assert not torch.isnan(x1.grad).any()
        assert not torch.isnan(x2.grad).any()

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        fusion = KroneckerFusion(dim1=128, dim2=128, output_dim=256)
        x1 = torch.randn(1, 128)
        x2 = torch.randn(1, 128)

        output = fusion(x1, x2)

        assert output.shape == (1, 256)


class TestCompactKroneckerFusion:
    """Test suite for CompactKroneckerFusion."""

    def test_basic_fusion(self):
        """Test basic compact Kronecker fusion."""
        fusion = CompactKroneckerFusion(
            dim1=512, dim2=512, output_dim=256, sketch_dim=2048
        )
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        output = fusion(x1, x2)

        assert output.shape == (4, 256)

    def test_memory_efficiency(self):
        """Test that compact version uses less memory than full."""
        # Full Kronecker would be 512*512 = 262,144 dimensions
        # Compact uses only 2048 dimensions
        fusion = CompactKroneckerFusion(
            dim1=512, dim2=512, output_dim=256, sketch_dim=2048
        )
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        output = fusion(x1, x2)

        assert output.shape == (4, 256)
        # Check that sketch dimension is much smaller than full Kronecker
        assert fusion.sketch_dim < fusion.dim1 * fusion.dim2

    def test_different_sketch_dims(self):
        """Test with different sketch dimensions."""
        for sketch_dim in [512, 1024, 2048]:
            fusion = CompactKroneckerFusion(
                dim1=256, dim2=256, output_dim=128, sketch_dim=sketch_dim
            )
            x1 = torch.randn(2, 256)
            x2 = torch.randn(2, 256)

            output = fusion(x1, x2)

            assert output.shape == (2, 128)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = CompactKroneckerFusion(
            dim1=128, dim2=128, output_dim=64, sketch_dim=512
        )
        x1 = torch.randn(2, 128, requires_grad=True)
        x2 = torch.randn(2, 128, requires_grad=True)

        output = fusion(x1, x2)
        loss = output.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestMultimodalKroneckerFusion:
    """Test suite for MultimodalKroneckerFusion."""

    def test_sequential_fusion(self):
        """Test sequential fusion strategy."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128, 128],
            output_dim=256,
            fusion_strategy="sequential",
        )
        features = [
            torch.randn(4, 128),
            torch.randn(4, 128),
            torch.randn(4, 128),
        ]

        output = fusion(features)

        assert output.shape == (4, 256)

    def test_pairwise_fusion(self):
        """Test pairwise fusion strategy."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128, 128],
            output_dim=256,
            fusion_strategy="pairwise",
        )
        features = [
            torch.randn(4, 128),
            torch.randn(4, 128),
            torch.randn(4, 128),
        ]

        output = fusion(features)

        assert output.shape == (4, 256)

    def test_star_fusion(self):
        """Test star fusion strategy."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128, 128],
            output_dim=256,
            fusion_strategy="star",
        )
        features = [
            torch.randn(4, 128),
            torch.randn(4, 128),
            torch.randn(4, 128),
        ]

        output = fusion(features)

        assert output.shape == (4, 256)

    def test_different_modality_dims(self):
        """Test with different modality dimensions."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[256, 512, 128],
            output_dim=256,
            fusion_strategy="sequential",
        )
        features = [
            torch.randn(2, 256),
            torch.randn(2, 512),
            torch.randn(2, 128),
        ]

        output = fusion(features)

        assert output.shape == (2, 256)

    def test_two_modalities(self):
        """Test with minimum number of modalities (2)."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128],
            output_dim=256,
            fusion_strategy="sequential",
        )
        features = [
            torch.randn(4, 128),
            torch.randn(4, 128),
        ]

        output = fusion(features)

        assert output.shape == (4, 256)

    def test_four_modalities(self):
        """Test with four modalities."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128, 128, 128],
            output_dim=256,
            fusion_strategy="sequential",
        )
        features = [
            torch.randn(2, 128),
            torch.randn(2, 128),
            torch.randn(2, 128),
            torch.randn(2, 128),
        ]

        output = fusion(features)

        assert output.shape == (2, 256)

    def test_invalid_num_modalities(self):
        """Test error handling for invalid number of modalities."""
        with pytest.raises(ValueError):
            MultimodalKroneckerFusion(
                modality_dims=[128],  # Only 1 modality
                output_dim=256,
            )

    def test_invalid_strategy(self):
        """Test error handling for invalid fusion strategy."""
        with pytest.raises(ValueError):
            MultimodalKroneckerFusion(
                modality_dims=[128, 128],
                output_dim=256,
                fusion_strategy="invalid",
            )

    def test_mismatched_features(self):
        """Test error handling for mismatched number of features."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128, 128],
            output_dim=256,
        )
        features = [
            torch.randn(4, 128),
            torch.randn(4, 128),
            # Missing third modality
        ]

        with pytest.raises(ValueError):
            fusion(features)

    def test_gradient_flow(self):
        """Test gradient flow through multimodal fusion."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[64, 64, 64],
            output_dim=128,
            fusion_strategy="sequential",
        )
        features = [
            torch.randn(2, 64, requires_grad=True),
            torch.randn(2, 64, requires_grad=True),
            torch.randn(2, 64, requires_grad=True),
        ]

        output = fusion(features)
        loss = output.sum()
        loss.backward()

        for feat in features:
            assert feat.grad is not None
            assert not torch.isnan(feat.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        fusion = MultimodalKroneckerFusion(
            modality_dims=[128, 128],
            output_dim=256,
        ).cuda()
        features = [
            torch.randn(4, 128).cuda(),
            torch.randn(4, 128).cuda(),
        ]

        output = fusion(features)

        assert output.shape == (4, 256)
        assert output.device.type == "cuda"
