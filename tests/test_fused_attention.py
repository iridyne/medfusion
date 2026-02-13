"""
Unit tests for Fused Attention Fusion.
"""

import pytest
import torch

from med_core.fusion.fused_attention import (
    CrossModalAttention,
    FusedAttentionFusion,
    MultimodalFusedAttention,
)


class TestFusedAttentionFusion:
    """Test suite for FusedAttentionFusion."""

    def test_basic_fusion(self):
        """Test basic fused attention fusion."""
        fusion = FusedAttentionFusion(dim1=128, dim2=128, output_dim=256)
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        output = fusion(x1, x2)

        assert output.shape == (4, 256)

    def test_different_input_dims(self):
        """Test with different input dimensions."""
        fusion = FusedAttentionFusion(dim1=256, dim2=512, output_dim=128)
        x1 = torch.randn(2, 256)
        x2 = torch.randn(2, 512)

        output = fusion(x1, x2)

        assert output.shape == (2, 128)

    def test_without_kronecker(self):
        """Test with Kronecker product disabled."""
        fusion = FusedAttentionFusion(
            dim1=128, dim2=128, output_dim=256, use_kronecker=False
        )
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        output = fusion(x1, x2)

        assert output.shape == (4, 256)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        fusion = FusedAttentionFusion(dim1=128, dim2=128, output_dim=256)
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        output, attn_weights = fusion(x1, x2, return_attention=True)

        assert output.shape == (4, 256)
        assert attn_weights is not None
        assert attn_weights.shape[0] == 4  # Batch size

    def test_different_num_heads(self):
        """Test with different number of attention heads."""
        for num_heads in [4, 8, 16]:
            fusion = FusedAttentionFusion(
                dim1=128, dim2=128, output_dim=256, num_heads=num_heads
            )
            x1 = torch.randn(2, 128)
            x2 = torch.randn(2, 128)

            output = fusion(x1, x2)

            assert output.shape == (2, 256)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = FusedAttentionFusion(dim1=64, dim2=64, output_dim=128)
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
        fusion = FusedAttentionFusion(dim1=128, dim2=128, output_dim=256)
        x1 = torch.randn(1, 128)
        x2 = torch.randn(1, 128)

        output = fusion(x1, x2)

        assert output.shape == (1, 256)


class TestCrossModalAttention:
    """Test suite for CrossModalAttention."""

    def test_basic_attention(self):
        """Test basic cross-modal attention."""
        attention = CrossModalAttention(dim1=128, dim2=128, num_heads=8)
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        attended_x1, attended_x2, attn_weights = attention(x1, x2)

        assert attended_x1.shape == (4, 128)
        assert attended_x2.shape == (4, 128)
        assert attn_weights is not None

    def test_different_dims(self):
        """Test with different modality dimensions."""
        attention = CrossModalAttention(dim1=256, dim2=512, num_heads=8)
        x1 = torch.randn(2, 256)
        x2 = torch.randn(2, 512)

        attended_x1, attended_x2, attn_weights = attention(x1, x2)

        assert attended_x1.shape == (2, 256)
        assert attended_x2.shape == (2, 512)

    def test_attention_symmetry(self):
        """Test that attention is applied symmetrically."""
        attention = CrossModalAttention(dim1=128, dim2=128, num_heads=8)
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)

        attended_x1, attended_x2, _ = attention(x1, x2)

        # Both modalities should be attended
        assert not torch.allclose(attended_x1, x1)
        assert not torch.allclose(attended_x2, x2)

    def test_gradient_flow(self):
        """Test gradient flow through attention."""
        attention = CrossModalAttention(dim1=64, dim2=64, num_heads=4)
        x1 = torch.randn(2, 64, requires_grad=True)
        x2 = torch.randn(2, 64, requires_grad=True)

        attended_x1, attended_x2, _ = attention(x1, x2)
        loss = (attended_x1.sum() + attended_x2.sum())
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestMultimodalFusedAttention:
    """Test suite for MultimodalFusedAttention."""

    def test_sequential_fusion(self):
        """Test sequential fusion strategy."""
        fusion = MultimodalFusedAttention(
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
        fusion = MultimodalFusedAttention(
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
        fusion = MultimodalFusedAttention(
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
        fusion = MultimodalFusedAttention(
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
        fusion = MultimodalFusedAttention(
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
        fusion = MultimodalFusedAttention(
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
            MultimodalFusedAttention(
                modality_dims=[128],
                output_dim=256,
            )

    def test_invalid_strategy(self):
        """Test error handling for invalid fusion strategy."""
        with pytest.raises(ValueError):
            MultimodalFusedAttention(
                modality_dims=[128, 128],
                output_dim=256,
                fusion_strategy="invalid",
            )

    def test_mismatched_features(self):
        """Test error handling for mismatched number of features."""
        fusion = MultimodalFusedAttention(
            modality_dims=[128, 128, 128],
            output_dim=256,
        )
        features = [
            torch.randn(4, 128),
            torch.randn(4, 128),
        ]

        with pytest.raises(ValueError):
            fusion(features)

    def test_gradient_flow(self):
        """Test gradient flow through multimodal fusion."""
        fusion = MultimodalFusedAttention(
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
        fusion = MultimodalFusedAttention(
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

    def test_different_num_heads(self):
        """Test with different number of attention heads."""
        for num_heads in [4, 8]:
            fusion = MultimodalFusedAttention(
                modality_dims=[128, 128],
                output_dim=256,
                num_heads=num_heads,
            )
            features = [
                torch.randn(2, 128),
                torch.randn(2, 128),
            ]

            output = fusion(features)

            assert output.shape == (2, 256)
