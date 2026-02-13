"""
Unit tests for Self-Attention Fusion.
"""

import pytest
import torch

from med_core.fusion.self_attention import (
    AdditiveAttentionFusion,
    BilinearAttentionFusion,
    GatedAttentionFusion,
    MultimodalSelfAttentionFusion,
    SelfAttentionFusion,
)


class TestSelfAttentionFusion:
    """Test suite for SelfAttentionFusion."""

    def test_basic_fusion(self):
        """Test basic self-attention fusion."""
        fusion = SelfAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 256)

    def test_different_input_dims(self):
        """Test with different input dimensions."""
        fusion = SelfAttentionFusion(dim1=512, dim2=256, output_dim=128)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 256)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 128)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        fusion = SelfAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused, attn_weights = fusion(x1, x2, return_attention=True)

        assert fused.shape == (4, 256)
        assert attn_weights is not None
        assert attn_weights.shape == (4, 2, 2)  # [B, seq_len, seq_len]

    def test_different_num_heads(self):
        """Test with different number of attention heads."""
        for num_heads in [4, 8, 16]:
            fusion = SelfAttentionFusion(
                dim1=512, dim2=512, output_dim=256, num_heads=num_heads
            )
            x1 = torch.randn(4, 512)
            x2 = torch.randn(4, 512)

            fused = fusion(x1, x2)

            assert fused.shape == (4, 256)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = SelfAttentionFusion(dim1=128, dim2=128, output_dim=64)
        x1 = torch.randn(4, 128, requires_grad=True)
        x2 = torch.randn(4, 128, requires_grad=True)

        fused = fusion(x1, x2)
        loss = fused.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        fusion = SelfAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(1, 512)
        x2 = torch.randn(1, 512)

        fused = fusion(x1, x2)

        assert fused.shape == (1, 256)


class TestAdditiveAttentionFusion:
    """Test suite for AdditiveAttentionFusion."""

    def test_basic_fusion(self):
        """Test basic additive attention fusion."""
        fusion = AdditiveAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 256)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        fusion = AdditiveAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused, attn_weights = fusion(x1, x2, return_attention=True)

        assert fused.shape == (4, 256)
        assert attn_weights.shape == (4, 2)
        # Attention weights should sum to 1
        assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_different_attention_dim(self):
        """Test with different attention dimensions."""
        for attention_dim in [64, 128, 256]:
            fusion = AdditiveAttentionFusion(
                dim1=512, dim2=512, output_dim=256, attention_dim=attention_dim
            )
            x1 = torch.randn(4, 512)
            x2 = torch.randn(4, 512)

            fused = fusion(x1, x2)

            assert fused.shape == (4, 256)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = AdditiveAttentionFusion(dim1=128, dim2=128, output_dim=64)
        x1 = torch.randn(4, 128, requires_grad=True)
        x2 = torch.randn(4, 128, requires_grad=True)

        fused = fusion(x1, x2)
        loss = fused.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestBilinearAttentionFusion:
    """Test suite for BilinearAttentionFusion."""

    def test_basic_fusion(self):
        """Test basic bilinear attention fusion."""
        fusion = BilinearAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 256)

    def test_return_attention_score(self):
        """Test returning attention score."""
        fusion = BilinearAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused, attn_score = fusion(x1, x2, return_attention=True)

        assert fused.shape == (4, 256)
        assert attn_score.shape == (4, 1)
        # Attention score should be in [0, 1] (sigmoid)
        assert (attn_score >= 0).all() and (attn_score <= 1).all()

    def test_different_input_dims(self):
        """Test with different input dimensions."""
        fusion = BilinearAttentionFusion(dim1=512, dim2=256, output_dim=128)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 256)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 128)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = BilinearAttentionFusion(dim1=128, dim2=128, output_dim=64)
        x1 = torch.randn(4, 128, requires_grad=True)
        x2 = torch.randn(4, 128, requires_grad=True)

        fused = fusion(x1, x2)
        loss = fused.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestGatedAttentionFusion:
    """Test suite for GatedAttentionFusion."""

    def test_basic_fusion(self):
        """Test basic gated attention fusion."""
        fusion = GatedAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 256)

    def test_return_gates(self):
        """Test returning gate values."""
        fusion = GatedAttentionFusion(dim1=512, dim2=512, output_dim=256)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        fused, (gate1, gate2) = fusion(x1, x2, return_gates=True)

        assert fused.shape == (4, 256)
        assert gate1.shape == (4, 512)
        assert gate2.shape == (4, 512)
        # Gates should be in [0, 1] (sigmoid)
        assert (gate1 >= 0).all() and (gate1 <= 1).all()
        assert (gate2 >= 0).all() and (gate2 <= 1).all()

    def test_different_input_dims(self):
        """Test with different input dimensions."""
        fusion = GatedAttentionFusion(dim1=512, dim2=256, output_dim=128)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 256)

        fused = fusion(x1, x2)

        assert fused.shape == (4, 128)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = GatedAttentionFusion(dim1=128, dim2=128, output_dim=64)
        x1 = torch.randn(4, 128, requires_grad=True)
        x2 = torch.randn(4, 128, requires_grad=True)

        fused = fusion(x1, x2)
        loss = fused.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestMultimodalSelfAttentionFusion:
    """Test suite for MultimodalSelfAttentionFusion."""

    def test_three_modalities(self):
        """Test with three modalities."""
        fusion = MultimodalSelfAttentionFusion(
            modality_dims=[512, 512, 256],
            output_dim=256
        )
        features = [
            torch.randn(4, 512),
            torch.randn(4, 512),
            torch.randn(4, 256),
        ]

        fused = fusion(features)

        assert fused.shape == (4, 256)

    def test_four_modalities(self):
        """Test with four modalities."""
        fusion = MultimodalSelfAttentionFusion(
            modality_dims=[512, 256, 128, 64],
            output_dim=128
        )
        features = [
            torch.randn(4, 512),
            torch.randn(4, 256),
            torch.randn(4, 128),
            torch.randn(4, 64),
        ]

        fused = fusion(features)

        assert fused.shape == (4, 128)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        fusion = MultimodalSelfAttentionFusion(
            modality_dims=[512, 512, 256],
            output_dim=256
        )
        features = [
            torch.randn(4, 512),
            torch.randn(4, 512),
            torch.randn(4, 256),
        ]

        fused, attn_weights = fusion(features, return_attention=True)

        assert fused.shape == (4, 256)
        assert attn_weights is not None
        assert attn_weights.shape == (4, 3, 3)  # [B, num_modalities, num_modalities]

    def test_invalid_num_features(self):
        """Test error handling for wrong number of features."""
        fusion = MultimodalSelfAttentionFusion(
            modality_dims=[512, 512, 256],
            output_dim=256
        )
        features = [
            torch.randn(4, 512),
            torch.randn(4, 512),
        ]  # Only 2 instead of 3

        with pytest.raises(ValueError):
            fusion(features)

    def test_gradient_flow(self):
        """Test gradient flow."""
        fusion = MultimodalSelfAttentionFusion(
            modality_dims=[128, 128, 64],
            output_dim=64
        )
        features = [
            torch.randn(4, 128, requires_grad=True),
            torch.randn(4, 128, requires_grad=True),
            torch.randn(4, 64, requires_grad=True),
        ]

        fused = fusion(features)
        loss = fused.sum()
        loss.backward()

        for feat in features:
            assert feat.grad is not None


class TestIntegration:
    """Integration tests for self-attention fusion."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        fusion = SelfAttentionFusion(dim1=512, dim2=512, output_dim=256).cuda()
        x1 = torch.randn(4, 512).cuda()
        x2 = torch.randn(4, 512).cuda()

        fused = fusion(x1, x2)

        assert fused.shape == (4, 256)
        assert fused.device.type == "cuda"

    def test_compare_fusion_strategies(self):
        """Compare different fusion strategies."""
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        strategies = {
            'self_attention': SelfAttentionFusion(512, 512, 256),
            'additive': AdditiveAttentionFusion(512, 512, 256),
            'bilinear': BilinearAttentionFusion(512, 512, 256),
            'gated': GatedAttentionFusion(512, 512, 256),
        }

        results = {}
        for name, fusion in strategies.items():
            fusion.eval()
            with torch.no_grad():
                fused = fusion(x1, x2)
            results[name] = fused
            assert fused.shape == (4, 256)

        # All strategies should produce different results
        for name1 in results:
            for name2 in results:
                if name1 != name2:
                    assert not torch.allclose(results[name1], results[name2])

    def test_training_vs_eval_mode(self):
        """Test behavior in training vs eval mode."""
        fusion = SelfAttentionFusion(dim1=512, dim2=512, output_dim=256, dropout=0.5)
        x1 = torch.randn(4, 512)
        x2 = torch.randn(4, 512)

        # Eval mode
        fusion.eval()
        with torch.no_grad():
            fused_eval1 = fusion(x1, x2)
            fused_eval2 = fusion(x1, x2)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(fused_eval1, fused_eval2)

    def test_multimodal_vs_pairwise(self):
        """Test multimodal fusion vs pairwise fusion."""
        # Multimodal fusion
        multimodal_fusion = MultimodalSelfAttentionFusion(
            modality_dims=[512, 512, 256],
            output_dim=256
        )
        features = [
            torch.randn(4, 512),
            torch.randn(4, 512),
            torch.randn(4, 256),
        ]

        multimodal_fusion.eval()
        with torch.no_grad():
            fused_multimodal = multimodal_fusion(features)

        assert fused_multimodal.shape == (4, 256)
