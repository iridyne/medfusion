"""
Unit tests for MIL Aggregators.
"""

import pytest
import torch

from med_core.aggregators.mil import (
    AttentionAggregator,
    DeepSetsAggregator,
    GatedAttentionAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    MILAggregator,
    TransformerAggregator,
)


class TestMeanPoolingAggregator:
    """Test suite for MeanPoolingAggregator."""

    def test_basic_aggregation(self):
        """Test basic mean pooling."""
        aggregator = MeanPoolingAggregator(input_dim=512)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_different_num_instances(self):
        """Test with different number of instances."""
        aggregator = MeanPoolingAggregator(input_dim=512)

        for num_instances in [1, 5, 10, 20]:
            features = torch.randn(4, num_instances, 512)
            aggregated = aggregator(features)
            assert aggregated.shape == (4, 512)

    def test_gradient_flow(self):
        """Test gradient flow."""
        aggregator = MeanPoolingAggregator(input_dim=128)
        features = torch.randn(4, 10, 128, requires_grad=True)

        aggregated = aggregator(features)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None


class TestMaxPoolingAggregator:
    """Test suite for MaxPoolingAggregator."""

    def test_basic_aggregation(self):
        """Test basic max pooling."""
        aggregator = MaxPoolingAggregator(input_dim=512)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_max_property(self):
        """Test that max pooling returns maximum values."""
        aggregator = MaxPoolingAggregator(input_dim=512)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        # Check that aggregated values are indeed max
        for i in range(4):
            for j in range(512):
                assert aggregated[i, j] == features[i, :, j].max()

    def test_gradient_flow(self):
        """Test gradient flow."""
        aggregator = MaxPoolingAggregator(input_dim=128)
        features = torch.randn(4, 10, 128, requires_grad=True)

        aggregated = aggregator(features)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None


class TestAttentionAggregator:
    """Test suite for AttentionAggregator."""

    def test_basic_aggregation(self):
        """Test basic attention aggregation."""
        aggregator = AttentionAggregator(input_dim=512, attention_dim=128)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        aggregator = AttentionAggregator(input_dim=512, attention_dim=128)
        features = torch.randn(4, 10, 512)

        aggregated, attention_weights = aggregator(features, return_attention=True)

        assert aggregated.shape == (4, 512)
        assert attention_weights.shape == (4, 10, 1)
        # Attention weights should sum to 1
        assert torch.allclose(
            attention_weights.sum(dim=1), torch.ones(4, 1), atol=1e-5
        )

    def test_different_attention_dim(self):
        """Test with different attention dimensions."""
        for attention_dim in [64, 128, 256]:
            aggregator = AttentionAggregator(input_dim=512, attention_dim=attention_dim)
            features = torch.randn(4, 10, 512)

            aggregated = aggregator(features)

            assert aggregated.shape == (4, 512)

    def test_gradient_flow(self):
        """Test gradient flow."""
        aggregator = AttentionAggregator(input_dim=128, attention_dim=64)
        features = torch.randn(4, 10, 128, requires_grad=True)

        aggregated = aggregator(features)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None


class TestGatedAttentionAggregator:
    """Test suite for GatedAttentionAggregator."""

    def test_basic_aggregation(self):
        """Test basic gated attention aggregation."""
        aggregator = GatedAttentionAggregator(input_dim=512, attention_dim=128)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        aggregator = GatedAttentionAggregator(input_dim=512, attention_dim=128)
        features = torch.randn(4, 10, 512)

        aggregated, attention_weights = aggregator(features, return_attention=True)

        assert aggregated.shape == (4, 512)
        assert attention_weights.shape == (4, 10, 1)
        assert torch.allclose(
            attention_weights.sum(dim=1), torch.ones(4, 1), atol=1e-5
        )

    def test_gradient_flow(self):
        """Test gradient flow."""
        aggregator = GatedAttentionAggregator(input_dim=128, attention_dim=64)
        features = torch.randn(4, 10, 128, requires_grad=True)

        aggregated = aggregator(features)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None


class TestDeepSetsAggregator:
    """Test suite for DeepSetsAggregator."""

    def test_basic_aggregation(self):
        """Test basic deep sets aggregation."""
        aggregator = DeepSetsAggregator(input_dim=512, output_dim=256)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 256)

    def test_permutation_invariance(self):
        """Test permutation invariance property."""
        aggregator = DeepSetsAggregator(input_dim=512, output_dim=256)
        features = torch.randn(4, 10, 512)

        # Shuffle instances
        indices = torch.randperm(10)
        features_shuffled = features[:, indices, :]

        aggregator.eval()
        with torch.no_grad():
            aggregated1 = aggregator(features)
            aggregated2 = aggregator(features_shuffled)

        # Results should be the same (permutation invariant)
        assert torch.allclose(aggregated1, aggregated2, atol=1e-5)

    def test_different_output_dim(self):
        """Test with different output dimensions."""
        for output_dim in [128, 256, 512]:
            aggregator = DeepSetsAggregator(input_dim=512, output_dim=output_dim)
            features = torch.randn(4, 10, 512)

            aggregated = aggregator(features)

            assert aggregated.shape == (4, output_dim)

    def test_gradient_flow(self):
        """Test gradient flow."""
        aggregator = DeepSetsAggregator(input_dim=128, output_dim=64)
        features = torch.randn(4, 10, 128, requires_grad=True)

        aggregated = aggregator(features)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None


class TestTransformerAggregator:
    """Test suite for TransformerAggregator."""

    def test_basic_aggregation(self):
        """Test basic transformer aggregation."""
        aggregator = TransformerAggregator(input_dim=512, num_heads=8)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_different_num_heads(self):
        """Test with different number of attention heads."""
        for num_heads in [4, 8, 16]:
            aggregator = TransformerAggregator(input_dim=512, num_heads=num_heads)
            features = torch.randn(4, 10, 512)

            aggregated = aggregator(features)

            assert aggregated.shape == (4, 512)

    def test_different_num_layers(self):
        """Test with different number of layers."""
        for num_layers in [1, 2, 4]:
            aggregator = TransformerAggregator(
                input_dim=512, num_heads=8, num_layers=num_layers
            )
            features = torch.randn(4, 10, 512)

            aggregated = aggregator(features)

            assert aggregated.shape == (4, 512)

    def test_gradient_flow(self):
        """Test gradient flow."""
        aggregator = TransformerAggregator(input_dim=128, num_heads=4)
        features = torch.randn(4, 10, 128, requires_grad=True)

        aggregated = aggregator(features)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None


class TestMILAggregator:
    """Test suite for unified MILAggregator."""

    def test_mean_strategy(self):
        """Test mean pooling strategy."""
        aggregator = MILAggregator(input_dim=512, strategy='mean')
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_max_strategy(self):
        """Test max pooling strategy."""
        aggregator = MILAggregator(input_dim=512, strategy='max')
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_attention_strategy(self):
        """Test attention strategy."""
        aggregator = MILAggregator(input_dim=512, strategy='attention')
        features = torch.randn(4, 10, 512)

        aggregated, attention_weights = aggregator(features, return_attention=True)

        assert aggregated.shape == (4, 512)
        assert attention_weights.shape == (4, 10, 1)

    def test_gated_strategy(self):
        """Test gated attention strategy."""
        aggregator = MILAggregator(input_dim=512, strategy='gated')
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_deepsets_strategy(self):
        """Test deep sets strategy."""
        aggregator = MILAggregator(input_dim=512, strategy='deepsets', output_dim=256)
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 256)

    def test_transformer_strategy(self):
        """Test transformer strategy."""
        aggregator = MILAggregator(input_dim=512, strategy='transformer')
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)

    def test_output_projection(self):
        """Test output projection."""
        aggregator = MILAggregator(
            input_dim=512, strategy='attention', output_dim=256
        )
        features = torch.randn(4, 10, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 256)

    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError):
            MILAggregator(input_dim=512, strategy='invalid')

    def test_gradient_flow_all_strategies(self):
        """Test gradient flow for all strategies."""
        strategies = ['mean', 'max', 'attention', 'gated', 'deepsets', 'transformer']

        for strategy in strategies:
            aggregator = MILAggregator(input_dim=128, strategy=strategy)
            features = torch.randn(4, 10, 128, requires_grad=True)

            aggregated = aggregator(features)
            loss = aggregated.sum()
            loss.backward()

            assert features.grad is not None, f"Gradient flow failed for {strategy}"


class TestIntegration:
    """Integration tests for MIL aggregators."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        aggregator = AttentionAggregator(input_dim=512, attention_dim=128).cuda()
        features = torch.randn(4, 10, 512).cuda()

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)
        assert aggregated.device.type == "cuda"

    def test_compare_strategies(self):
        """Compare different aggregation strategies."""
        features = torch.randn(4, 10, 512)

        strategies = ['mean', 'max', 'attention', 'gated', 'deepsets', 'transformer']
        results = {}

        for strategy in strategies:
            aggregator = MILAggregator(input_dim=512, strategy=strategy)
            aggregator.eval()
            with torch.no_grad():
                aggregated = aggregator(features)
            results[strategy] = aggregated
            assert aggregated.shape == (4, 512)

        # Different strategies should produce different results
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                assert not torch.allclose(results[s1], results[s2])

    def test_training_vs_eval_mode(self):
        """Test behavior in training vs eval mode."""
        aggregator = AttentionAggregator(input_dim=512, attention_dim=128, dropout=0.5)
        features = torch.randn(4, 10, 512)

        # Eval mode
        aggregator.eval()
        with torch.no_grad():
            aggregated_eval1 = aggregator(features)
            aggregated_eval2 = aggregator(features)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(aggregated_eval1, aggregated_eval2)

    def test_variable_num_instances(self):
        """Test with variable number of instances per batch."""
        aggregator = MILAggregator(input_dim=512, strategy='attention')

        # Different number of instances
        for num_instances in [1, 5, 10, 20, 50]:
            features = torch.randn(4, num_instances, 512)
            aggregated = aggregator(features)
            assert aggregated.shape == (4, 512)

    def test_single_instance(self):
        """Test with single instance (edge case)."""
        aggregator = MILAggregator(input_dim=512, strategy='attention')
        features = torch.randn(4, 1, 512)

        aggregated = aggregator(features)

        assert aggregated.shape == (4, 512)
        # With single instance, should be close to the instance itself
        assert torch.allclose(aggregated, features.squeeze(1), atol=1e-5)
