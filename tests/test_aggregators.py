"""
Tests for MIL aggregators.
"""

import pytest
import torch
import torch.nn as nn

from med_core.aggregators.mil import (
    AttentionAggregator,
    DeepSetsAggregator,
    GatedAttentionAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    MILAggregator,
    TransformerAggregator,
)


@pytest.fixture
def batch_features():
    """Create batch of instance features."""
    batch_size = 4
    num_instances = 10
    feature_dim = 512
    return torch.randn(batch_size, num_instances, feature_dim)


class TestMeanPoolingAggregator:
    """Tests for MeanPoolingAggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = MeanPoolingAggregator(input_dim=512)
        assert aggregator.input_dim == 512
        assert isinstance(aggregator, nn.Module)

    def test_forward(self, batch_features):
        """Test forward pass."""
        aggregator = MeanPoolingAggregator(input_dim=512)
        output = aggregator(batch_features)

        assert output.shape == (4, 512)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mean_computation(self):
        """Test that mean is computed correctly."""
        aggregator = MeanPoolingAggregator(input_dim=3)

        # Create simple input
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ]
        )

        output = aggregator(x)
        expected = torch.tensor([[2.5, 3.5, 4.5]])

        assert torch.allclose(output, expected)


class TestMaxPoolingAggregator:
    """Tests for MaxPoolingAggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = MaxPoolingAggregator(input_dim=512)
        assert aggregator.input_dim == 512

    def test_forward(self, batch_features):
        """Test forward pass."""
        aggregator = MaxPoolingAggregator(input_dim=512)
        output = aggregator(batch_features)

        assert output.shape == (4, 512)
        assert not torch.isnan(output).any()

    def test_max_computation(self):
        """Test that max is computed correctly."""
        aggregator = MaxPoolingAggregator(input_dim=3)

        x = torch.tensor(
            [
                [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]],
            ]
        )

        output = aggregator(x)
        expected = torch.tensor([[4.0, 5.0, 6.0]])

        assert torch.allclose(output, expected)


class TestAttentionAggregator:
    """Tests for AttentionAggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = AttentionAggregator(input_dim=512, attention_dim=128, dropout=0.1)
        assert aggregator.input_dim == 512
        assert aggregator.attention_dim == 128

    def test_forward(self, batch_features):
        """Test forward pass."""
        aggregator = AttentionAggregator(input_dim=512)
        output = aggregator(batch_features)

        assert output.shape == (4, 512)
        assert not torch.isnan(output).any()

    def test_forward_with_attention(self, batch_features):
        """Test forward pass with attention weights."""
        aggregator = AttentionAggregator(input_dim=512)
        output, attention = aggregator(batch_features, return_attention=True)

        assert output.shape == (4, 512)
        assert attention.shape == (4, 10, 1)

        # Check attention weights sum to 1
        attention_sum = attention.sum(dim=1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)

    def test_attention_weights_positive(self, batch_features):
        """Test that attention weights are positive."""
        aggregator = AttentionAggregator(input_dim=512)
        _, attention = aggregator(batch_features, return_attention=True)

        assert (attention >= 0).all()
        assert (attention <= 1).all()


class TestGatedAttentionAggregator:
    """Tests for GatedAttentionAggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = GatedAttentionAggregator(
            input_dim=512, attention_dim=128, dropout=0.1
        )
        assert aggregator.input_dim == 512
        assert aggregator.attention_dim == 128

    def test_forward(self, batch_features):
        """Test forward pass."""
        aggregator = GatedAttentionAggregator(input_dim=512)
        output = aggregator(batch_features)

        assert output.shape == (4, 512)
        assert not torch.isnan(output).any()

    def test_forward_with_attention(self, batch_features):
        """Test forward pass with attention weights."""
        aggregator = GatedAttentionAggregator(input_dim=512)
        output, attention = aggregator(batch_features, return_attention=True)

        assert output.shape == (4, 512)
        assert attention.shape == (4, 10, 1)

        # Check attention weights sum to 1
        attention_sum = attention.sum(dim=1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)


class TestDeepSetsAggregator:
    """Tests for DeepSetsAggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = DeepSetsAggregator(
            input_dim=512, hidden_dim=256, output_dim=128, dropout=0.1
        )
        assert aggregator.input_dim == 512
        assert aggregator.hidden_dim == 256
        assert aggregator.output_dim == 128

    def test_initialization_default_output_dim(self):
        """Test initialization with default output_dim."""
        aggregator = DeepSetsAggregator(input_dim=512)
        assert aggregator.output_dim == 512

    def test_forward(self, batch_features):
        """Test forward pass."""
        aggregator = DeepSetsAggregator(input_dim=512, output_dim=256)
        output = aggregator(batch_features)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_permutation_invariance(self):
        """Test that aggregator is permutation invariant."""
        aggregator = DeepSetsAggregator(input_dim=3, hidden_dim=8, output_dim=3)
        aggregator.eval()

        x = torch.randn(2, 5, 3)

        # Permute instances
        perm = torch.randperm(5)
        x_permuted = x[:, perm, :]

        with torch.no_grad():
            output1 = aggregator(x)
            output2 = aggregator(x_permuted)

        assert torch.allclose(output1, output2, atol=1e-5)


class TestTransformerAggregator:
    """Tests for TransformerAggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = TransformerAggregator(
            input_dim=512, num_heads=8, num_layers=2, dropout=0.1
        )
        assert aggregator.input_dim == 512
        assert aggregator.num_heads == 8
        assert aggregator.num_layers == 2

    def test_forward(self, batch_features):
        """Test forward pass."""
        aggregator = TransformerAggregator(input_dim=512)
        output = aggregator(batch_features)

        assert output.shape == (4, 512)
        assert not torch.isnan(output).any()

    def test_query_token(self):
        """Test that query token is learnable."""
        aggregator = TransformerAggregator(input_dim=512)
        assert aggregator.query_token.requires_grad

    def test_different_num_instances(self):
        """Test with different number of instances."""
        aggregator = TransformerAggregator(input_dim=512)

        x1 = torch.randn(2, 5, 512)
        x2 = torch.randn(2, 10, 512)

        output1 = aggregator(x1)
        output2 = aggregator(x2)

        assert output1.shape == (2, 512)
        assert output2.shape == (2, 512)


class TestMILAggregator:
    """Tests for unified MILAggregator."""

    @pytest.mark.parametrize(
        "strategy", ["mean", "max", "attention", "gated", "deepsets", "transformer"]
    )
    def test_all_strategies(self, strategy, batch_features):
        """Test all aggregation strategies."""
        aggregator = MILAggregator(input_dim=512, strategy=strategy)
        output = aggregator(batch_features)

        assert output.shape == (4, 512)
        assert not torch.isnan(output).any()

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            MILAggregator(input_dim=512, strategy="invalid")

    def test_output_projection(self, batch_features):
        """Test output projection."""
        aggregator = MILAggregator(input_dim=512, strategy="mean", output_dim=256)
        output = aggregator(batch_features)

        assert output.shape == (4, 256)

    def test_attention_return(self, batch_features):
        """Test returning attention weights."""
        aggregator = MILAggregator(input_dim=512, strategy="attention")
        output, attention = aggregator(batch_features, return_attention=True)

        assert output.shape == (4, 512)
        assert attention.shape == (4, 10, 1)

    def test_gated_attention_return(self, batch_features):
        """Test returning gated attention weights."""
        aggregator = MILAggregator(input_dim=512, strategy="gated")
        output, attention = aggregator(batch_features, return_attention=True)

        assert output.shape == (4, 512)
        assert attention.shape == (4, 10, 1)

    def test_no_attention_for_mean(self, batch_features):
        """Test that mean strategy doesn't return attention."""
        aggregator = MILAggregator(input_dim=512, strategy="mean")
        output = aggregator(batch_features, return_attention=True)

        # Should only return output, not tuple
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 512)

    def test_deepsets_with_custom_dims(self, batch_features):
        """Test DeepSets with custom dimensions."""
        aggregator = MILAggregator(
            input_dim=512, strategy="deepsets", hidden_dim=256, output_dim=128
        )
        output = aggregator(batch_features)

        assert output.shape == (4, 128)

    def test_transformer_with_custom_params(self, batch_features):
        """Test Transformer with custom parameters."""
        aggregator = MILAggregator(
            input_dim=512, strategy="transformer", num_heads=4, num_layers=3
        )
        output = aggregator(batch_features)

        assert output.shape == (4, 512)


class TestAggregatorGradients:
    """Test gradient flow through aggregators."""

    @pytest.mark.parametrize(
        "aggregator_class",
        [
            MeanPoolingAggregator,
            MaxPoolingAggregator,
            AttentionAggregator,
            GatedAttentionAggregator,
            DeepSetsAggregator,
            TransformerAggregator,
        ],
    )
    def test_gradient_flow(self, aggregator_class):
        """Test that gradients flow through aggregator."""
        if aggregator_class == DeepSetsAggregator:
            aggregator = aggregator_class(input_dim=512, output_dim=512)
        else:
            aggregator = aggregator_class(input_dim=512)

        x = torch.randn(2, 5, 512, requires_grad=True)

        if aggregator_class in [AttentionAggregator, GatedAttentionAggregator]:
            output = aggregator(x, return_attention=False)
        else:
            output = aggregator(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestAggregatorEdgeCases:
    """Test edge cases for aggregators."""

    def test_single_instance(self):
        """Test with single instance."""
        aggregator = MILAggregator(input_dim=512, strategy="attention")
        x = torch.randn(2, 1, 512)
        output = aggregator(x)

        assert output.shape == (2, 512)

    def test_large_batch(self):
        """Test with large batch size."""
        aggregator = MILAggregator(input_dim=512, strategy="mean")
        x = torch.randn(128, 10, 512)
        output = aggregator(x)

        assert output.shape == (128, 512)

    def test_many_instances(self):
        """Test with many instances."""
        aggregator = MILAggregator(input_dim=512, strategy="attention")
        x = torch.randn(2, 100, 512)
        output = aggregator(x)

        assert output.shape == (2, 512)

    def test_eval_mode(self):
        """Test in evaluation mode."""
        aggregator = MILAggregator(input_dim=512, strategy="attention")
        aggregator.eval()

        x = torch.randn(2, 10, 512)

        with torch.no_grad():
            output = aggregator(x)

        assert output.shape == (2, 512)


class TestAggregatorComparison:
    """Compare different aggregation strategies."""

    def test_mean_vs_max(self):
        """Test that mean and max produce different results."""
        x = torch.randn(2, 10, 512)

        mean_agg = MeanPoolingAggregator(input_dim=512)
        max_agg = MaxPoolingAggregator(input_dim=512)

        mean_output = mean_agg(x)
        max_output = max_agg(x)

        # Should be different
        assert not torch.allclose(mean_output, max_output)

    def test_attention_learns_weights(self):
        """Test that attention aggregator learns different weights."""
        aggregator = AttentionAggregator(input_dim=512)

        # Create two different inputs
        x1 = torch.randn(2, 10, 512)
        x2 = torch.randn(2, 10, 512)

        _, attn1 = aggregator(x1, return_attention=True)
        _, attn2 = aggregator(x2, return_attention=True)

        # Attention weights should be different for different inputs
        assert not torch.allclose(attn1, attn2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
