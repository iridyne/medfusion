"""
Tests for view aggregator modules.

Tests all 5 view aggregation strategies:
- MaxPoolAggregator
- MeanPoolAggregator
- AttentionAggregator
- CrossViewAttentionAggregator
- LearnedWeightAggregator
"""

import pytest
import torch

from med_core.backbones import create_view_aggregator


class TestViewAggregators:
    """Test view aggregator modules."""

    @pytest.fixture
    def view_features(self, batch_size):
        """Create sample view features as dictionary."""
        feature_dim = 128
        return {
            "axial": torch.randn(batch_size, feature_dim),
            "coronal": torch.randn(batch_size, feature_dim),
            "sagittal": torch.randn(batch_size, feature_dim),
        }

    @pytest.fixture
    def view_mask(self, batch_size):
        """Create sample view mask (some views missing)."""
        return {
            "axial": torch.ones(batch_size, dtype=torch.bool),
            "coronal": torch.tensor([True, True, False, False]),
            "sagittal": torch.tensor([True, False, True, False]),
        }

    def test_max_pool_aggregator(self, view_features, batch_size):
        """Test MaxPoolAggregator."""
        aggregator = create_view_aggregator(
            aggregator_type="max",
            feature_dim=128,
            view_names=list(view_features.keys()),
        )

        output, _ = aggregator(view_features)
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_mean_pool_aggregator(self, view_features, batch_size):
        """Test MeanPoolAggregator."""
        aggregator = create_view_aggregator(
            aggregator_type="mean",
            feature_dim=128,
            view_names=list(view_features.keys()),
        )

        output, _ = aggregator(view_features)
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_mean_pool_with_mask(self, view_features, view_mask, batch_size):
        """Test MeanPoolAggregator with view mask."""
        aggregator = create_view_aggregator(
            aggregator_type="mean",
            feature_dim=128,
            view_names=list(view_features.keys()),
        )

        output, _ = aggregator(view_features, view_mask=view_mask)
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_attention_aggregator(self, view_features, batch_size):
        """Test AttentionAggregator."""
        aggregator = create_view_aggregator(
            aggregator_type="attention",
            feature_dim=128,
            view_names=list(view_features.keys()),
            num_heads=4,
        )

        output, _ = aggregator(view_features)
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_attention_aggregator_returns_weights(self, view_features, batch_size):
        """Test AttentionAggregator returns attention weights."""
        aggregator = create_view_aggregator(
            aggregator_type="attention",
            feature_dim=128,
            view_names=list(view_features.keys()),
            num_heads=4,
        )

        # Aggregator always returns attention weights in metadata dict
        output, metadata = aggregator(view_features)

        assert output.shape == (batch_size, 128)
        assert "attention_weights" in metadata
        attn_weights_dict = metadata["attention_weights"]
        assert isinstance(attn_weights_dict, dict)
        # Check each view has attention weights
        for view_name in view_features.keys():
            assert view_name in attn_weights_dict
            assert attn_weights_dict[view_name].shape[0] == batch_size

    def test_cross_view_attention_aggregator(self, view_features, batch_size):
        """Test CrossViewAttentionAggregator."""
        aggregator = create_view_aggregator(
            aggregator_type="cross_attention",
            feature_dim=128,
            view_names=list(view_features.keys()),
            num_heads=4,
        )

        output, _ = aggregator(view_features)
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_learned_weight_aggregator(self, view_features, batch_size):
        """Test LearnedWeightAggregator."""
        aggregator = create_view_aggregator(
            aggregator_type="learned_weight",
            feature_dim=128,
            view_names=list(view_features.keys()),
        )

        output, _ = aggregator(view_features)
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_learned_weight_aggregator_weights_sum_to_one(self, view_features):
        """Test LearnedWeightAggregator weights sum to 1."""
        aggregator = create_view_aggregator(
            aggregator_type="learned_weight",
            feature_dim=128,
            view_names=list(view_features.keys()),
        )

        # Get learned weights
        weights = torch.softmax(aggregator.view_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_aggregator_with_single_view(self, batch_size):
        """Test aggregators work with single view."""
        view_features = {"single": torch.randn(batch_size, 128)}

        for agg_type in [
            "max",
            "mean",
            "attention",
            "cross_attention",
            "learned_weight",
        ]:
            aggregator = create_view_aggregator(
                aggregator_type=agg_type,
                feature_dim=128,
                view_names=["single"],
            )

            output, _ = aggregator(view_features)
            assert output.shape == (batch_size, 128)

    def test_aggregator_with_many_views(self, batch_size):
        """Test aggregators work with many views."""
        view_features = {f"view_{i}": torch.randn(batch_size, 128) for i in range(10)}

        for agg_type in [
            "max",
            "mean",
            "attention",
            "cross_attention",
            "learned_weight",
        ]:
            aggregator = create_view_aggregator(
                aggregator_type=agg_type,
                feature_dim=128,
                view_names=list(view_features.keys()),
            )

            output, _ = aggregator(view_features)
            assert output.shape == (batch_size, 128)

    def test_aggregator_gradient_flow(self, view_features):
        """Test gradients flow through aggregators."""
        for agg_type in [
            "max",
            "mean",
            "attention",
            "cross_attention",
            "learned_weight",
        ]:
            aggregator = create_view_aggregator(
                aggregator_type=agg_type,
                feature_dim=128,
                view_names=list(view_features.keys()),
            )

            # Make input require grad
            view_features_grad = {
                k: v.clone().requires_grad_(True) for k, v in view_features.items()
            }

            output, _ = aggregator(view_features_grad)
            loss = output.sum()
            loss.backward()

            # Check gradients exist
            for v in view_features_grad.values():
                assert v.grad is not None
                assert not torch.isnan(v.grad).any()

    def test_invalid_aggregator_type(self):
        """Test invalid aggregator type raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_view_aggregator(
                aggregator_type="invalid_type",
                feature_dim=128,
                view_names=["view1", "view2"],
            )

    def test_aggregator_output_dim(self, view_features, batch_size):
        """Test aggregator output dimension matches feature_dim."""
        feature_dim = 128

        aggregator = create_view_aggregator(
            aggregator_type="attention",
            feature_dim=feature_dim,
            view_names=list(view_features.keys()),
            num_heads=4,
        )

        output, _ = aggregator(view_features)
        assert output.shape == (batch_size, feature_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
