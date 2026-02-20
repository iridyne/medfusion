"""
Tests for classification heads.
"""

import pytest
import torch

from med_core.heads.classification import (
    AttentionClassificationHead,
    ClassificationHead,
    EnsembleClassificationHead,
    MultiLabelClassificationHead,
    OrdinalClassificationHead,
)


@pytest.fixture
def batch_features():
    """Create batch of features."""
    return torch.randn(8, 512)


class TestClassificationHead:
    """Tests for ClassificationHead."""

    def test_initialization(self):
        """Test head initialization."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        assert head.input_dim == 512
        assert head.num_classes == 4

    def test_forward(self, batch_features):
        """Test forward pass."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        logits = head(batch_features)

        assert logits.shape == (8, 4)
        assert not torch.isnan(logits).any()

    def test_with_hidden_layers(self, batch_features):
        """Test with hidden layers."""
        head = ClassificationHead(
            input_dim=512,
            num_classes=4,
            hidden_dims=[256, 128]
        )
        logits = head(batch_features)
        assert logits.shape == (8, 4)

    @pytest.mark.parametrize("activation", ["relu", "gelu", "leaky_relu"])
    def test_activations(self, activation, batch_features):
        """Test different activation functions."""
        head = ClassificationHead(
            input_dim=512,
            num_classes=4,
            hidden_dims=[256],
            activation=activation
        )
        logits = head(batch_features)
        assert logits.shape == (8, 4)

    def test_with_batch_norm(self, batch_features):
        """Test with batch normalization."""
        head = ClassificationHead(
            input_dim=512,
            num_classes=4,
            hidden_dims=[256],
            use_batch_norm=True
        )
        logits = head(batch_features)
        assert logits.shape == (8, 4)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        x = torch.randn(8, 512, requires_grad=True)

        logits = head(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None


class TestMultiLabelClassificationHead:
    """Tests for MultiLabelClassificationHead."""

    def test_initialization(self):
        """Test head initialization."""
        head = MultiLabelClassificationHead(input_dim=512, num_labels=5)
        assert head.input_dim == 512
        assert head.num_labels == 5

    def test_forward(self, batch_features):
        """Test forward pass."""
        head = MultiLabelClassificationHead(input_dim=512, num_labels=5)
        logits = head(batch_features)

        assert logits.shape == (8, 5)
        assert not torch.isnan(logits).any()

    def test_independent_classifiers(self, batch_features):
        """Test with independent classifiers."""
        head = MultiLabelClassificationHead(
            input_dim=512,
            num_labels=5,
            use_independent_classifiers=True
        )
        logits = head(batch_features)
        assert logits.shape == (8, 5)

    def test_shared_classifier(self, batch_features):
        """Test with shared classifier."""
        head = MultiLabelClassificationHead(
            input_dim=512,
            num_labels=5,
            use_independent_classifiers=False
        )
        logits = head(batch_features)
        assert logits.shape == (8, 5)


class TestOrdinalClassificationHead:
    """Tests for OrdinalClassificationHead."""

    def test_initialization(self):
        """Test head initialization."""
        head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        assert head.input_dim == 512
        assert head.num_classes == 4
        assert head.num_thresholds == 3

    def test_forward(self, batch_features):
        """Test forward pass."""
        head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        logits = head(batch_features)

        # Should output K-1 thresholds
        assert logits.shape == (8, 3)
        assert not torch.isnan(logits).any()

    def test_predict_probabilities(self, batch_features):
        """Test probability prediction."""
        head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        probs = head.predict_probabilities(batch_features)

        assert probs.shape == (8, 4)

        # Probabilities should sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

        # Probabilities should be non-negative
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_ordinal_property(self):
        """Test ordinal property of predictions."""
        head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        head.eval()

        x = torch.randn(2, 512)

        with torch.no_grad():
            probs = head.predict_probabilities(x)

        # Check that probabilities are valid
        assert probs.shape == (2, 4)
        assert (probs >= 0).all()


class TestAttentionClassificationHead:
    """Tests for AttentionClassificationHead."""

    def test_initialization(self):
        """Test head initialization."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        assert head.input_dim == 512
        assert head.num_classes == 4

    def test_forward_2d(self, batch_features):
        """Test forward pass with 2D input."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        logits = head(batch_features)

        assert logits.shape == (8, 4)
        assert not torch.isnan(logits).any()

    def test_forward_3d(self):
        """Test forward pass with 3D input."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        x = torch.randn(8, 10, 512)  # Multiple feature vectors

        logits = head(x)
        assert logits.shape == (8, 4)

    def test_return_attention_2d(self, batch_features):
        """Test returning attention weights with 2D input."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        logits, attention = head(batch_features, return_attention=True)

        assert logits.shape == (8, 4)
        assert attention.shape == (8, 1)

    def test_return_attention_3d(self):
        """Test returning attention weights with 3D input."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        x = torch.randn(8, 10, 512)

        logits, attention = head(x, return_attention=True)

        assert logits.shape == (8, 4)
        assert attention.shape == (8, 10, 1)

        # Attention should sum to 1
        attention_sum = attention.sum(dim=1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)


class TestEnsembleClassificationHead:
    """Tests for EnsembleClassificationHead."""

    def test_initialization(self):
        """Test head initialization."""
        head = EnsembleClassificationHead(
            input_dim=512,
            num_classes=4,
            num_heads=3
        )
        assert head.input_dim == 512
        assert head.num_classes == 4
        assert head.num_heads == 3
        assert len(head.heads) == 3

    def test_forward(self, batch_features):
        """Test forward pass."""
        head = EnsembleClassificationHead(
            input_dim=512,
            num_classes=4,
            num_heads=3
        )
        logits = head(batch_features)

        assert logits.shape == (8, 4)
        assert not torch.isnan(logits).any()

    @pytest.mark.parametrize("aggregation", ["mean", "max", "vote"])
    def test_aggregation_methods(self, aggregation, batch_features):
        """Test different aggregation methods."""
        head = EnsembleClassificationHead(
            input_dim=512,
            num_classes=4,
            num_heads=3,
            aggregation=aggregation
        )
        logits = head(batch_features)
        assert logits.shape == (8, 4)

    def test_return_individual(self, batch_features):
        """Test returning individual predictions."""
        head = EnsembleClassificationHead(
            input_dim=512,
            num_classes=4,
            num_heads=3
        )
        logits, individual = head(batch_features, return_individual=True)

        assert logits.shape == (8, 4)
        assert individual.shape == (8, 3, 4)

    def test_invalid_aggregation(self):
        """Test that invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            head = EnsembleClassificationHead(
                input_dim=512,
                num_classes=4,
                aggregation="invalid"
            )
            head(torch.randn(8, 512))


class TestHeadsEdgeCases:
    """Test edge cases for classification heads."""

    def test_single_sample(self):
        """Test with single sample."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        x = torch.randn(1, 512)
        logits = head(x)
        assert logits.shape == (1, 4)

    def test_large_batch(self):
        """Test with large batch."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        x = torch.randn(128, 512)
        logits = head(x)
        assert logits.shape == (128, 4)

    def test_binary_classification(self):
        """Test binary classification."""
        head = ClassificationHead(input_dim=512, num_classes=2)
        x = torch.randn(8, 512)
        logits = head(x)
        assert logits.shape == (8, 2)

    def test_many_classes(self):
        """Test with many classes."""
        head = ClassificationHead(input_dim=512, num_classes=100)
        x = torch.randn(8, 512)
        logits = head(x)
        assert logits.shape == (8, 100)

    def test_eval_mode(self):
        """Test in evaluation mode."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        head.eval()

        x = torch.randn(8, 512)
        with torch.no_grad():
            logits = head(x)

        assert logits.shape == (8, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
