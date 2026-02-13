"""
Unit tests for Classification Heads.
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


class TestClassificationHead:
    """Test suite for ClassificationHead."""

    def test_basic_classification(self):
        """Test basic classification head."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 4)

    def test_with_hidden_layers(self):
        """Test with hidden layers."""
        head = ClassificationHead(input_dim=512, num_classes=4, hidden_dims=[256, 128])
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 4)

    def test_with_batch_norm(self):
        """Test with batch normalization."""
        head = ClassificationHead(
            input_dim=512, num_classes=4, hidden_dims=[256], use_batch_norm=True
        )
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 4)

    def test_different_activations(self):
        """Test with different activation functions."""
        for activation in ["relu", "gelu", "leaky_relu"]:
            head = ClassificationHead(
                input_dim=512, num_classes=4, activation=activation
            )
            features = torch.randn(8, 512)

            logits = head(features)

            assert logits.shape == (8, 4)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = ClassificationHead(input_dim=128, num_classes=4)
        features = torch.randn(4, 128, requires_grad=True)

        logits = head(features)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        head = ClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(1, 512)

        logits = head(features)

        assert logits.shape == (1, 4)


class TestMultiLabelClassificationHead:
    """Test suite for MultiLabelClassificationHead."""

    def test_basic_multilabel(self):
        """Test basic multi-label classification."""
        head = MultiLabelClassificationHead(input_dim=512, num_labels=5)
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 5)

    def test_independent_classifiers(self):
        """Test with independent classifiers."""
        head = MultiLabelClassificationHead(
            input_dim=512, num_labels=5, use_independent_classifiers=True
        )
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 5)

    def test_with_hidden_layers(self):
        """Test with hidden layers."""
        head = MultiLabelClassificationHead(
            input_dim=512, num_labels=5, hidden_dims=[256, 128]
        )
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 5)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = MultiLabelClassificationHead(input_dim=128, num_labels=5)
        features = torch.randn(4, 128, requires_grad=True)

        logits = head(features)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None


class TestOrdinalClassificationHead:
    """Test suite for OrdinalClassificationHead."""

    def test_basic_ordinal(self):
        """Test basic ordinal classification."""
        head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(8, 512)

        logits = head(features)

        # K classes -> K-1 thresholds
        assert logits.shape == (8, 3)

    def test_predict_probabilities(self):
        """Test probability prediction."""
        head = OrdinalClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(8, 512)

        probs = head.predict_probabilities(features)

        assert probs.shape == (8, 4)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_different_num_classes(self):
        """Test with different number of classes."""
        for num_classes in [3, 5, 7]:
            head = OrdinalClassificationHead(input_dim=512, num_classes=num_classes)
            features = torch.randn(4, 512)

            logits = head(features)
            probs = head.predict_probabilities(features)

            assert logits.shape == (4, num_classes - 1)
            assert probs.shape == (4, num_classes)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = OrdinalClassificationHead(input_dim=128, num_classes=4)
        features = torch.randn(4, 128, requires_grad=True)

        logits = head(features)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None


class TestAttentionClassificationHead:
    """Test suite for AttentionClassificationHead."""

    def test_basic_attention(self):
        """Test basic attention classification."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 4)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(8, 512)

        logits, attention_weights = head(features, return_attention=True)

        assert logits.shape == (8, 4)
        assert attention_weights is not None
        assert attention_weights.shape[0] == 8

    def test_multiple_features(self):
        """Test with multiple feature vectors."""
        head = AttentionClassificationHead(input_dim=512, num_classes=4)
        features = torch.randn(8, 10, 512)  # 10 feature vectors per sample

        logits, attention_weights = head(features, return_attention=True)

        assert logits.shape == (8, 4)
        assert attention_weights.shape == (8, 10, 1)
        # Attention weights should sum to 1
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(8, 1), atol=1e-5)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = AttentionClassificationHead(input_dim=128, num_classes=4)
        features = torch.randn(4, 128, requires_grad=True)

        logits = head(features)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None


class TestEnsembleClassificationHead:
    """Test suite for EnsembleClassificationHead."""

    def test_basic_ensemble(self):
        """Test basic ensemble classification."""
        head = EnsembleClassificationHead(input_dim=512, num_classes=4, num_heads=3)
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 4)

    def test_return_individual_predictions(self):
        """Test returning individual head predictions."""
        head = EnsembleClassificationHead(input_dim=512, num_classes=4, num_heads=3)
        features = torch.randn(8, 512)

        logits, individual = head(features, return_individual=True)

        assert logits.shape == (8, 4)
        assert individual.shape == (8, 3, 4)  # [B, num_heads, num_classes]

    def test_different_aggregations(self):
        """Test different aggregation methods."""
        for aggregation in ["mean", "max", "vote"]:
            head = EnsembleClassificationHead(
                input_dim=512, num_classes=4, num_heads=3, aggregation=aggregation
            )
            features = torch.randn(8, 512)

            logits = head(features)

            assert logits.shape == (8, 4)

    def test_different_num_heads(self):
        """Test with different number of heads."""
        for num_heads in [2, 5, 7]:
            head = EnsembleClassificationHead(
                input_dim=512, num_classes=4, num_heads=num_heads
            )
            features = torch.randn(4, 512)

            logits = head(features)

            assert logits.shape == (4, 4)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = EnsembleClassificationHead(input_dim=128, num_classes=4, num_heads=3)
        features = torch.randn(4, 128, requires_grad=True)

        logits = head(features)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None

    def test_invalid_aggregation(self):
        """Test error handling for invalid aggregation."""
        head = EnsembleClassificationHead(
            input_dim=512, num_classes=4, num_heads=3, aggregation="invalid"
        )
        features = torch.randn(8, 512)

        with pytest.raises(ValueError):
            head(features)


class TestIntegration:
    """Integration tests for classification heads."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        head = ClassificationHead(input_dim=512, num_classes=4).cuda()
        features = torch.randn(8, 512).cuda()

        logits = head(features)

        assert logits.shape == (8, 4)
        assert logits.device.type == "cuda"

    def test_binary_classification(self):
        """Test binary classification (2 classes)."""
        head = ClassificationHead(input_dim=512, num_classes=2)
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 2)

    def test_large_num_classes(self):
        """Test with large number of classes."""
        head = ClassificationHead(input_dim=512, num_classes=100)
        features = torch.randn(8, 512)

        logits = head(features)

        assert logits.shape == (8, 100)

    def test_dropout_training_vs_eval(self):
        """Test dropout behavior in training vs eval mode."""
        head = ClassificationHead(input_dim=512, num_classes=4, dropout=0.5)
        features = torch.randn(8, 512)

        # Training mode
        head.train()
        _logits_train1 = head(features)
        _logits_train2 = head(features)

        # Eval mode
        head.eval()
        logits_eval1 = head(features)
        logits_eval2 = head(features)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(logits_eval1, logits_eval2)
