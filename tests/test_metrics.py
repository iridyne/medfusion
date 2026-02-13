"""
Tests for metrics calculation utilities.
"""

import numpy as np

from med_core.shared.model_utils import calculate_binary_metrics


class TestBinaryMetrics:
    """Test binary classification metrics calculation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert metrics["accuracy"] == 1.0

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1

    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.9, 0.8, 0.9, 0.95])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_all_negative_predictions(self):
        """Test metrics when all predictions are negative."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.05])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "recall" in metrics

    def test_balanced_dataset(self):
        """Test metrics with balanced dataset."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.9, 0.4, 0.85])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc" in metrics

    def test_imbalanced_dataset(self):
        """Test metrics with imbalanced dataset."""
        # 90% negative, 10% positive
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "auc" in metrics

    def test_metrics_without_probabilities(self):
        """Test metrics calculation without probability scores."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob=None)

        assert metrics is not None
        assert "accuracy" in metrics
        # AUC should not be present without probabilities
        # or should handle gracefully

    def test_metrics_types(self):
        """Test that metrics return correct types."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float, np.number))

    def test_edge_case_single_class(self):
        """Test edge case where only one class is present."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.95, 0.85, 0.92])

        # Should handle gracefully or raise appropriate error
        try:
            metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
            assert metrics is not None
        except (ValueError, ZeroDivisionError):
            # Expected for some metrics with single class
            pass

    def test_metrics_range(self):
        """Test that all metrics are within valid ranges."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        # All metrics should be between 0 and 1
        for key, value in metrics.items():
            if key not in ["confusion_matrix"]:  # Skip non-scalar metrics
                assert 0 <= value <= 1, f"{key} = {value} is out of range [0, 1]"


class TestMetricsIntegration:
    """Test metrics calculation integration."""

    def test_metrics_consistency(self):
        """Test that metrics are consistent across multiple calculations."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics1 = calculate_binary_metrics(y_true, y_pred, y_prob)
        metrics2 = calculate_binary_metrics(y_true, y_pred, y_prob)

        # Results should be identical
        for key in metrics1.keys():
            if key not in ["confusion_matrix"]:
                assert metrics1[key] == metrics2[key]

    def test_large_dataset(self):
        """Test metrics calculation on large dataset."""
        np.random.seed(42)
        n_samples = 10000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)

        assert metrics is not None
        assert "accuracy" in metrics
        assert "auc" in metrics
