"""
Tests for evaluation metrics and visualization.

Tests cover:
- Binary classification metrics calculation
- Multiclass metrics calculation
- Confidence interval computation
- Visualization functions
"""

import unittest

import numpy as np

from med_core.evaluation import (
    calculate_binary_metrics,
)


class TestBinaryMetrics(unittest.TestCase):
    """Test binary classification metrics."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 100

        # Perfect predictions
        self.y_true_perfect = np.array([0] * 50 + [1] * 50)
        self.y_pred_perfect = self.y_true_perfect.copy()
        self.y_prob_perfect = self.y_true_perfect.astype(float)

        # Random predictions
        self.y_true_random = np.random.randint(0, 2, self.n_samples)
        self.y_pred_random = np.random.randint(0, 2, self.n_samples)
        self.y_prob_random = np.random.rand(self.n_samples)

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        metrics = calculate_binary_metrics(
            self.y_true_perfect,
            self.y_pred_perfect,
            self.y_prob_perfect
        )
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.auc, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.sensitivity, 1.0)
        self.assertEqual(metrics.specificity, 1.0)

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        metrics = calculate_binary_metrics(
            self.y_true_random,
            self.y_pred_random,
            self.y_prob_random
        )
        self.assertIsInstance(metrics.accuracy, float)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertIsInstance(metrics.auc, float)

    def test_imbalanced_data(self):
        """Test metrics with imbalanced classes."""
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 85 + [1] * 15)
        y_prob = np.concatenate([np.random.rand(90) * 0.3, np.random.rand(10) * 0.7 + 0.3])

        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
        self.assertIsInstance(metrics.accuracy, float)
        self.assertIsInstance(metrics.precision, float)
        self.assertIsInstance(metrics.recall, float)

    def test_metrics_summary(self):
        """Test metrics summary string."""
        metrics = calculate_binary_metrics(
            self.y_true_perfect,
            self.y_pred_perfect,
            self.y_prob_perfect
        )
        summary = metrics.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Accuracy", summary)
