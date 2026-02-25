"""
Medical ML Metrics Calculation
==============================
Comprehensive metrics for binary and multi-class classification tasks.
"""

import logging
from collections.abc import ItemsView, KeysView, ValuesView
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Provide a lightweight alias for array-like types to help static checkers while
# keeping runtime imports unchanged. This reduces noise from unresolved third-party
# types in environments where stubs aren't available.
if TYPE_CHECKING:
    from numpy import ndarray as NPArray  # type: ignore
else:
    NPArray = object  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class BinaryMetrics:
    """Binary classification metrics container."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    # Backwards-compatible alias used by some callers/tests
    auc: float = 0.0

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    num_samples: int = 0
    optimal_threshold: float = 0.5

    # Confidence intervals (95% CI) - for report generation
    ci_auc_roc: tuple[float, float] | None = None
    ci_accuracy: tuple[float, float] | None = None
    ci_sensitivity: tuple[float, float] | None = None
    ci_specificity: tuple[float, float] | None = None

    # Aliases for backward compatibility with report generator
    @property
    def true_positives(self) -> int:
        return self.tp

    @property
    def true_negatives(self) -> int:
        return self.tn

    @property
    def false_positives(self) -> int:
        return self.fp

    @property
    def false_negatives(self) -> int:
        return self.fn

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "auc": self.auc,
            "true_positives": self.tp,
            "true_negatives": self.tn,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "num_samples": self.num_samples,
            "optimal_threshold": self.optimal_threshold,
        }

    def __getitem__(self, key: str) -> object:
        """
        Allow dict-like access to metrics, e.g. metrics['accuracy'].
        This keeps backward compatibility with tests that index into the result.
        """
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like behavior."""
        return key in self.to_dict()

    def keys(self) -> KeysView[str]:
        """Return dict keys for dict-like behavior."""
        return self.to_dict().keys()

    def values(self) -> ValuesView[float]:
        """Return dict values for dict-like behavior."""
        return self.to_dict().values()

    def items(self) -> ItemsView[str, float]:
        """Return dict items for dict-like behavior."""
        return self.to_dict().items()

    def summary(self) -> str:
        """Return formatted summary."""
        return (
            f"Accuracy: {self.accuracy:.4f} | AUC: {self.auc_roc:.4f} | "
            f"F1: {self.f1:.4f} | Sens: {self.sensitivity:.4f} | "
            f"Spec: {self.specificity:.4f}"
        )


def calculate_binary_metrics(
    y_true: "NPArray",
    y_pred: "NPArray",
    y_prob: "NPArray | None" = None,
    find_optimal_threshold: bool = True,
) -> BinaryMetrics:
    """
    Calculate comprehensive binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for positive class (optional)
        find_optimal_threshold: Whether to find optimal threshold using Youden's J

    Returns:
        BinaryMetrics object
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()

    metrics = BinaryMetrics()
    metrics.num_samples = len(y_true)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    except Exception as e:
        logger.warning(f"Error computing confusion matrix: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics.tp = int(tp)
    metrics.tn = int(tn)
    metrics.fp = int(fp)
    metrics.fn = int(fn)

    # Basic metrics
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.precision = precision_score(y_true, y_pred, zero_division=0.0)
    metrics.recall = recall_score(y_true, y_pred, zero_division=0.0)
    metrics.f1 = f1_score(y_true, y_pred, zero_division=0.0)

    # Medical metrics
    metrics.sensitivity = metrics.recall
    metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics.ppv = metrics.precision
    metrics.npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # AUC metrics
    if y_prob is not None and len(np.unique(y_true)) >= 2:
        try:
            metrics.auc_roc = roc_auc_score(y_true, y_prob)

            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            metrics.auc_pr = auc(recall_curve, precision_curve)

            # Backwards-compatible alias expected by some callers/tests
            metrics.auc = float(metrics.auc_roc)

            if find_optimal_threshold:
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                metrics.optimal_threshold = float(thresholds[best_idx])
        except Exception as e:
            logger.warning(f"Error computing AUC: {e}")
            metrics.auc_roc = float("nan")
            metrics.auc_pr = float("nan")
            metrics.auc = float("nan")

    return metrics


def calculate_multiclass_metrics(
    y_true: "NPArray",
    y_pred: "NPArray",
) -> dict[str, object]:
    """
    Calculate multi-class classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0.0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0.0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0.0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0.0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "num_samples": len(y_true),
    }


def calculate_confidence_intervals(
    y_true: "NPArray",
    y_prob: "NPArray",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Calculate confidence intervals using bootstrapping.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed

    Returns:
        Dictionary mapping metric names to (lower, upper) CI tuples
    """
    np.random.seed(random_seed)

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    n_samples = len(y_true)
    alpha = 1 - confidence_level

    auc_scores = []
    accuracy_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= 0.5).astype(int)

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            auc_scores.append(roc_auc_score(y_true_boot, y_prob_boot))
            accuracy_scores.append(accuracy_score(y_true_boot, y_pred_boot))
        except Exception as e:
            logger.debug(f"Bootstrap iteration failed: {e}")
            continue

    ci_results = {}

    if auc_scores:
        ci_results["auc_roc"] = (
            float(np.percentile(auc_scores, alpha / 2 * 100)),
            float(np.percentile(auc_scores, (1 - alpha / 2) * 100)),
        )

    if accuracy_scores:
        ci_results["accuracy"] = (
            float(np.percentile(accuracy_scores, alpha / 2 * 100)),
            float(np.percentile(accuracy_scores, (1 - alpha / 2) * 100)),
        )

    return ci_results
