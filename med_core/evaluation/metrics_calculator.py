"""
Metrics calculation for evaluation reports.

Provides a dedicated class for calculating and formatting metrics
from evaluation results.
"""

from typing import Any


class MetricsCalculator:
    """
    Calculates and formats evaluation metrics.

    Separates metrics calculation logic from report generation
    to improve modularity and testability.

    Example:
        >>> calculator = MetricsCalculator()
        >>> formatted = calculator.format_binary_metrics(metrics)
        >>> formatted = calculator.format_multiclass_metrics(metrics_dict)
    """

    def format_binary_metrics(self, metrics: Any) -> dict[str, Any]:
        """
        Format binary classification metrics for display.

        Args:
            metrics: Binary metrics object with attributes like auc_roc, accuracy, etc.

        Returns:
            Dictionary with formatted metrics
        """
        def fmt_ci(ci):
            return f"({ci[0]:.4f}, {ci[1]:.4f})" if ci else "-"

        return {
            "performance": {
                "AUC-ROC": {
                    "value": f"{metrics.auc_roc:.4f}",
                    "ci": fmt_ci(metrics.ci_auc_roc),
                    "highlight": True,
                },
                "Accuracy": {
                    "value": f"{metrics.accuracy:.4f}",
                    "ci": fmt_ci(metrics.ci_accuracy),
                },
                "F1 Score": {
                    "value": f"{metrics.f1:.4f}",
                    "ci": "-",
                },
                "Sensitivity": {
                    "value": f"{metrics.sensitivity:.4f}",
                    "ci": fmt_ci(metrics.ci_sensitivity),
                },
                "Specificity": {
                    "value": f"{metrics.specificity:.4f}",
                    "ci": fmt_ci(metrics.ci_specificity),
                },
                "Precision (PPV)": {
                    "value": f"{metrics.ppv:.4f}",
                    "ci": "-",
                },
                "NPV": {
                    "value": f"{metrics.npv:.4f}",
                    "ci": "-",
                },
            },
            "confusion_matrix": {
                "true_negatives": metrics.true_negatives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "true_positives": metrics.true_positives,
            },
        }

    def format_multiclass_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Format multiclass classification metrics for display.

        Args:
            metrics: Dictionary containing multiclass metrics

        Returns:
            Dictionary with formatted metrics
        """
        formatted = {
            "overall": {
                "Accuracy": {
                    "value": f"{metrics.get('accuracy', 0.0):.4f}",
                    "highlight": True,
                },
                "Macro F1": {
                    "value": f"{metrics.get('macro_f1', 0.0):.4f}",
                },
                "Weighted F1": {
                    "value": f"{metrics.get('weighted_f1', 0.0):.4f}",
                },
            }
        }

        # Add per-class metrics if available
        if "per_class_f1" in metrics:
            per_class = {}
            for cls in sorted(metrics["per_class_f1"].keys()):
                per_class[cls] = {
                    "precision": f"{metrics['per_class_precision'][cls]:.4f}",
                    "recall": f"{metrics['per_class_recall'][cls]:.4f}",
                    "f1": f"{metrics['per_class_f1'][cls]:.4f}",
                    "support": metrics["per_class_support"][cls],
                }
            formatted["per_class"] = per_class

        return formatted
