"""
Evaluation module for medical multimodal models.

Provides comprehensive evaluation tools for medical research:
- Classification metrics (accuracy, AUC, F1, sensitivity, specificity)
- Visualization (ROC curves, PR curves, confusion matrices)
- Interpretability (Grad-CAM, attention visualization)
- Report generation for publications
"""

from med_core.evaluation.interpretability import (
    GradCAM,
    visualize_attention_weights,
    visualize_gradcam,
)
from med_core.evaluation.metrics import calculate_binary_metrics
from med_core.evaluation.metrics_calculator import MetricsCalculator
from med_core.evaluation.report import (
    EvaluationReport,
    generate_evaluation_report,
)
from med_core.evaluation.report_generator import ReportGenerator
from med_core.evaluation.report_visualizer import ReportVisualizer
from med_core.evaluation.visualization import plot_confusion_matrix, plot_roc_curve

__all__ = [
    # Metrics
    "calculate_binary_metrics",
    "MetricsCalculator",
    # Visualization
    "plot_confusion_matrix",
    "plot_roc_curve",
    "ReportVisualizer",
    # Interpretability
    "GradCAM",
    "visualize_gradcam",
    "visualize_attention_weights",
    # Report generation
    "EvaluationReport",  # Legacy alias for backward compatibility
    "ReportGenerator",
    "generate_evaluation_report",
]
