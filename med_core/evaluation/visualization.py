"""
Visualization module for evaluation plots.

This module wraps the shared visualization utilities for consistency across projects.
"""

from med_core.shared.visualization import plot_confusion_matrix, plot_roc_curve

__all__ = [
    "plot_roc_curve",
    "plot_confusion_matrix",
]
