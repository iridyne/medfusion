"""
Medical ML Visualization Module
================================
Publication-ready plots for model evaluation and analysis.
"""

from .font_utils import configure_matplotlib_fonts
from .plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_probability_distribution,
    plot_roc_curve,
    plot_training_curves,
)

__all__ = [
    "configure_matplotlib_fonts",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_probability_distribution",
    "plot_roc_curve",
    "plot_training_curves",
]
