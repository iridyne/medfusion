"""
Medical ML Visualization Module
================================
Publication-ready plots for model evaluation and analysis.
"""

from .font_utils import configure_matplotlib_fonts
from .heatmaps import (
    build_rendering_metadata,
    compute_gradcam_volume,
    map_slice_index_between_depths,
    prepare_overlay_image,
    render_overlay_artifact,
    resize_heatmap_slice,
    select_representative_slice,
)
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
    "build_rendering_metadata",
    "compute_gradcam_volume",
    "map_slice_index_between_depths",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_probability_distribution",
    "plot_roc_curve",
    "prepare_overlay_image",
    "render_overlay_artifact",
    "resize_heatmap_slice",
    "select_representative_slice",
    "plot_training_curves",
]
