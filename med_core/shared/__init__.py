"""
Shared utilities for medical ML projects.

This module provides common utilities that can be reused across different
medical machine learning projects:

- data_utils: DICOM loading, image preprocessing, tabular preprocessing
- model_utils: Training utilities, metrics calculation
- visualization: Plotting functions for evaluation
"""

# Note: Submodules are imported on-demand to avoid circular dependencies
# and to allow projects to use only what they need.

__all__ = [
    "data_utils",
    "model_utils",
    "visualization",
]
