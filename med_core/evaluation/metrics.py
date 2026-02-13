"""
Metrics calculation module for medical classification tasks.

This module wraps the shared metrics utilities for consistency across projects.
"""

from med_core.shared.model_utils import calculate_binary_metrics

__all__ = [
    "calculate_binary_metrics",
]
