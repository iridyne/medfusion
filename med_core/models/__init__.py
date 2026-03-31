"""
Medical imaging models.

This module provides:
- Generic multi-modal model builder for flexible model construction
- Native three-phase CT fusion model for case-level classification
- Factory functions for common model configurations
"""

from .builder import (
    GenericMultiModalModel,
    MultiModalModelBuilder,
    build_model_from_config,
)
from .three_phase_ct_fusion import ThreePhaseCTFusionModel

__all__ = [
    # Generic model builder
    "GenericMultiModalModel",
    "MultiModalModelBuilder",
    "build_model_from_config",
    # Native task models
    "ThreePhaseCTFusionModel",
]
