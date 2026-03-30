"""
Medical imaging models.

This module provides:
- Generic multi-modal model builder for flexible model construction
- SMuRF models for radiology-pathology fusion
- Factory functions for common model configurations
"""

from .builder import (
    GenericMultiModalModel,
    MultiModalModelBuilder,
    build_model_from_config,
)
from .smurf import (
    SMuRFModel,
    SMuRFWithMIL,
    smurf_base,
    smurf_small,
    smurf_with_mil_small,
)
from .three_phase_ct_fusion import ThreePhaseCTFusionModel

__all__ = [
    # Generic model builder
    "GenericMultiModalModel",
    "MultiModalModelBuilder",
    "build_model_from_config",
    # SMuRF models
    "SMuRFModel",
    "SMuRFWithMIL",
    "ThreePhaseCTFusionModel",
    "smurf_small",
    "smurf_base",
    "smurf_with_mil_small",
]
