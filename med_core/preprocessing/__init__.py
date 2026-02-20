"""
Medical image preprocessing module.

Provides specialized preprocessing pipelines for medical imaging:
- Image normalization (intensity, contrast)
- Region of interest (ROI) extraction
- Artifact removal and quality enhancement
- DICOM/NIfTI support utilities
"""

from med_core.preprocessing.image import (
    ImagePreprocessor,
    apply_clahe,
    crop_center,
    normalize_intensity,
)
from med_core.preprocessing.quality import (
    QualityMetrics,
    assess_image_quality,
    detect_artifacts,
)

__all__ = [
    "ImagePreprocessor",
    "normalize_intensity",
    "crop_center",
    "apply_clahe",
    "assess_image_quality",
    "detect_artifacts",
    "QualityMetrics",
]
