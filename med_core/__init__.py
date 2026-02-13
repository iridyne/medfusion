"""
Med-Core: A Pluggable Multimodal Medical Research Framework
============================================================

This framework provides:
- Pluggable vision backbones (ResNet, MobileNet, EfficientNet, ViT, Swin)
- Adaptive tabular MLP modules
- Flexible fusion strategies (Concatenate, Gated, Attention, Cross-Attention)
- Medical-specific preprocessing pipelines
- Comprehensive evaluation (ROC, PR curves, Grad-CAM, confusion matrices)
"""

from med_core.version import __version__

__all__ = ["__version__"]
