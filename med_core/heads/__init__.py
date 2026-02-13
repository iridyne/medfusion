"""
Task-specific heads for medical imaging models.
"""

from .classification import (
    ClassificationHead,
    MultiLabelClassificationHead,
    OrdinalClassificationHead,
    AttentionClassificationHead,
    EnsembleClassificationHead,
)
from .survival import (
    CoxSurvivalHead,
    DiscreteTimeSurvivalHead,
    DeepSurvivalHead,
    MultiTaskSurvivalHead,
    RankingSurvivalHead,
)

__all__ = [
    # Classification heads
    "ClassificationHead",
    "MultiLabelClassificationHead",
    "OrdinalClassificationHead",
    "AttentionClassificationHead",
    "EnsembleClassificationHead",
    # Survival heads
    "CoxSurvivalHead",
    "DiscreteTimeSurvivalHead",
    "DeepSurvivalHead",
    "MultiTaskSurvivalHead",
    "RankingSurvivalHead",
]
