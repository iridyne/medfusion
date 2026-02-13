"""
Task-specific heads for medical imaging models.
"""

from .classification import (
    AttentionClassificationHead,
    ClassificationHead,
    EnsembleClassificationHead,
    MultiLabelClassificationHead,
    OrdinalClassificationHead,
)
from .survival import (
    CoxSurvivalHead,
    DeepSurvivalHead,
    DiscreteTimeSurvivalHead,
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
