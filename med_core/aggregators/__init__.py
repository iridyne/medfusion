"""
Aggregators for multiple instance learning and multi-view fusion.
"""

from .mil import (
    MeanPoolingAggregator,
    MaxPoolingAggregator,
    AttentionAggregator,
    GatedAttentionAggregator,
    DeepSetsAggregator,
    TransformerAggregator,
    MILAggregator,
)

__all__ = [
    "MeanPoolingAggregator",
    "MaxPoolingAggregator",
    "AttentionAggregator",
    "GatedAttentionAggregator",
    "DeepSetsAggregator",
    "TransformerAggregator",
    "MILAggregator",
]
