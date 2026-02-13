"""
Aggregators for multiple instance learning and multi-view fusion.
"""

from .mil import (
    AttentionAggregator,
    DeepSetsAggregator,
    GatedAttentionAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    MILAggregator,
    TransformerAggregator,
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
