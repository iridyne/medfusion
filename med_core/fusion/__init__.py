"""
Fusion modules for multimodal learning.

Provides various strategies for combining features from different modalities:
- Concatenation fusion
- Gated fusion
- Attention-based fusion
- Cross-attention fusion
- Bilinear fusion
- Kronecker product fusion
- Fused attention fusion
- Self-attention fusion
"""

from med_core.fusion.base import (
    BaseFusion,
    MultiModalFusionModel,
    create_fusion_model,
)
from med_core.fusion.fused_attention import (
    CrossModalAttention,
    FusedAttentionFusion,
    MultimodalFusedAttention,
)
from med_core.fusion.kronecker import (
    CompactKroneckerFusion,
    KroneckerFusion,
    MultimodalKroneckerFusion,
)
from med_core.fusion.multiview_model import (
    MultiViewMultiModalFusionModel,
    create_multiview_fusion_model,
)
from med_core.fusion.self_attention import (
    AdditiveAttentionFusion,
    BilinearAttentionFusion,
    GatedAttentionFusion,
    MultimodalSelfAttentionFusion,
    SelfAttentionFusion,
)
from med_core.fusion.strategies import (
    AttentionFusion,
    BilinearFusion,
    ConcatenateFusion,
    CrossAttentionFusion,
    GatedFusion,
    create_fusion_module,
)

__all__ = [
    # Base classes
    "BaseFusion",
    "MultiModalFusionModel",
    "create_fusion_model",
    "MultiViewMultiModalFusionModel",
    "create_multiview_fusion_model",
    # Basic fusion strategies
    "ConcatenateFusion",
    "GatedFusion",
    "AttentionFusion",
    "CrossAttentionFusion",
    "BilinearFusion",
    "create_fusion_module",
    # Kronecker fusion
    "KroneckerFusion",
    "CompactKroneckerFusion",
    "MultimodalKroneckerFusion",
    # Fused attention fusion
    "FusedAttentionFusion",
    "CrossModalAttention",
    "MultimodalFusedAttention",
    # Self-attention fusion
    "SelfAttentionFusion",
    "AdditiveAttentionFusion",
    "BilinearAttentionFusion",
    "GatedAttentionFusion",
    "MultimodalSelfAttentionFusion",
]
