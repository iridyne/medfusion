"""
注意力监督模块

提供离线训练阶段的注意力引导功能，无需人工实时标注。

支持的方法：
1. 分割掩码监督（Segmentation Mask Supervision）
2. CAM自监督（CAM-based Self-Supervision）
3. 多实例学习（Multiple Instance Learning）
4. SE/ECA/Transformer 注意力（v0.2.0 新增）
"""

from med_core.attention_supervision.advanced_attention import (
    CBAM,
    ECAAttention,
    MultiHeadSelfAttention,
    SEAttention,
    SpatialAttention,
    TransformerAttention2D,
    create_attention_module,
)
from med_core.attention_supervision.advanced_supervision import (
    ChannelAttentionSupervision,
    HybridAttentionSupervision,
    SpatialAttentionSupervision,
    TransformerAttentionSupervision,
    create_attention_supervision,
)
from med_core.attention_supervision.base import (
    AttentionLoss,
    BaseAttentionSupervision,
)
from med_core.attention_supervision.cam_supervision import (
    CAMSelfSupervision,
    generate_cam,
)
from med_core.attention_supervision.mask_supervision import (
    MaskSupervisedAttention,
    mask_to_attention_target,
)
from med_core.attention_supervision.mil_supervision import (
    MultiInstanceLearning,
    extract_patches,
)

__all__ = [
    # 基类
    "BaseAttentionSupervision",
    "AttentionLoss",
    # 分割掩码监督
    "MaskSupervisedAttention",
    "mask_to_attention_target",
    # CAM自监督
    "CAMSelfSupervision",
    "generate_cam",
    # 多实例学习
    "MultiInstanceLearning",
    "extract_patches",
    # 高级注意力模块 (v0.2.0)
    "SEAttention",
    "ECAAttention",
    "SpatialAttention",
    "CBAM",
    "MultiHeadSelfAttention",
    "TransformerAttention2D",
    "create_attention_module",
    # 高级注意力监督 (v0.2.0)
    "ChannelAttentionSupervision",
    "SpatialAttentionSupervision",
    "TransformerAttentionSupervision",
    "HybridAttentionSupervision",
    "create_attention_supervision",
]
