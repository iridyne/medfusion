"""
Pluggable backbone modules.

Supports:
- Vision backbones (ResNet, MobileNet, EfficientNet, EfficientNetV2, ConvNeXt, MaxViT, RegNet, ViT, Swin)
- Tabular MLP modules
- Attention mechanisms (CBAM, SE, ECA)
"""

from med_core.backbones.attention import (
    CBAM,
    ECABlock,
    SEBlock,
    create_attention_module,
)
from med_core.backbones.base import (
    BaseBackbone,
    BaseTabularBackbone,
    BaseVisionBackbone,
)
from med_core.backbones.multiview_vision import (
    MultiViewVisionBackbone,
    create_multiview_vision_backbone,
)
from med_core.backbones.tabular import AdaptiveMLP, create_tabular_backbone
from med_core.backbones.view_aggregator import (
    AttentionAggregator,
    BaseViewAggregator,
    CrossViewAttentionAggregator,
    LearnedWeightAggregator,
    MaxPoolAggregator,
    MeanPoolAggregator,
    create_view_aggregator,
)
from med_core.backbones.vision import (
    ConvNeXtBackbone,
    EfficientNetBackbone,
    EfficientNetV2Backbone,
    MaxViTBackbone,
    MobileNetBackbone,
    RegNetBackbone,
    ResNetBackbone,
    SwinBackbone,
    ViTBackbone,
    create_vision_backbone,
)

__all__ = [
    # Base classes
    "BaseBackbone",
    "BaseVisionBackbone",
    "BaseTabularBackbone",
    # Vision backbones
    "ResNetBackbone",
    "MobileNetBackbone",
    "EfficientNetBackbone",
    "EfficientNetV2Backbone",
    "ConvNeXtBackbone",
    "MaxViTBackbone",
    "RegNetBackbone",
    "ViTBackbone",
    "SwinBackbone",
    "create_vision_backbone",
    # Multi-view support
    "MultiViewVisionBackbone",
    "create_multiview_vision_backbone",
    # View aggregators
    "BaseViewAggregator",
    "MaxPoolAggregator",
    "MeanPoolAggregator",
    "AttentionAggregator",
    "CrossViewAttentionAggregator",
    "LearnedWeightAggregator",
    "create_view_aggregator",
    # Tabular backbones
    "AdaptiveMLP",
    "create_tabular_backbone",
    # Attention modules
    "CBAM",
    "SEBlock",
    "ECABlock",
    "create_attention_module",
]
