"""
Vision backbone implementations.

Provides pluggable vision encoders including:
- ResNet family (18, 34, 50, 101)
- MobileNet family (V2, V3)
- EfficientNet family (B0-B2)
- EfficientNetV2 family (S, M, L)
- ConvNeXt family (Tiny, Small, Base, Large)
- MaxViT (Tiny)
- RegNet family (Y-series)
- Vision Transformer (ViT)
- Swin Transformer
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from med_core.backbones.attention import create_attention_module
from med_core.backbones.base import BaseVisionBackbone


class ResNetBackbone(BaseVisionBackbone):
    """
    ResNet-based vision backbone.

    Supports ResNet18, 34, 50, 101 with optional attention mechanisms.
    """

    VARIANTS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
    }

    def __init__(
        self,
        variant: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
        attention_type: str = "cbam",
        use_roi_guidance: bool = False,
        enable_attention_supervision: bool = False,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown ResNet variant: {variant}. Choose from {list(self.VARIANTS.keys())}")

        self.variant = variant
        self.enable_attention_supervision = enable_attention_supervision
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        # Load pretrained model
        backbone_model = model_fn(weights=weights if pretrained else None)

        # Remove classification head, keep feature extractor
        self._backbone = nn.Sequential(*list(backbone_model.children())[:-2])

        # Optional attention module
        # If attention supervision is enabled and using CBAM, configure it to return weights
        return_attention_weights = enable_attention_supervision and attention_type == "cbam"
        self._attention = create_attention_module(
            attention_type, self._backbone_out_dim, use_roi_guidance, return_attention_weights
        )

        # Global pooling
        self._pool = nn.AdaptiveAvgPool2d(1)

        # Projection head
        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        # Apply freezing if requested
        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through the backbone.

        Args:
            x: Input images (B, C, H, W)
            return_intermediates: If True, return intermediate features for attention supervision

        Returns:
            If return_intermediates=False:
                Output features (B, feature_dim)
            If return_intermediates=True:
                Dictionary containing:
                    - "features": Output features (B, feature_dim)
                    - "feature_maps": Feature maps before pooling (B, C, H, W)
                    - "attention_weights": Spatial attention weights (B, 1, H, W) or None
        """
        # Extract features
        feature_maps = self.extract_features(x)  # (B, C, H, W)

        # Apply attention
        attention_weights = None
        if self._attention is not None:
            if self.enable_attention_supervision:
                # CBAM returns (features, weights_dict)
                feature_maps, weights_dict = self._attention(feature_maps)
                attention_weights = weights_dict.get("spatial_weights")  # (B, 1, H, W) or None
            else:
                # Normal mode, just apply attention
                feature_maps = self._attention(feature_maps)

        # Global average pooling
        pooled = self._pool(feature_maps)
        pooled = pooled.view(pooled.size(0), -1)

        # Project to target dimension
        features = self._projection(pooled)

        # Return based on request
        if return_intermediates:
            return {
                "features": features,
                "feature_maps": feature_maps,
                "attention_weights": attention_weights,
            }
        else:
            return features


class MobileNetBackbone(BaseVisionBackbone):
    """
    MobileNet-based vision backbone.

    Lightweight backbone suitable for resource-constrained scenarios.
    """

    VARIANTS = {
        "mobilenetv2": (
            models.mobilenet_v2,
            models.MobileNet_V2_Weights.IMAGENET1K_V2,
            1280,
        ),
        "mobilenetv3_small": (
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            576,
        ),
        "mobilenetv3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
            960,
        ),
    }

    def __init__(
        self,
        variant: Literal["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"] = "mobilenetv2",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
        attention_type: str = "cbam",
        use_roi_guidance: bool = False,
        enable_attention_supervision: bool = False,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown MobileNet variant: {variant}")

        self.variant = variant
        self.enable_attention_supervision = enable_attention_supervision
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        # Load pretrained model
        backbone_model = model_fn(weights=weights if pretrained else None)

        # Extract feature layers
        self._backbone = backbone_model.features

        # Optional attention
        return_attention_weights = enable_attention_supervision and attention_type == "cbam"
        self._attention = create_attention_module(
            attention_type, self._backbone_out_dim, use_roi_guidance, return_attention_weights
        )

        # Pooling and projection
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)


class EfficientNetBackbone(BaseVisionBackbone):
    """
    EfficientNet-based vision backbone.

    Provides good accuracy-efficiency trade-off.
    """

    VARIANTS = {
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            1280,
        ),
        "efficientnet_b1": (
            models.efficientnet_b1,
            models.EfficientNet_B1_Weights.IMAGENET1K_V2,
            1280,
        ),
        "efficientnet_b2": (
            models.efficientnet_b2,
            models.EfficientNet_B2_Weights.IMAGENET1K_V1,
            1408,
        ),
    }

    def __init__(
        self,
        variant: Literal["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"] = "efficientnet_b0",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
        attention_type: str = "cbam",
        use_roi_guidance: bool = False,
        enable_attention_supervision: bool = False,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        self.variant = variant
        self.enable_attention_supervision = enable_attention_supervision
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        backbone_model = model_fn(weights=weights if pretrained else None)
        self._backbone = backbone_model.features

        return_attention_weights = enable_attention_supervision and attention_type == "cbam"
        self._attention = create_attention_module(
            attention_type, self._backbone_out_dim, use_roi_guidance, return_attention_weights
        )

        self._pool = nn.AdaptiveAvgPool2d(1)
        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)


class ViTBackbone(BaseVisionBackbone):
    """
    Vision Transformer (ViT) backbone.

    Transformer-based architecture for image understanding.
    """

    VARIANTS = {
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, 768),
        "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1, 768),
    }

    def __init__(
        self,
        variant: Literal["vit_b_16", "vit_b_32"] = "vit_b_16",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown ViT variant: {variant}")

        self.variant = variant
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        self._backbone = model_fn(weights=weights if pretrained else None)

        # Replace classification head with identity
        self._backbone.heads = nn.Identity()

        # Projection head
        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT outputs (B, hidden_dim) directly
        features = self.extract_features(x)
        features = self._projection(features)
        return features

    def freeze_backbone(self, strategy: str = "full", unfreeze_last_n: int = 2) -> None:
        """Freeze ViT backbone with transformer-specific handling."""
        if strategy == "none":
            return

        # Freeze all parameters
        for param in self._backbone.parameters():
            param.requires_grad = False

        if strategy == "partial":
            # Unfreeze last n encoder blocks
            encoder_blocks = self._backbone.encoder.layers
            for block in encoder_blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True


class SwinBackbone(BaseVisionBackbone):
    """
    Swin Transformer backbone.

    Hierarchical vision transformer with shifted windows.
    """

    VARIANTS = {
        "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1, 768),
        "swin_s": (models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1, 768),
    }

    def __init__(
        self,
        variant: Literal["swin_t", "swin_s"] = "swin_t",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown Swin variant: {variant}")

        self.variant = variant
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        self._backbone = model_fn(weights=weights if pretrained else None)

        # Replace classification head
        self._backbone.head = nn.Identity()

        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        features = self._projection(features)
        return features


class ConvNeXtBackbone(BaseVisionBackbone):
    """
    ConvNeXt backbone - Modern CNN architecture.

    ConvNeXt modernizes the standard ResNet design by incorporating
    design choices from Vision Transformers, achieving competitive
    performance with pure convolutional networks.

    Paper: "A ConvNet for the 2020s" (https://arxiv.org/abs/2201.03545)
    """

    VARIANTS = {
        "convnext_tiny": (
            models.convnext_tiny,
            models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            768
        ),
        "convnext_small": (
            models.convnext_small,
            models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
            768
        ),
        "convnext_base": (
            models.convnext_base,
            models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
            1024
        ),
        "convnext_large": (
            models.convnext_large,
            models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
            1536
        ),
    }

    def __init__(
        self,
        variant: Literal["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"] = "convnext_tiny",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
        attention_type: str = "none",
        use_roi_guidance: bool = False,
        enable_attention_supervision: bool = False,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}. Choose from {list(self.VARIANTS.keys())}")

        self.variant = variant
        self.enable_attention_supervision = enable_attention_supervision
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        # Load pretrained model
        backbone_model = model_fn(weights=weights if pretrained else None)

        # Remove classification head, keep feature extractor
        self._backbone = nn.Sequential(*list(backbone_model.children())[:-1])

        # Optional attention module (ConvNeXt already has good attention-like mechanisms)
        self._attention = None
        if attention_type != "none":
            return_attention_weights = enable_attention_supervision and attention_type == "cbam"
            self._attention = create_attention_module(
                attention_type, self._backbone_out_dim, use_roi_guidance, return_attention_weights
            )

        # Projection head
        self._projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),  # ConvNeXt uses LayerNorm
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.extract_features(x)

        # Apply optional attention
        if self._attention is not None:
            features = self._attention(features)

        # Project to target dimension
        features = self._projection(features)

        return features


class MaxViTBackbone(BaseVisionBackbone):
    """
    MaxViT backbone - Multi-axis Vision Transformer.

    MaxViT combines local and global spatial interactions through
    multi-axis attention (block attention + grid attention).

    Paper: "MaxViT: Multi-Axis Vision Transformer" (https://arxiv.org/abs/2204.01697)
    """

    VARIANTS = {
        "maxvit_t": (
            models.maxvit_t,
            models.MaxVit_T_Weights.IMAGENET1K_V1,
            512
        ),
    }

    def __init__(
        self,
        variant: Literal["maxvit_t"] = "maxvit_t",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown MaxViT variant: {variant}. Choose from {list(self.VARIANTS.keys())}")

        self.variant = variant
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        # Load pretrained model
        self._backbone = model_fn(weights=weights if pretrained else None)

        # Replace classification head
        self._backbone.classifier = nn.Identity()

        # Projection head
        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MaxViT outputs (B, hidden_dim) directly
        features = self.extract_features(x)
        features = self._projection(features)
        return features


class EfficientNetV2Backbone(BaseVisionBackbone):
    """
    EfficientNetV2 backbone - Improved EfficientNet.

    EfficientNetV2 improves training speed and parameter efficiency
    over the original EfficientNet through Fused-MBConv and progressive learning.

    Paper: "EfficientNetV2: Smaller Models and Faster Training" (https://arxiv.org/abs/2104.00298)
    """

    VARIANTS = {
        "efficientnet_v2_s": (
            models.efficientnet_v2_s,
            models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            1280,
        ),
        "efficientnet_v2_m": (
            models.efficientnet_v2_m,
            models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
            1280,
        ),
        "efficientnet_v2_l": (
            models.efficientnet_v2_l,
            models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
            1280,
        ),
    }

    def __init__(
        self,
        variant: Literal["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"] = "efficientnet_v2_s",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
        attention_type: str = "none",
        use_roi_guidance: bool = False,
        enable_attention_supervision: bool = False,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown EfficientNetV2 variant: {variant}")

        self.variant = variant
        self.enable_attention_supervision = enable_attention_supervision
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        backbone_model = model_fn(weights=weights if pretrained else None)
        self._backbone = backbone_model.features

        # Optional attention (EfficientNetV2 already has SE blocks)
        self._attention = None
        if attention_type != "none":
            return_attention_weights = enable_attention_supervision and attention_type == "cbam"
            self._attention = create_attention_module(
                attention_type, self._backbone_out_dim, use_roi_guidance, return_attention_weights
            )

        self._pool = nn.AdaptiveAvgPool2d(1)
        self._projection = nn.Sequential(
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)


class RegNetBackbone(BaseVisionBackbone):
    """
    RegNet backbone - Design space optimized networks.

    RegNet is discovered through design space exploration, providing
    good performance across different computational budgets.

    Paper: "Designing Network Design Spaces" (https://arxiv.org/abs/2003.13678)
    """

    VARIANTS = {
        "regnet_y_400mf": (
            models.regnet_y_400mf,
            models.RegNet_Y_400MF_Weights.IMAGENET1K_V1,
            440,
        ),
        "regnet_y_800mf": (
            models.regnet_y_800mf,
            models.RegNet_Y_800MF_Weights.IMAGENET1K_V1,
            784,
        ),
        "regnet_y_1_6gf": (
            models.regnet_y_1_6gf,
            models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1,
            888,
        ),
        "regnet_y_3_2gf": (
            models.regnet_y_3_2gf,
            models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1,
            1512,
        ),
        "regnet_y_8gf": (
            models.regnet_y_8gf,
            models.RegNet_Y_8GF_Weights.IMAGENET1K_V1,
            2016,
        ),
        "regnet_y_16gf": (
            models.regnet_y_16gf,
            models.RegNet_Y_16GF_Weights.IMAGENET1K_V1,
            3024,
        ),
        "regnet_y_32gf": (
            models.regnet_y_32gf,
            models.RegNet_Y_32GF_Weights.IMAGENET1K_V1,
            3712,
        ),
    }

    def __init__(
        self,
        variant: Literal[
            "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
            "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf"
        ] = "regnet_y_400mf",
        pretrained: bool = True,
        freeze: bool = False,
        feature_dim: int = 128,
        dropout: float = 0.3,
        attention_type: str = "none",
        use_roi_guidance: bool = False,
        enable_attention_supervision: bool = False,
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown RegNet variant: {variant}. Choose from {list(self.VARIANTS.keys())}")

        self.variant = variant
        self.enable_attention_supervision = enable_attention_supervision
        model_fn, weights, self._backbone_out_dim = self.VARIANTS[variant]

        # Load pretrained model
        backbone_model = model_fn(weights=weights if pretrained else None)

        # Remove classification head, keep feature extractor
        self._backbone = nn.Sequential(*list(backbone_model.children())[:-1])

        # Optional attention module
        self._attention = None
        if attention_type != "none":
            return_attention_weights = enable_attention_supervision and attention_type == "cbam"
            self._attention = create_attention_module(
                attention_type, self._backbone_out_dim, use_roi_guidance, return_attention_weights
            )

        # Global pooling
        self._pool = nn.AdaptiveAvgPool2d(1)

        # Projection head
        self._projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
        )

        self._output_dim = feature_dim

        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.extract_features(x)

        # Apply optional attention
        if self._attention is not None:
            features = self._attention(features)

        # Global average pooling
        features = self._pool(features)

        # Project to target dimension
        features = self._projection(features)

        return features


# Factory function for creating vision backbones
BACKBONE_REGISTRY = {
    # ResNet family
    "resnet18": lambda **kwargs: ResNetBackbone(variant="resnet18", **kwargs),
    "resnet34": lambda **kwargs: ResNetBackbone(variant="resnet34", **kwargs),
    "resnet50": lambda **kwargs: ResNetBackbone(variant="resnet50", **kwargs),
    "resnet101": lambda **kwargs: ResNetBackbone(variant="resnet101", **kwargs),

    # MobileNet family
    "mobilenetv2": lambda **kwargs: MobileNetBackbone(variant="mobilenetv2", **kwargs),
    "mobilenetv3_small": lambda **kwargs: MobileNetBackbone(variant="mobilenetv3_small", **kwargs),
    "mobilenetv3_large": lambda **kwargs: MobileNetBackbone(variant="mobilenetv3_large", **kwargs),

    # EfficientNet family (V1)
    "efficientnet_b0": lambda **kwargs: EfficientNetBackbone(variant="efficientnet_b0", **kwargs),
    "efficientnet_b1": lambda **kwargs: EfficientNetBackbone(variant="efficientnet_b1", **kwargs),
    "efficientnet_b2": lambda **kwargs: EfficientNetBackbone(variant="efficientnet_b2", **kwargs),

    # EfficientNetV2 family (NEW)
    "efficientnet_v2_s": lambda **kwargs: EfficientNetV2Backbone(variant="efficientnet_v2_s", **kwargs),
    "efficientnet_v2_m": lambda **kwargs: EfficientNetV2Backbone(variant="efficientnet_v2_m", **kwargs),
    "efficientnet_v2_l": lambda **kwargs: EfficientNetV2Backbone(variant="efficientnet_v2_l", **kwargs),

    # ConvNeXt family (NEW)
    "convnext_tiny": lambda **kwargs: ConvNeXtBackbone(variant="convnext_tiny", **kwargs),
    "convnext_small": lambda **kwargs: ConvNeXtBackbone(variant="convnext_small", **kwargs),
    "convnext_base": lambda **kwargs: ConvNeXtBackbone(variant="convnext_base", **kwargs),
    "convnext_large": lambda **kwargs: ConvNeXtBackbone(variant="convnext_large", **kwargs),

    # MaxViT family (NEW)
    "maxvit_t": lambda **kwargs: MaxViTBackbone(variant="maxvit_t", **kwargs),

    # RegNet family (NEW)
    "regnet_y_400mf": lambda **kwargs: RegNetBackbone(variant="regnet_y_400mf", **kwargs),
    "regnet_y_800mf": lambda **kwargs: RegNetBackbone(variant="regnet_y_800mf", **kwargs),
    "regnet_y_1_6gf": lambda **kwargs: RegNetBackbone(variant="regnet_y_1_6gf", **kwargs),
    "regnet_y_3_2gf": lambda **kwargs: RegNetBackbone(variant="regnet_y_3_2gf", **kwargs),
    "regnet_y_8gf": lambda **kwargs: RegNetBackbone(variant="regnet_y_8gf", **kwargs),
    "regnet_y_16gf": lambda **kwargs: RegNetBackbone(variant="regnet_y_16gf", **kwargs),
    "regnet_y_32gf": lambda **kwargs: RegNetBackbone(variant="regnet_y_32gf", **kwargs),

    # Vision Transformer family
    "vit_b_16": lambda **kwargs: ViTBackbone(variant="vit_b_16", **kwargs),
    "vit_b_32": lambda **kwargs: ViTBackbone(variant="vit_b_32", **kwargs),

    # Swin Transformer family
    "swin_t": lambda **kwargs: SwinBackbone(variant="swin_t", **kwargs),
    "swin_s": lambda **kwargs: SwinBackbone(variant="swin_s", **kwargs),
}


def create_vision_backbone(
    backbone_name: str,
    pretrained: bool = True,
    freeze: bool = False,
    feature_dim: int = 128,
    dropout: float = 0.3,
    attention_type: str = "cbam",
    use_roi_guidance: bool = False,
) -> BaseVisionBackbone:
    """
    Factory function to create vision backbones.

    Args:
        backbone_name: Name of the backbone architecture
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone parameters
        feature_dim: Output feature dimension
        dropout: Dropout rate for projection head
        attention_type: Type of attention mechanism ("cbam", "se", "eca", "none")
        use_roi_guidance: Whether to use ROI-guided spatial attention

    Returns:
        Vision backbone instance

    Example:
        >>> backbone = create_vision_backbone("resnet18", pretrained=True, feature_dim=128)
        >>> features = backbone(images)  # (B, 128)
    """
    if backbone_name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"Unknown backbone: {backbone_name}. Available: {available}"
        )

    # ViT and Swin don't support CNN-style attention
    if backbone_name.startswith(("vit_", "swin_")):
        return BACKBONE_REGISTRY[backbone_name](
            pretrained=pretrained,
            freeze=freeze,
            feature_dim=feature_dim,
            dropout=dropout,
        )

    return BACKBONE_REGISTRY[backbone_name](
        pretrained=pretrained,
        freeze=freeze,
        feature_dim=feature_dim,
        dropout=dropout,
        attention_type=attention_type,
        use_roi_guidance=use_roi_guidance,
    )


def list_available_backbones() -> list[str]:
    """Return list of available backbone names."""
    return list(BACKBONE_REGISTRY.keys())
