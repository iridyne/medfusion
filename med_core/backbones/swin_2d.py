"""
2D Swin Transformer Backbone for pathology images.

This module provides a 2D Swin Transformer implementation optimized for
pathology whole slide images (WSI). It supports ImageNet pretrained weights
and is compatible with the timm library.
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from .base import BaseVisionBackbone
from .swin_components import (
    PatchEmbed2D,
    PatchMerging2D,
    SwinTransformerBlock2D,
)

logger = logging.getLogger(__name__)


class SwinTransformer2D(nn.Module):
    """
    2D Swin Transformer for pathology images.

    Args:
        img_size: Input image size (H, W)
        patch_size: Patch size
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        window_size: Window size for each stage
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Use bias in QKV projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: list[list[int]] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        if window_size is None:
            window_size = [[7, 7]] * len(depths)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer2D(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            normalize: Apply layer normalization

        Returns:
            Output tensor [B, H', W', C']
        """
        # Patch embedding: [B, C, H, W] -> [B, H/4, W/4, embed_dim]
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Apply Swin Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Normalize
        if normalize:
            x = self.norm(x)

        return x


class BasicLayer2D(nn.Module):
    """
    A basic 2D Swin Transformer layer for one stage.

    Args:
        dim: Number of input channels
        depth: Number of blocks
        num_heads: Number of attention heads
        window_size: Local window size
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Use bias in QKV projection
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
        downsample: Downsample layer at the end of the layer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: list[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: list[float] = None,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        if drop_path is None:
            drop_path = [0.0] * depth

        # Build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock2D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0, 0]
                    if (i % 2 == 0)
                    else [ws // 2 for ws in window_size],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinTransformer2DBackbone(BaseVisionBackbone):
    """
    2D Swin Transformer backbone for pathology images.

    Supports three variants:
    - tiny: ~28M parameters
    - small: ~50M parameters
    - base: ~88M parameters

    Example:
        >>> from med_core.backbones.swin_2d import swin2d_tiny
        >>> backbone = swin2d_tiny(in_channels=3, feature_dim=512)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features = backbone(x)  # [2, 512]
    """

    VARIANTS = {
        "tiny": {
            "embed_dim": 96,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": [[7, 7], [7, 7], [7, 7], [7, 7]],
        },
        "small": {
            "embed_dim": 96,
            "depths": (2, 2, 18, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": [[7, 7], [7, 7], [7, 7], [7, 7]],
        },
        "base": {
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": [[7, 7], [7, 7], [7, 7], [7, 7]],
        },
    }

    def __init__(
        self,
        variant: Literal["tiny", "small", "base"] = "tiny",
        in_channels: int = 3,
        feature_dim: int = 512,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pretrained: bool = False,
        freeze: bool = False,
        return_intermediate: bool = False,
    ):
        super().__init__()

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}"
            )

        config = self.VARIANTS[variant]
        self.variant = variant
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.return_intermediate = return_intermediate

        # Build backbone
        self._backbone = SwinTransformer2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            window_size=config["window_size"],
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )

        self._backbone_out_dim = self._backbone.num_features

        # Global pooling
        self._pool = nn.AdaptiveAvgPool2d(1)

        # Dimension reduction
        self.dim_reduction = nn.Conv2d(
            self._backbone_out_dim, feature_dim, kernel_size=1
        )

        # Projection head
        self._projection = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self._output_dim = feature_dim

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()

        # Apply freezing if requested
        if freeze:
            self.freeze_backbone()

    @property
    def output_dim(self) -> int:
        """Return output feature dimension."""
        return self._output_dim

    @property
    def backbone_output_dim(self) -> int:
        """Return the raw backbone output dimension before projection."""
        return self._backbone_out_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor [B, H', W', C']
        """
        return self._backbone(x, normalize=True)

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | dict:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            return_intermediates: Return intermediate features

        Returns:
            Feature tensor [B, feature_dim] or dict with intermediates
        """
        # Extract features: [B, C, H, W] -> [B, H', W', C']
        hidden_states = self.extract_features(x)

        # Store intermediate features if needed
        intermediate_features = []
        if self.return_intermediate or return_intermediates:
            # Collect features from each stage
            x_temp = self._backbone.patch_embed(x)
            x_temp = self._backbone.pos_drop(x_temp)
            for layer in self._backbone.layers:
                x_temp = layer(x_temp)
                intermediate_features.append(x_temp)

        # Get final hidden output
        hidden_output = hidden_states

        # Rearrange: [B, H, W, C] -> [B, C, H, W]
        normalized = rearrange(hidden_output, "b h w c -> b c h w")

        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        pooled = self._pool(normalized)

        # Dimension reduction: [B, C, 1, 1] -> [B, feature_dim, 1, 1]
        reduced = self.dim_reduction(pooled)

        # Projection: [B, feature_dim, 1, 1] -> [B, feature_dim]
        features = self._projection(reduced)

        if self.return_intermediate or return_intermediates:
            return {
                "features": features,
                "intermediate": intermediate_features,
                "hidden_states": hidden_states,
            }

        return features

    def _load_pretrained_weights(self):
        """Load pretrained weights from timm."""
        try:
            import timm

            # Map variant to timm model name
            timm_names = {
                "tiny": "swin_tiny_patch4_window7_224",
                "small": "swin_small_patch4_window7_224",
                "base": "swin_base_patch4_window7_224",
            }

            model_name = timm_names[self.variant]
            pretrained_model = timm.create_model(model_name, pretrained=True)

            # Load weights (only backbone, not head)
            state_dict = pretrained_model.state_dict()
            backbone_state_dict = {}

            for k, v in state_dict.items():
                if k.startswith("head"):
                    continue
                backbone_state_dict[k] = v

            # Load into our model
            self._backbone.load_state_dict(backbone_state_dict, strict=False)
            logger.info(f"Loaded pretrained weights for {model_name}")

        except ImportError:
            logger.warning("timm not installed, skipping pretrained weights")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self._backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self._backbone.parameters():
            param.requires_grad = True

    def get_config(self) -> dict:
        """Return backbone configuration."""
        return {
            "variant": self.variant,
            "in_channels": self.in_channels,
            "feature_dim": self.feature_dim,
            "return_intermediate": self.return_intermediate,
        }


# Convenience functions
def swin2d_tiny(
    in_channels: int = 3,
    feature_dim: int = 512,
    pretrained: bool = False,
    **kwargs,
) -> SwinTransformer2DBackbone:
    """Create Swin Transformer 2D Tiny variant."""
    return SwinTransformer2DBackbone(
        variant="tiny",
        in_channels=in_channels,
        feature_dim=feature_dim,
        pretrained=pretrained,
        **kwargs,
    )


def swin2d_small(
    in_channels: int = 3,
    feature_dim: int = 512,
    pretrained: bool = False,
    **kwargs,
) -> SwinTransformer2DBackbone:
    """Create Swin Transformer 2D Small variant."""
    return SwinTransformer2DBackbone(
        variant="small",
        in_channels=in_channels,
        feature_dim=feature_dim,
        pretrained=pretrained,
        **kwargs,
    )


def swin2d_base(
    in_channels: int = 3,
    feature_dim: int = 512,
    pretrained: bool = False,
    **kwargs,
) -> SwinTransformer2DBackbone:
    """Create Swin Transformer 2D Base variant."""
    return SwinTransformer2DBackbone(
        variant="base",
        in_channels=in_channels,
        feature_dim=feature_dim,
        pretrained=pretrained,
        **kwargs,
    )
