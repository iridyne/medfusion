"""
3D Swin Transformer backbone for medical imaging.

Implements a 3D Swin Transformer for volumetric medical images (CT, MRI).
Based on the SMuRF implementation with adaptations for the med-framework architecture.

Reference:
    Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    https://arxiv.org/abs/2103.14030
"""

from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from med_core.backbones.base import BaseVisionBackbone


class SwinTransformer3DBackbone(BaseVisionBackbone):
    """
    3D Swin Transformer backbone for volumetric medical imaging.

    Supports configurable architectures (tiny, small, base) and can extract
    features from 3D medical images like CT or MRI scans.

    Features:
    - Hierarchical feature extraction with shifted windows
    - Multi-scale feature maps
    - Optional intermediate feature return for attention supervision
    - Memory-efficient gradient checkpointing

    Example:
        >>> backbone = SwinTransformer3DBackbone(
        ...     variant="tiny",
        ...     in_channels=1,
        ...     feature_dim=128,
        ...     pretrained=False
        ... )
        >>> x = torch.randn(2, 1, 32, 64, 64)
        >>> features = backbone(x)  # [2, 128]
    """

    VARIANTS = {
        "tiny": {
            "embed_dim": 48,
            "depths": (2, 2),
            "num_heads": (3, 6),
            "window_size": [[4, 4, 4], [4, 4, 4]],
        },
        "small": {
            "embed_dim": 48,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": [[4, 4, 4], [4, 4, 4], [2, 2, 2], [2, 2, 2]],
        },
        "base": {
            "embed_dim": 96,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": [[4, 4, 4], [4, 4, 4], [2, 2, 2], [2, 2, 2]],
        },
    }

    def __init__(
        self,
        variant: Literal["tiny", "small", "base"] = "tiny",
        in_channels: int = 1,
        feature_dim: int = 128,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        pretrained: bool = False,
        freeze: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        path_dropout: float = 0.0,
        use_checkpoint: bool = False,
        return_intermediate: bool = False,
        intermediate_layer: int = -2,
    ):
        """
        Initialize 3D Swin Transformer backbone.

        Args:
            variant: Model size variant (tiny/small/base)
            in_channels: Number of input channels (1 for CT, 4 for multi-sequence MRI)
            feature_dim: Output feature dimension after projection
            patch_size: Size of patches for initial embedding (D, H, W)
            pretrained: Whether to load pretrained weights (not implemented yet)
            freeze: Whether to freeze backbone parameters
            dropout: Dropout rate for projection head
            attn_dropout: Attention dropout rate
            path_dropout: Stochastic depth rate
            use_checkpoint: Use gradient checkpointing to save memory
            return_intermediate: Return intermediate features for attention supervision
            intermediate_layer: Which layer to use for intermediate features (-2 = second to last)
        """
        super().__init__(pretrained=pretrained, freeze=freeze, feature_dim=feature_dim)

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}")

        self.variant = variant
        self.in_channels = in_channels
        self.return_intermediate = return_intermediate
        self.intermediate_layer = intermediate_layer
        self.use_checkpoint = use_checkpoint

        # Get variant configuration
        config = self.VARIANTS[variant]
        embed_dim = config["embed_dim"]
        depths = config["depths"]
        num_heads = config["num_heads"]
        window_size = config["window_size"]

        # Import Swin components from SMuRF
        # Note: We'll need to copy the SwinTransformer implementation
        # For now, we'll create a placeholder that matches the interface
        from med_core.backbones.swin_components import SwinTransformer3D

        self._backbone = SwinTransformer3D(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=dropout,
            attn_drop_rate=attn_dropout,
            drop_path_rate=path_dropout,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        # Calculate backbone output dimension
        # After len(depths) stages, channels = embed_dim * 2^(len(depths)-1)
        self._backbone_out_dim = embed_dim * (2 ** (len(depths) - 1))

        # Normalization layer
        self.norm = nn.LayerNorm(self._backbone_out_dim)

        # Global pooling
        self._pool = nn.AdaptiveAvgPool3d(1)

        # Dimension reduction
        self.dim_reduction = nn.Conv3d(self._backbone_out_dim, feature_dim, kernel_size=1)

        # Projection head (optional, for consistency with other backbones)
        self._projection = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self._output_dim = feature_dim

        # Apply freezing if requested
        if freeze:
            self.freeze_backbone()

    @property
    def backbone_output_dim(self) -> int:
        """Return the raw backbone output dimension before projection."""
        return self._backbone_out_dim

    def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
        """
        Enable gradient checkpointing for 3D Swin Transformer backbone.

        Similar to 2D Swin, the 3D version has multiple stages that can be checkpointed.

        Args:
            segments: Number of segments (None = number of stages, typically 4)
        """
        super().enable_gradient_checkpointing(segments)

        # Swin3D has multiple stages in self._backbone.layers
        if segments is None:
            segments = len(self._backbone.layers)

        from med_core.utils.gradient_checkpointing import checkpoint_sequential

        # Store original components
        patch_embed = self._backbone.patch_embed
        pos_drop = self._backbone.pos_drop
        layers = list(self._backbone.layers)
        norm = self._backbone.norm if hasattr(self._backbone, 'norm') else None

        # Create a new forward function
        def checkpointed_forward(x: torch.Tensor, normalize: bool = True) -> list[torch.Tensor]:
            if not self.training or not self._gradient_checkpointing_enabled:
                # Normal forward pass
                x = patch_embed(x)
                x = pos_drop(x)
                features = []
                for layer in layers:
                    x = layer(x)
                    features.append(x)
                if normalize and norm is not None:
                    features = [norm(f) for f in features]
                return features

            # Patch embedding
            x = patch_embed(x)
            x = pos_drop(x)

            # Collect features from each stage with checkpointing
            features = []
            for layer in layers:
                x = checkpoint_sequential(
                    [layer],
                    segments=1,
                    input=x,
                    use_reentrant=False,
                )
                features.append(x)

            # Apply normalization to features if requested
            if normalize and norm is not None:
                features = [norm(f) for f in features]

            return features

        # Replace forward method
        self._backbone.forward = checkpointed_forward

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features from the backbone.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            List of feature tensors at different scales
        """
        return self._backbone(x, normalize=True)

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass through the 3D Swin Transformer.

        Args:
            x: Input tensor [B, C, D, H, W]
            return_intermediates: Whether to return intermediate features

        Returns:
            If return_intermediates is False:
                Feature tensor [B, feature_dim]
            If return_intermediates is True:
                Dict with keys:
                    - "features": Final features [B, feature_dim]
                    - "intermediate": Intermediate features [B, C, D', H', W']
                    - "attention_maps": Attention maps (if available)
        """
        # Extract multi-scale features
        hidden_states = self.extract_features(x)

        # Use the last layer for final features
        # Shape: [B, C, D, H, W]
        hidden_output = hidden_states[-1]

        # Select intermediate layer for attention supervision
        intermediate_features = None
        if self.return_intermediate or return_intermediates:
            intermediate_features = hidden_states[self.intermediate_layer]

        # Rearrange for LayerNorm: [B, C, D, H, W] -> [B, D, H, W, C]
        hidden_output = rearrange(hidden_output, "b c d h w -> b d h w c")

        # Apply normalization
        normalized = self.norm(hidden_output)

        # Rearrange back: [B, D, H, W, C] -> [B, C, D, H, W]
        normalized = rearrange(normalized, "b d h w c -> b c d h w")

        # Global average pooling: [B, C, D, H, W] -> [B, C, 1, 1, 1]
        pooled = self._pool(normalized)

        # Dimension reduction: [B, C, 1, 1, 1] -> [B, feature_dim, 1, 1, 1]
        reduced = self.dim_reduction(pooled)

        # Projection: [B, feature_dim, 1, 1, 1] -> [B, feature_dim]
        features = self._projection(reduced)

        if self.return_intermediate or return_intermediates:
            return {
                "features": features,
                "intermediate": intermediate_features,
                "hidden_states": hidden_states,
            }

        return features

    def get_config(self) -> dict:
        """Return backbone configuration for serialization."""
        return {
            "variant": self.variant,
            "in_channels": self.in_channels,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "pretrained": self.pretrained,
            "freeze": self.freeze,
        }


# Convenience functions for creating common configurations
def swin3d_tiny(in_channels: int = 1, feature_dim: int = 128, **kwargs) -> SwinTransformer3DBackbone:
    """Create a tiny 3D Swin Transformer (fastest, lowest memory)."""
    return SwinTransformer3DBackbone(variant="tiny", in_channels=in_channels, feature_dim=feature_dim, **kwargs)


def swin3d_small(in_channels: int = 1, feature_dim: int = 128, **kwargs) -> SwinTransformer3DBackbone:
    """Create a small 3D Swin Transformer (balanced)."""
    return SwinTransformer3DBackbone(variant="small", in_channels=in_channels, feature_dim=feature_dim, **kwargs)


def swin3d_base(in_channels: int = 1, feature_dim: int = 128, **kwargs) -> SwinTransformer3DBackbone:
    """Create a base 3D Swin Transformer (highest capacity)."""
    return SwinTransformer3DBackbone(variant="base", in_channels=in_channels, feature_dim=feature_dim, **kwargs)
