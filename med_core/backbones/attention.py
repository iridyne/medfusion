"""
Attention mechanisms for vision backbones.

Implements:
- CBAM (Convolutional Block Attention Module)
- SE (Squeeze-and-Excitation)
- ECA (Efficient Channel Attention)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, C, H, W)
            return_weights: If True, return (output, weights)

        Returns:
            If return_weights=False: output features (B, C, H, W)
            If return_weights=True: (output features, channel weights (B, C, 1, 1))
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        weights = self.sigmoid(avg_out + max_out)

        if return_weights:
            return weights, weights
        else:
            return weights


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, C, H, W)
            return_weights: If True, return (output, weights)

        Returns:
            If return_weights=False: spatial attention weights (B, 1, H, W)
            If return_weights=True: (weights, weights) for consistency
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        weights = self.sigmoid(self.conv(concat))

        if return_weights:
            return weights, weights
        else:
            return weights


class ROIGuidedSpatialAttention(nn.Module):
    """
    Spatial attention with ROI guidance for medical images.

    Suppresses attention on known artifact regions (e.g., watermarks, rulers)
    and enhances attention on clinically relevant areas.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        roi_top_ratio: float = 0.85,  # Focus on top 85% (avoid bottom watermarks)
        roi_side_ratio: float = 0.85,  # Focus on central 85% width
        suppression_weight: float = 2.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.roi_top_ratio = roi_top_ratio
        self.roi_side_ratio = roi_side_ratio
        self.suppression_weight = suppression_weight
        self._roi_mask = None

    def _create_roi_mask(self, size: tuple[int, int], device: torch.device) -> torch.Tensor:
        """Create ROI mask that suppresses artifact regions."""
        h, w = size
        mask = torch.zeros(1, 1, h, w, device=device)

        # Define ROI region (center, avoiding edges and bottom)
        h_start = 0
        h_end = int(h * self.roi_top_ratio)
        w_start = int(w * (1 - self.roi_side_ratio) / 2)
        w_end = int(w * (1 + self.roi_side_ratio) / 2)

        # Positive weight for ROI
        mask[:, :, h_start:h_end, w_start:w_end] = 1.0

        # Negative weight for suppression regions
        mask[:, :, h_end:, :] = -self.suppression_weight  # Bottom region

        return mask

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, C, H, W)
            return_weights: If True, return (output, weights)

        Returns:
            If return_weights=False: spatial attention weights (B, 1, H, W)
            If return_weights=True: (weights, weights) for consistency
        """
        # Compute base spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        raw_attention = self.conv(concat)

        # Apply ROI guidance
        if self._roi_mask is None or self._roi_mask.shape[-2:] != x.shape[-2:]:
            self._roi_mask = self._create_roi_mask(x.shape[-2:], x.device)

        # Enhance ROI, suppress artifacts
        guided_attention = raw_attention + 0.3 * self._roi_mask
        weights = self.sigmoid(guided_attention)

        if return_weights:
            return weights, weights
        else:
            return weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention.

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

    Args:
        in_channels: Number of input channels
        reduction_ratio: Channel reduction ratio for channel attention
        use_spatial: Whether to use spatial attention
        use_roi_guidance: Whether to use ROI-guided spatial attention
        return_attention_weights: If True, forward() returns (output, weights_dict)
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        use_spatial: bool = True,
        use_roi_guidance: bool = False,
        return_attention_weights: bool = False,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.use_spatial = use_spatial
        self.return_attention_weights = return_attention_weights

        if use_spatial:
            if use_roi_guidance:
                self.spatial_attention = ROIGuidedSpatialAttention()
            else:
                self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: Input features (B, C, H, W)

        Returns:
            If return_attention_weights=False:
                Output features (B, C, H, W)
            If return_attention_weights=True:
                (output features, {"channel_weights": (B, C, 1, 1), "spatial_weights": (B, 1, H, W) or None})
        """
        # Channel attention
        channel_weights = self.channel_attention(x, return_weights=False)
        x = x * channel_weights

        # Spatial attention
        spatial_weights = None
        if self.use_spatial:
            spatial_weights = self.spatial_attention(x, return_weights=False)
            x = x * spatial_weights

        # Return based on configuration
        if self.return_attention_weights:
            weights_dict = {
                "channel_weights": channel_weights,
                "spatial_weights": spatial_weights,
            }
            return x, weights_dict
        else:
            return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.

    Reference:
        Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(b, c)

        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block.

    Uses 1D convolution instead of fully connected layers
    for more efficient channel attention.

    Reference:
        Wang et al., "ECA-Net: Efficient Channel Attention", CVPR 2020
    """

    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Average pooling
        y = self.avg_pool(x).view(b, 1, c)

        # 1D convolution for channel interaction
        y = self.conv(y).view(b, c, 1, 1)

        return x * self.sigmoid(y).expand_as(x)


def create_attention_module(
    attention_type: str,
    in_channels: int,
    use_roi_guidance: bool = False,
    return_attention_weights: bool = False,
) -> nn.Module | None:
    """
    Factory function to create attention modules.

    Args:
        attention_type: Type of attention ("cbam", "se", "eca", "none")
        in_channels: Number of input channels
        use_roi_guidance: Whether to use ROI-guided spatial attention (CBAM only)
        return_attention_weights: Whether to return attention weights (CBAM only)

    Returns:
        Attention module or None

    Note:
        Only CBAM supports return_attention_weights. SE and ECA are channel-only
        attention mechanisms and do not have spatial attention weights to return.
    """
    if attention_type == "cbam":
        return CBAM(
            in_channels,
            use_roi_guidance=use_roi_guidance,
            return_attention_weights=return_attention_weights,
        )
    elif attention_type == "se":
        return SEBlock(in_channels)
    elif attention_type == "eca":
        return ECABlock(in_channels)
    elif attention_type == "none":
        return None
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
