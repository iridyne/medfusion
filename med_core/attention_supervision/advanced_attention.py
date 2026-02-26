"""
高级注意力模块

实现 SE (Squeeze-and-Excitation)、ECA (Efficient Channel Attention)
和 Transformer 注意力机制。
"""

import math
from typing import Any

import torch
import torch.nn as nn


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation 注意力

    通过全局平均池化和两层全连接网络学习通道注意力。

    Reference:
        Hu et al. "Squeeze-and-Excitation Networks" CVPR 2018

    Args:
        channels: 输入通道数
        reduction: 降维比例（默认 16）
        activation: 激活函数（默认 ReLU）

    Example:
        >>> se = SEAttention(channels=256, reduction=16)
        >>> x = torch.randn(2, 256, 14, 14)
        >>> out = se(x)
        >>> out.shape
        torch.Size([2, 256, 14, 14])
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # 降维
        reduced_channels = max(channels // reduction, 1)

        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: 两层全连接
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            self._get_activation(activation),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            加权后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Squeeze: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)

        # Excitation: (B, C, 1, 1) -> (B, C)
        y = y.view(B, C)
        y = self.fc(y)

        # 重塑并应用注意力: (B, C) -> (B, C, 1, 1)
        y = y.view(B, C, 1, 1)

        # 加权
        return x * y.expand_as(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            注意力权重 (B, C)
        """
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        weights = self.fc(y)
        return weights


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention

    使用 1D 卷积实现高效的通道注意力，避免降维。

    Reference:
        Wang et al. "ECA-Net: Efficient Channel Attention for Deep CNNs" CVPR 2020

    Args:
        channels: 输入通道数
        kernel_size: 1D 卷积核大小（默认自动计算）
        gamma: 自动计算 kernel_size 的参数
        b: 自动计算 kernel_size 的参数

    Example:
        >>> eca = ECAAttention(channels=256)
        >>> x = torch.randn(2, 256, 14, 14)
        >>> out = eca(x)
        >>> out.shape
        torch.Size([2, 256, 14, 14])
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | None = None,
        gamma: int = 2,
        b: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels

        # 自动计算 kernel_size
        if kernel_size is None:
            # k = |log2(C) / gamma + b / gamma|_odd
            t = int(abs(math.log2(channels) / gamma + b / gamma))
            kernel_size = t if t % 2 else t + 1

        self.kernel_size = kernel_size

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1D 卷积
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            加权后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)

        # 转换为 1D: (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        # 1D 卷积: (B, 1, C) -> (B, 1, C)
        y = self.conv(y)

        # Sigmoid 激活: (B, 1, C) -> (B, C, 1, 1)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)

        # 加权
        return x * y.expand_as(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            注意力权重 (B, C)
        """
        B, C, _, _ = x.shape
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        weights = self.sigmoid(y).squeeze(1)
        return weights


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    学习空间维度的注意力权重。

    Args:
        kernel_size: 卷积核大小

    Example:
        >>> sa = SpatialAttention(kernel_size=7)
        >>> x = torch.randn(2, 256, 14, 14)
        >>> out = sa(x)
        >>> out.shape
        torch.Size([2, 256, 14, 14])
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            加权后的特征 (B, C, H, W)
        """
        # 计算通道维度的统计量
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # 拼接
        y = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        # 卷积 + Sigmoid
        y = self.conv(y)  # (B, 1, H, W)
        y = self.sigmoid(y)

        # 加权
        return x * y

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            注意力权重 (B, 1, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        weights = self.sigmoid(y)
        return weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module

    结合通道注意力和空间注意力。

    Reference:
        Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018

    Args:
        channels: 输入通道数
        reduction: 通道注意力的降维比例
        spatial_kernel: 空间注意力的卷积核大小

    Example:
        >>> cbam = CBAM(channels=256)
        >>> x = torch.randn(2, 256, 14, 14)
        >>> out = cbam(x)
        >>> out.shape
        torch.Size([2, 256, 14, 14])
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = SEAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            加权后的特征 (B, C, H, W)
        """
        # 通道注意力
        x = self.channel_attention(x)

        # 空间注意力
        x = self.spatial_attention(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制

    Transformer 风格的自注意力。

    Args:
        dim: 输入维度
        num_heads: 注意力头数
        qkv_bias: 是否使用 QKV 偏置
        attn_drop: 注意力 dropout
        proj_drop: 投影 dropout

    Example:
        >>> mhsa = MultiHeadSelfAttention(dim=256, num_heads=8)
        >>> x = torch.randn(2, 196, 256)  # (B, N, C)
        >>> out = mhsa(x)
        >>> out.shape
        torch.Size([2, 196, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, N, C)

        Returns:
            输出特征 (B, N, C)
        """
        B, N, C = x.shape

        # QKV 投影: (B, N, C) -> (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和: (B, num_heads, N, N) @ (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重

        Args:
            x: 输入特征 (B, N, C)

        Returns:
            注意力权重 (B, num_heads, N, N)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        return attn


class TransformerAttention2D(nn.Module):
    """
    2D 特征图的 Transformer 注意力

    将 2D 特征图转换为序列，应用 Transformer 注意力。

    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        qkv_bias: 是否使用 QKV 偏置
        attn_drop: 注意力 dropout
        proj_drop: 投影 dropout

    Example:
        >>> ta = TransformerAttention2D(channels=256, num_heads=8)
        >>> x = torch.randn(2, 256, 14, 14)
        >>> out = ta(x)
        >>> out.shape
        torch.Size([2, 256, 14, 14])
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.attention = MultiHeadSelfAttention(
            dim=channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            输出特征 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 转换为序列: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)

        # 应用注意力
        x_seq = self.attention(x_seq)

        # 转换回 2D: (B, H*W, C) -> (B, C, H, W)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            注意力权重 (B, num_heads, H*W, H*W)
        """
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        attn = self.attention.get_attention_weights(x_seq)
        return attn


def create_attention_module(
    attention_type: str,
    channels: int,
    **kwargs,
) -> nn.Module:
    """
    创建注意力模块的工厂函数

    Args:
        attention_type: 注意力类型
            - "se": Squeeze-and-Excitation
            - "eca": Efficient Channel Attention
            - "spatial": Spatial Attention
            - "cbam": CBAM (Channel + Spatial)
            - "transformer": Transformer Attention
        channels: 输入通道数
        **kwargs: 额外参数

    Returns:
        注意力模块

    Example:
        >>> attn = create_attention_module("se", channels=256, reduction=16)
        >>> x = torch.randn(2, 256, 14, 14)
        >>> out = attn(x)
    """
    if attention_type == "se":
        return SEAttention(channels, **kwargs)
    elif attention_type == "eca":
        return ECAAttention(channels, **kwargs)
    elif attention_type == "spatial":
        return SpatialAttention(**kwargs)
    elif attention_type == "cbam":
        return CBAM(channels, **kwargs)
    elif attention_type == "transformer":
        return TransformerAttention2D(channels, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
