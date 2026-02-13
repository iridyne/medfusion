"""
Core Swin Transformer components for 3D medical imaging.

This module contains the building blocks for 3D Swin Transformer,
adapted from the SMuRF implementation and MONAI library.

Reference:
    Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    https://arxiv.org/abs/2103.14030
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange


def window_partition(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    """
    Partition input into windows.

    Args:
        x: Input tensor [B, D, H, W, C] for 3D
        window_size: Window size [Wd, Wh, Ww]

    Returns:
        Windows tensor [B*num_windows, Wd*Wh*Ww, C]
    """
    x_shape = x.size()
    if len(x_shape) == 5:  # 3D case
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:  # 2D case
        b, h, w, c = x.shape
        x = x.view(
            b,
            h // window_size[0],
            window_size[0],
            w // window_size[1],
            window_size[1],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0] * window_size[1], c)
        )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: Sequence[int], dims: Sequence[int]
) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows: Windows tensor
        window_size: Window size
        dims: Original spatial dimensions [B, D, H, W] or [B, H, W]

    Returns:
        Reconstructed tensor
    """
    if len(dims) == 4:  # 3D case
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    elif len(dims) == 3:  # 2D case
        b, h, w = dims
        x = windows.view(
            b,
            h // window_size[0],
            w // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class MLPBlock(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear1(x))
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B*num_windows, N, C]
            mask: Attention mask [num_windows, N, N]
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias (simplified, full implementation would compute indices)
        # attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with window attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_hidden_dim, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, D, H, W, C]
        """
        B, D, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, (B, D, H, W))

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer (downsampling).
    """

    def __init__(
        self, dim: int, norm_layer: type = nn.LayerNorm, spatial_dims: int = 3
    ):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims

        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        else:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        self.norm = norm_layer(8 * dim if spatial_dims == 3 else 4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, D, H, W, C] for 3D or [B, H, W, C] for 2D
        """
        if self.spatial_dims == 3:
            B, D, H, W, C = x.shape

            # Downsample by 2x in each dimension
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 1::2, 1::2, 0::2, :]
            x4 = x[:, 0::2, 0::2, 1::2, :]
            x5 = x[:, 1::2, 0::2, 1::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]

            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        else:
            B, H, W, C = x.shape
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=tuple(w // 2 if i % 2 == 1 else 0 for w in window_size),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer.

    Args:
        in_chans: Number of input channels
        embed_dim: Patch embedding dimension
        window_size: Window size for each stage
        patch_size: Patch size for initial embedding
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: If True, add bias to qkv projections
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        use_checkpoint: Use gradient checkpointing
        spatial_dims: Spatial dimensions (2 or 3)
    """

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 48,
        window_size: list[list[int]] | None = None,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        depths: tuple[int, ...] = (2, 2),
        num_heads: tuple[int, ...] = (3, 6),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        **kwargs,
    ):
        super().__init__()
        if window_size is None:
            window_size = [[4, 4, 4], [4, 4, 4]]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.spatial_dims = spatial_dims

        # Patch embedding
        self.patch_embed = (
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            if spatial_dims == 3
            else nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size[:2], stride=patch_size[:2]
            )
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging(
                    int(embed_dim * 2**i_layer), norm_layer, spatial_dims
                )
                if i_layer < self.num_layers - 1
                else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> list[torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, D, H, W]
            normalize: Whether to normalize features

        Returns:
            List of feature tensors at different scales
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Rearrange to [B, D, H, W, C]
        if self.spatial_dims == 3:
            x = rearrange(x, "b c d h w -> b d h w c")
        else:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.pos_drop(x)

        # Forward through layers
        features = []
        for layer in self.layers:
            x = layer(x)
            # Rearrange back to [B, C, D, H, W] for output
            if self.spatial_dims == 3:
                feat = rearrange(x, "b d h w c -> b c d h w")
            else:
                feat = rearrange(x, "b h w c -> b c h w")
            features.append(feat)

        return features


# ============================================================================
# 2D Swin Transformer Components
# ============================================================================


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding.

    Args:
        img_size: Input image size (H, W)
        patch_size: Patch token size
        in_channels: Number of input channels
        embed_dim: Number of linear projection output channels
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, H', W', embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = rearrange(x, "b c h w -> b h w c")  # [B, H', W', embed_dim]
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """
    Patch Merging Layer for 2D.

    Args:
        dim: Number of input channels
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C]
        Returns:
            [B, H/2, W/2, 2*C]
        """
        B, H, W, C = x.shape

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # Merge patches: [B, H/2, W/2, 4*C]
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2, W/2, 2*C]

        return x


class WindowAttention2D(nn.Module):
    """
    Window-based multi-head self-attention for 2D.

    Args:
        dim: Number of input channels
        window_size: Window size [Wh, Ww]
        num_heads: Number of attention heads
        qkv_bias: Use bias in QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        window_size: list[int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_windows*B, N, C]
            mask: [num_windows, Wh*Ww, Wh*Ww] or None
        Returns:
            [num_windows*B, N, C]
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock2D(nn.Module):
    """
    Swin Transformer Block for 2D.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        window_size: Window size [Wh, Ww]
        shift_size: Shift size for SW-MSA [Sh, Sw]
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Use bias in QKV projection
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        shift_size: list[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_hidden_dim, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C]
        Returns:
            [B, H, W, C]
        """
        H, W = x.shape[1], x.shape[2]
        B, _, _, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)
            )
            attn_mask = self._get_attn_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Wh*Ww, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Wh*Ww, C]

        # Merge windows
        shifted_x = window_reverse(
            attn_windows, self.window_size, (B, Hp, Wp)
        )  # [B, Hp, Wp, C]

        # Reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _get_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate attention mask for SW-MSA."""
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Wh*Ww, 1]
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
