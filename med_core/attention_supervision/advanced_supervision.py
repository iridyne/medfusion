"""
高级注意力监督

为 SE、ECA 和 Transformer 注意力提供监督损失。
"""

import torch
import torch.nn.functional as F

from med_core.attention_supervision.base import (
    AttentionLoss,
    BaseAttentionSupervision,
)


class ChannelAttentionSupervision(BaseAttentionSupervision):
    """
    通道注意力监督

    用于 SE 和 ECA 等通道注意力机制。

    Args:
        loss_weight: 损失权重
        target_channels: 目标通道索引（如果已知哪些通道重要）
        diversity_weight: 多样性损失权重（鼓励不同通道关注不同特征）
        sparsity_weight: 稀疏性损失权重（鼓励只激活少数通道）

    Example:
        >>> supervision = ChannelAttentionSupervision(loss_weight=0.1)
        >>> channel_weights = torch.randn(2, 256)  # (B, C)
        >>> features = torch.randn(2, 256, 14, 14)
        >>> loss = supervision(channel_weights, features)
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        target_channels: list[int] | None = None,
        diversity_weight: float = 0.1,
        sparsity_weight: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)
        self.target_channels = target_channels
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算通道注意力监督损失

        Args:
            attention_weights: 通道注意力权重 (B, C)
            features: 特征图 (B, C, H, W)
            targets: 目标通道索引（可选）

        Returns:
            AttentionLoss
        """
        components = {}
        total_loss = torch.tensor(0.0, device=attention_weights.device)

        # 1. 目标通道损失（如果提供）
        if self.target_channels is not None or targets is not None:
            target_idx = targets if targets is not None else self.target_channels

            # 创建目标掩码
            B, C = attention_weights.shape
            target_mask = torch.zeros_like(attention_weights)
            target_mask[:, target_idx] = 1.0

            # BCE 损失
            target_loss = F.binary_cross_entropy(
                attention_weights,
                target_mask,
                reduction="mean",
            )

            components["target_loss"] = target_loss
            total_loss = total_loss + target_loss

        # 2. 多样性损失（鼓励不同样本关注不同通道）
        if self.diversity_weight > 0:
            # 计算样本间的相似度
            attn_norm = F.normalize(attention_weights, p=2, dim=1)
            similarity = torch.mm(attn_norm, attn_norm.t())  # (B, B)

            # 去除对角线（自己和自己的相似度）
            mask = torch.eye(similarity.size(0), device=similarity.device)
            similarity = similarity * (1 - mask)

            # 鼓励低相似度
            diversity_loss = similarity.mean()

            components["diversity_loss"] = diversity_loss
            total_loss = total_loss + self.diversity_weight * diversity_loss

        # 3. 稀疏性损失（鼓励只激活少数通道）
        if self.sparsity_weight > 0:
            # L1 正则化
            sparsity_loss = attention_weights.abs().mean()

            components["sparsity_loss"] = sparsity_loss
            total_loss = total_loss + self.sparsity_weight * sparsity_loss

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_weights,
        )


class SpatialAttentionSupervision(BaseAttentionSupervision):
    """
    空间注意力监督

    用于空间注意力机制。

    Args:
        loss_weight: 损失权重
        consistency_weight: 一致性损失权重
        smoothness_weight: 平滑性损失权重

    Example:
        >>> supervision = SpatialAttentionSupervision(loss_weight=0.1)
        >>> spatial_weights = torch.randn(2, 1, 14, 14)  # (B, 1, H, W)
        >>> features = torch.randn(2, 256, 14, 14)
        >>> loss = supervision(spatial_weights, features)
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        consistency_weight: float = 0.1,
        smoothness_weight: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算空间注意力监督损失

        Args:
            attention_weights: 空间注意力权重 (B, 1, H, W) 或 (B, H, W)
            features: 特征图 (B, C, H, W)
            targets: 目标掩码（可选）

        Returns:
            AttentionLoss
        """
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.unsqueeze(1)

        components = {}
        total_loss = torch.tensor(0.0, device=attention_weights.device)

        # 1. 目标掩码损失（如果提供）
        if targets is not None:
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)

            # 调整大小
            if targets.shape[-2:] != attention_weights.shape[-2:]:
                targets = F.interpolate(
                    targets.float(),
                    size=attention_weights.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # BCE 损失
            target_loss = F.binary_cross_entropy(
                attention_weights,
                targets,
                reduction="mean",
            )

            components["target_loss"] = target_loss
            total_loss = total_loss + target_loss

        # 2. 一致性损失（鼓励注意力集中）
        if self.consistency_weight > 0:
            # 熵损失
            attn_flat = attention_weights.flatten(2)  # (B, 1, H*W)
            attn_flat = attn_flat + 1e-8  # 避免 log(0)
            entropy = -(attn_flat * torch.log(attn_flat)).sum(dim=2).mean()

            components["consistency_loss"] = entropy
            total_loss = total_loss + self.consistency_weight * entropy

        # 3. 平滑性损失（鼓励空间平滑）
        if self.smoothness_weight > 0:
            # Total Variation
            diff_h = torch.abs(
                attention_weights[:, :, 1:, :] - attention_weights[:, :, :-1, :]
            )
            diff_w = torch.abs(
                attention_weights[:, :, :, 1:] - attention_weights[:, :, :, :-1]
            )
            smoothness_loss = diff_h.mean() + diff_w.mean()

            components["smoothness_loss"] = smoothness_loss
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_weights.squeeze(1),
        )


class TransformerAttentionSupervision(BaseAttentionSupervision):
    """
    Transformer 注意力监督

    用于多头自注意力机制。

    Args:
        loss_weight: 损失权重
        head_diversity_weight: 头多样性损失权重（鼓励不同头关注不同模式）
        locality_weight: 局部性损失权重（鼓励关注局部区域）

    Example:
        >>> supervision = TransformerAttentionSupervision(loss_weight=0.1)
        >>> attn_weights = torch.randn(2, 8, 196, 196)  # (B, num_heads, N, N)
        >>> features = torch.randn(2, 196, 256)
        >>> loss = supervision(attn_weights, features)
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        head_diversity_weight: float = 0.1,
        locality_weight: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)
        self.head_diversity_weight = head_diversity_weight
        self.locality_weight = locality_weight

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算 Transformer 注意力监督损失

        Args:
            attention_weights: 注意力权重 (B, num_heads, N, N)
            features: 特征 (B, N, C) 或 (B, C, H, W)
            targets: 目标注意力模式（可选）

        Returns:
            AttentionLoss
        """
        components = {}
        total_loss = torch.tensor(0.0, device=attention_weights.device)

        B, num_heads, N, _ = attention_weights.shape

        # 1. 头多样性损失（鼓励不同头学习不同模式）
        if self.head_diversity_weight > 0:
            # 计算头之间的相似度
            attn_flat = attention_weights.view(B, num_heads, -1)  # (B, num_heads, N*N)
            attn_norm = F.normalize(attn_flat, p=2, dim=2)

            # 对每个样本计算头之间的相似度
            diversity_loss = 0
            for b in range(B):
                similarity = torch.mm(
                    attn_norm[b], attn_norm[b].t()
                )  # (num_heads, num_heads)

                # 去除对角线
                mask = torch.eye(num_heads, device=similarity.device)
                similarity = similarity * (1 - mask)

                # 鼓励低相似度
                diversity_loss = diversity_loss + similarity.mean()

            diversity_loss = diversity_loss / B

            components["head_diversity_loss"] = diversity_loss
            total_loss = total_loss + self.head_diversity_weight * diversity_loss

        # 2. 局部性损失（鼓励关注局部区域）
        if self.locality_weight > 0:
            # 计算注意力的距离加权
            # 假设 N = H * W，可以重塑为 2D
            H = W = int(N ** 0.5)
            if H * W == N:
                # 创建位置索引
                pos_y, pos_x = torch.meshgrid(
                    torch.arange(H, device=attention_weights.device),
                    torch.arange(W, device=attention_weights.device),
                    indexing="ij",
                )
                pos = torch.stack([pos_y, pos_x], dim=-1).view(N, 2).float()

                # 计算距离矩阵
                dist = torch.cdist(pos, pos, p=2)  # (N, N)

                # 距离加权的注意力
                # 鼓励注意力集中在近距离
                weighted_attn = attention_weights * dist.unsqueeze(0).unsqueeze(0)
                locality_loss = weighted_attn.mean()

                components["locality_loss"] = locality_loss
                total_loss = total_loss + self.locality_weight * locality_loss

        # 3. 目标注意力损失（如果提供）
        if targets is not None:
            target_loss = F.mse_loss(attention_weights, targets)

            components["target_loss"] = target_loss
            total_loss = total_loss + target_loss

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_weights,
        )


class HybridAttentionSupervision(BaseAttentionSupervision):
    """
    混合注意力监督

    结合通道、空间和 Transformer 注意力的监督。

    Args:
        loss_weight: 损失权重
        channel_weight: 通道注意力权重
        spatial_weight: 空间注意力权重
        transformer_weight: Transformer 注意力权重

    Example:
        >>> supervision = HybridAttentionSupervision(loss_weight=0.1)
        >>> attentions = {
        ...     "channel": torch.randn(2, 256),
        ...     "spatial": torch.randn(2, 1, 14, 14),
        ... }
        >>> features = torch.randn(2, 256, 14, 14)
        >>> loss = supervision(attentions, features)
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        channel_weight: float = 1.0,
        spatial_weight: float = 1.0,
        transformer_weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)

        self.channel_supervision = ChannelAttentionSupervision(
            loss_weight=channel_weight,
            enabled=enabled,
        )
        self.spatial_supervision = SpatialAttentionSupervision(
            loss_weight=spatial_weight,
            enabled=enabled,
        )
        self.transformer_supervision = TransformerAttentionSupervision(
            loss_weight=transformer_weight,
            enabled=enabled,
        )

    def compute_attention_loss(
        self,
        attention_weights: dict[str, torch.Tensor],
        features: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算混合注意力监督损失

        Args:
            attention_weights: 注意力权重字典
                - "channel": (B, C)
                - "spatial": (B, 1, H, W)
                - "transformer": (B, num_heads, N, N)
            features: 特征图
            targets: 目标字典（可选）

        Returns:
            AttentionLoss
        """
        components = {}
        total_loss = torch.tensor(0.0, device=features.device)

        targets = targets or {}

        # 通道注意力
        if "channel" in attention_weights:
            channel_loss = self.channel_supervision(
                attention_weights["channel"],
                features,
                targets.get("channel"),
            )
            components.update(
                {f"channel_{k}": v for k, v in channel_loss.components.items()}
            )
            total_loss = total_loss + channel_loss.total_loss

        # 空间注意力
        if "spatial" in attention_weights:
            spatial_loss = self.spatial_supervision(
                attention_weights["spatial"],
                features,
                targets.get("spatial"),
            )
            components.update(
                {f"spatial_{k}": v for k, v in spatial_loss.components.items()}
            )
            total_loss = total_loss + spatial_loss.total_loss

        # Transformer 注意力
        if "transformer" in attention_weights:
            transformer_loss = self.transformer_supervision(
                attention_weights["transformer"],
                features,
                targets.get("transformer"),
            )
            components.update(
                {f"transformer_{k}": v for k, v in transformer_loss.components.items()}
            )
            total_loss = total_loss + transformer_loss.total_loss

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_weights,
        )


def create_attention_supervision(
    supervision_type: str,
    loss_weight: float = 0.1,
    **kwargs,
) -> BaseAttentionSupervision:
    """
    创建注意力监督的工厂函数

    Args:
        supervision_type: 监督类型
            - "channel": 通道注意力监督
            - "spatial": 空间注意力监督
            - "transformer": Transformer 注意力监督
            - "hybrid": 混合注意力监督
        loss_weight: 损失权重
        **kwargs: 额外参数

    Returns:
        注意力监督模块

    Example:
        >>> supervision = create_attention_supervision(
        ...     "channel",
        ...     loss_weight=0.1,
        ...     diversity_weight=0.1,
        ... )
    """
    if supervision_type == "channel":
        return ChannelAttentionSupervision(loss_weight=loss_weight, **kwargs)
    elif supervision_type == "spatial":
        return SpatialAttentionSupervision(loss_weight=loss_weight, **kwargs)
    elif supervision_type == "transformer":
        return TransformerAttentionSupervision(loss_weight=loss_weight, **kwargs)
    elif supervision_type == "hybrid":
        return HybridAttentionSupervision(loss_weight=loss_weight, **kwargs)
    else:
        raise ValueError(f"Unknown supervision type: {supervision_type}")
