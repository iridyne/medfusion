"""
注意力监督基类

定义注意力监督的抽象接口和通用功能。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class AttentionLoss:
    """注意力损失的返回结果"""

    total_loss: torch.Tensor
    """总损失"""

    components: dict[str, torch.Tensor]
    """损失组件（用于日志记录）"""

    attention_weights: torch.Tensor | None = None
    """注意力权重（用于可视化）"""

    metadata: dict[str, Any] | None = None
    """额外的元数据"""


class BaseAttentionSupervision(ABC, nn.Module):
    """
    注意力监督基类

    所有注意力监督方法的抽象基类，定义统一的接口。

    Args:
        loss_weight: 注意力损失的权重（相对于主任务损失）
        enabled: 是否启用注意力监督
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.enabled = enabled

    @abstractmethod
    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AttentionLoss:
        """
        计算注意力监督损失

        Args:
            attention_weights: 模型计算的注意力权重 (B, H, W) 或 (B, 1, H, W)
            features: 特征图 (B, C, H, W)
            targets: 监督目标（如分割掩码），可选
            **kwargs: 额外的参数

        Returns:
            AttentionLoss: 包含损失值和元数据的对象
        """

    def forward(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AttentionLoss:
        """
        前向传播

        Args:
            attention_weights: 注意力权重
            features: 特征图
            targets: 监督目标
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        if not self.enabled or not self.training:
            # 不启用或推理阶段，返回零损失
            return AttentionLoss(
                total_loss=torch.tensor(0.0, device=attention_weights.device),
                components={},
                attention_weights=attention_weights,
            )

        # 计算注意力损失
        loss_result = self.compute_attention_loss(
            attention_weights=attention_weights,
            features=features,
            targets=targets,
            **kwargs,
        )

        # 应用权重
        loss_result.total_loss = loss_result.total_loss * self.loss_weight

        return loss_result

    def normalize_attention(
        self,
        attention: torch.Tensor,
        method: str = "softmax",
    ) -> torch.Tensor:
        """
        归一化注意力权重

        Args:
            attention: 注意力权重 (B, H, W) 或 (B, 1, H, W)
            method: 归一化方法 ("softmax", "sigmoid", "minmax")

        Returns:
            归一化后的注意力权重
        """
        if attention.dim() == 4 and attention.size(1) == 1:
            attention = attention.squeeze(1)  # (B, H, W)

        if method == "softmax":
            # 展平并应用 softmax
            B, H, W = attention.shape
            attention_flat = attention.view(B, -1)
            attention_norm = F.softmax(attention_flat, dim=1)
            attention_norm = attention_norm.view(B, H, W)

        elif method == "sigmoid":
            attention_norm = torch.sigmoid(attention)

        elif method == "minmax":
            # 最小-最大归一化
            B, H, W = attention.shape
            attention_flat = attention.view(B, -1)
            min_val = attention_flat.min(dim=1, keepdim=True)[0]
            max_val = attention_flat.max(dim=1, keepdim=True)[0]
            attention_norm = (attention_flat - min_val) / (max_val - min_val + 1e-8)
            attention_norm = attention_norm.view(B, H, W)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return attention_norm

    def resize_target(
        self,
        target: torch.Tensor,
        size: tuple[int, int],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        调整目标尺寸以匹配注意力权重

        Args:
            target: 目标张量 (B, H, W) 或 (B, 1, H, W)
            size: 目标尺寸 (H, W)
            mode: 插值模式

        Returns:
            调整后的目标张量
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, 1, H, W)

        if target.shape[-2:] != size:
            target = F.interpolate(
                target.float(),
                size=size,
                mode=mode,
                align_corners=False if mode != "nearest" else None,
            )

        return target.squeeze(1)  # (B, H, W)


class AttentionConsistencyLoss(nn.Module):
    """
    注意力一致性损失

    鼓励注意力集中在少数区域，避免过于分散。
    """

    def __init__(self, method: str = "entropy"):
        """
        Args:
            method: 一致性度量方法
                - "entropy": 熵损失（熵越小，注意力越集中）
                - "variance": 方差损失（方差越大，注意力越集中）
                - "gini": 基尼系数（越大越集中）
        """
        super().__init__()
        self.method = method

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """
        计算一致性损失

        Args:
            attention: 注意力权重 (B, H, W)，已归一化

        Returns:
            一致性损失
        """
        B, H, W = attention.shape
        attention_flat = attention.view(B, -1)  # (B, H*W)

        if self.method == "entropy":
            # 熵损失：鼓励低熵（集中的注意力）
            entropy = -(attention_flat * torch.log(attention_flat + 1e-8)).sum(dim=1)
            loss = entropy.mean()

        elif self.method == "variance":
            # 方差损失：鼓励高方差（集中的注意力）
            variance = attention_flat.var(dim=1)
            loss = -variance.mean()  # 取负数，因为我们要最大化方差

        elif self.method == "gini":
            # 基尼系数：鼓励高基尼系数（集中的注意力）
            sorted_attention = torch.sort(attention_flat, dim=1)[0]
            n = sorted_attention.size(1)
            index = torch.arange(1, n + 1, device=attention.device).float()
            gini = (2 * (sorted_attention * index).sum(dim=1)) / (
                n * sorted_attention.sum(dim=1) + 1e-8
            ) - (n + 1) / n
            loss = -gini.mean()  # 取负数，因为我们要最大化基尼系数

        else:
            raise ValueError(f"Unknown consistency method: {self.method}")

        return loss


class AttentionSmoothLoss(nn.Module):
    """
    注意力平滑损失

    鼓励注意力在空间上平滑，避免过于跳跃。
    """

    def __init__(self, method: str = "tv"):
        """
        Args:
            method: 平滑度量方法
                - "tv": Total Variation（总变差）
                - "gradient": 梯度范数
        """
        super().__init__()
        self.method = method

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """
        计算平滑损失

        Args:
            attention: 注意力权重 (B, H, W)

        Returns:
            平滑损失
        """
        if self.method == "tv":
            # Total Variation
            diff_h = torch.abs(attention[:, 1:, :] - attention[:, :-1, :])
            diff_w = torch.abs(attention[:, :, 1:] - attention[:, :, :-1])
            loss = diff_h.mean() + diff_w.mean()

        elif self.method == "gradient":
            # 梯度范数
            grad_h = attention[:, 1:, :] - attention[:, :-1, :]
            grad_w = attention[:, :, 1:] - attention[:, :, :-1]
            loss = (grad_h**2).mean() + (grad_w**2).mean()

        else:
            raise ValueError(f"Unknown smooth method: {self.method}")

        return loss
