"""
多实例学习（MIL）注意力监督

将图像分成多个patches，自动学习哪些patches重要。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from med_core.attention_supervision.base import (
    AttentionLoss,
    BaseAttentionSupervision,
)


def extract_patches(
    images: torch.Tensor,
    patch_size: int,
    stride: int | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    从图像中提取patches

    Args:
        images: 输入图像 (B, C, H, W)
        patch_size: Patch 大小
        stride: 步长，如果为 None 则等于 patch_size（无重叠）

    Returns:
        patches: (B, num_patches, C, patch_size, patch_size)
        grid_size: (num_patches_h, num_patches_w)

    Example:
        >>> images = torch.randn(2, 3, 224, 224)
        >>> patches, grid_size = extract_patches(images, patch_size=16)
        >>> patches.shape
        torch.Size([2, 196, 3, 16, 16])
        >>> grid_size
        (14, 14)
    """
    if stride is None:
        stride = patch_size

    B, C, H, W = images.shape

    # 使用 unfold 提取 patches
    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # (B, C, num_patches_h, num_patches_w, patch_size, patch_size)

    num_patches_h = patches.size(2)
    num_patches_w = patches.size(3)

    # 重排维度
    patches = patches.contiguous().view(B, C, num_patches_h * num_patches_w, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    # (B, num_patches, C, patch_size, patch_size)

    return patches, (num_patches_h, num_patches_w)


class MultiInstanceLearning(nn.Module):
    """
    多实例学习模块

    将图像分成多个patches，使用注意力机制自动学习哪些patches重要。

    Args:
        feature_dim: 特征维度
        num_classes: 类别数
        patch_size: Patch 大小
        attention_dim: 注意力隐藏层维度
        pooling_mode: 池化模式
            - "attention": 注意力加权池化
            - "max": 最大池化
            - "mean": 平均池化

    Example:
        >>> mil = MultiInstanceLearning(
        ...     feature_dim=512,
        ...     num_classes=2,
        ...     patch_size=16,
        ... )
        >>>
        >>> images = torch.randn(2, 3, 224, 224)
        >>> outputs = mil(images)
        >>> outputs["logits"].shape
        torch.Size([2, 2])
        >>> outputs["attention_weights"].shape
        torch.Size([2, 196])
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        patch_size: int = 16,
        attention_dim: int = 128,
        pooling_mode: str = "attention",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.pooling_mode = pooling_mode

        # 注意力池化网络
        if pooling_mode == "attention":
            self.attention_net = nn.Sequential(
                nn.Linear(feature_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1),
            )

        # 实例级分类器
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        patch_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            patch_features: Patch 特征 (B, num_patches, feature_dim)

        Returns:
            字典包含:
                - logits: 分类logits (B, num_classes)
                - attention_weights: 注意力权重 (B, num_patches)
                - aggregated_features: 聚合后的特征 (B, feature_dim)
        """
        B, N, D = patch_features.shape

        if self.pooling_mode == "attention":
            # 计算注意力权重
            attention_scores = self.attention_net(patch_features)  # (B, N, 1)
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, N, 1)

            # 加权聚合
            aggregated = (patch_features * attention_weights).sum(dim=1)  # (B, D)
            attention_weights = attention_weights.squeeze(-1)  # (B, N)

        elif self.pooling_mode == "max":
            # 最大池化
            aggregated = patch_features.max(dim=1)[0]  # (B, D)
            attention_weights = torch.zeros(B, N, device=patch_features.device)

        elif self.pooling_mode == "mean":
            # 平均池化
            aggregated = patch_features.mean(dim=1)  # (B, D)
            attention_weights = torch.ones(B, N, device=patch_features.device) / N

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        # 分类
        logits = self.classifier(aggregated)  # (B, num_classes)

        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "aggregated_features": aggregated,
        }


class MILSupervision(BaseAttentionSupervision):
    """
    基于多实例学习的注意力监督

    使用 MIL 自动学习重要的图像区域，无需标注。

    Args:
        loss_weight: 注意力损失的权重
        patch_size: Patch 大小
        attention_dim: 注意力隐藏层维度
        diversity_weight: 多样性损失权重（鼓励关注多个区域）
        enabled: 是否启用

    Example:
        >>> supervision = MILSupervision(
        ...     loss_weight=0.1,
        ...     patch_size=16,
        ... )
        >>>
        >>> # 提取 patch 特征
        >>> patches, grid_size = extract_patches(images, patch_size=16)
        >>> patch_features = backbone(patches.flatten(0, 1))  # 批量处理
        >>> patch_features = patch_features.view(B, -1, feature_dim)
        >>>
        >>> # MIL 前向传播
        >>> mil_outputs = mil_module(patch_features)
        >>>
        >>> # 计算监督损失
        >>> loss_result = supervision(
        ...     attention_weights=mil_outputs["attention_weights"],
        ...     features=patch_features,
        ...     grid_size=grid_size,
        ... )
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        patch_size: int = 16,
        attention_dim: int = 128,
        diversity_weight: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)

        self.patch_size = patch_size
        self.attention_dim = attention_dim
        self.diversity_weight = diversity_weight

    def compute_diversity_loss(
        self,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算多样性损失

        鼓励注意力分布在多个区域，避免只关注一个patch。

        Args:
            attention_weights: 注意力权重 (B, num_patches)

        Returns:
            多样性损失
        """
        # 计算注意力的熵（熵越大，分布越均匀）
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1)

        # 损失：鼓励高熵（但不要太高）
        # 使用目标熵（如 log(num_patches) 的一半）
        num_patches = attention_weights.size(1)
        target_entropy = 0.5 * torch.log(torch.tensor(num_patches, dtype=torch.float32))

        diversity_loss = F.mse_loss(entropy, target_entropy.to(entropy.device))

        return diversity_loss

    def attention_to_map(
        self,
        attention_weights: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        将 patch 级别的注意力权重转换为空间注意力图

        Args:
            attention_weights: Patch 注意力权重 (B, num_patches)
            grid_size: Patch 网格尺寸 (num_patches_h, num_patches_w)

        Returns:
            注意力图 (B, H, W)
        """
        B = attention_weights.size(0)
        num_patches_h, num_patches_w = grid_size

        # 重塑为 2D 网格
        attention_map = attention_weights.view(B, num_patches_h, num_patches_w)

        return attention_map

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算注意力监督损失

        Args:
            attention_weights: Patch 注意力权重 (B, num_patches)
            features: Patch 特征 (B, num_patches, feature_dim)
            targets: 不使用
            grid_size: Patch 网格尺寸 (num_patches_h, num_patches_w)
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        components = {}

        # 多样性损失
        if self.diversity_weight > 0:
            diversity_loss = self.compute_diversity_loss(attention_weights)
            components["diversity"] = diversity_loss
            total_loss = self.diversity_weight * diversity_loss
        else:
            total_loss = torch.tensor(0.0, device=attention_weights.device)

        # 转换为空间注意力图（用于可视化）
        attention_map = None
        if grid_size is not None:
            attention_map = self.attention_to_map(attention_weights, grid_size)

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_map if attention_map is not None else attention_weights,
            metadata={
                "patch_attention": attention_weights,
                "grid_size": grid_size,
            },
        )


class AttentionMIL(nn.Module):
    """
    完整的注意力 MIL 模型

    结合 backbone、MIL 和注意力监督的完整模型。

    Args:
        backbone: 特征提取器
        feature_dim: 特征维度
        num_classes: 类别数
        patch_size: Patch 大小
        attention_dim: 注意力隐藏层维度
        pooling_mode: 池化模式

    Example:
        >>> from torchvision.models import resnet18
        >>> backbone = resnet18(pretrained=True)
        >>> backbone = nn.Sequential(*list(backbone.children())[:-2])  # 移除分类层
        >>>
        >>> model = AttentionMIL(
        ...     backbone=backbone,
        ...     feature_dim=512,
        ...     num_classes=2,
        ...     patch_size=16,
        ... )
        >>>
        >>> images = torch.randn(2, 3, 224, 224)
        >>> outputs = model(images)
        >>> outputs["logits"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        patch_size: int = 16,
        attention_dim: int = 128,
        pooling_mode: str = "attention",
    ):
        super().__init__()

        self.backbone = backbone
        self.patch_size = patch_size

        # MIL 模块
        self.mil = MultiInstanceLearning(
            feature_dim=feature_dim,
            num_classes=num_classes,
            patch_size=patch_size,
            attention_dim=attention_dim,
            pooling_mode=pooling_mode,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: 输入图像 (B, C, H, W)

        Returns:
            字典包含:
                - logits: 分类 logits (B, num_classes)
                - attention_weights: 注意力权重 (B, num_patches)
                - attention_map: 空间注意力图 (B, H, W)
                - patch_features: Patch 特征 (B, num_patches, feature_dim)
        """
        B, C, H, W = images.shape

        # 提取 patches
        patches, grid_size = extract_patches(images, self.patch_size)
        # (B, num_patches, C, patch_size, patch_size)

        num_patches = patches.size(1)

        # 批量提取特征
        patches_flat = patches.view(B * num_patches, C, self.patch_size, self.patch_size)
        patch_features_flat = self.backbone(patches_flat)  # (B*num_patches, feature_dim, h, w)

        # 全局平均池化
        patch_features_flat = F.adaptive_avg_pool2d(patch_features_flat, 1)
        patch_features_flat = patch_features_flat.flatten(1)  # (B*num_patches, feature_dim)

        # 重塑
        patch_features = patch_features_flat.view(B, num_patches, -1)
        # (B, num_patches, feature_dim)

        # MIL 聚合
        mil_outputs = self.mil(patch_features)

        # 转换为空间注意力图
        attention_map = mil_outputs["attention_weights"].view(B, grid_size[0], grid_size[1])

        return {
            "logits": mil_outputs["logits"],
            "attention_weights": mil_outputs["attention_weights"],
            "attention_map": attention_map,
            "patch_features": patch_features,
            "aggregated_features": mil_outputs["aggregated_features"],
            "grid_size": grid_size,
        }
