"""
基于分割掩码的注意力监督

使用数据集中的分割掩码来监督模型的注意力权重。
"""

from typing import Any

import torch
import torch.nn.functional as F

from med_core.attention_supervision.base import (
    AttentionLoss,
    AttentionSmoothLoss,
    BaseAttentionSupervision,
)


def mask_to_attention_target(
    mask: torch.Tensor,
    size: tuple[int, int] | None = None,
    temperature: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    将分割掩码转换为注意力目标分布

    Args:
        mask: 分割掩码 (B, H, W) 或 (B, 1, H, W)，值为 0 或 1
        size: 目标尺寸 (H, W)，如果为 None 则保持原尺寸
        temperature: 温度参数，用于增强对比度
        normalize: 是否归一化为概率分布

    Returns:
        注意力目标 (B, H, W)

    Example:
        >>> mask = torch.zeros(2, 256, 256)
        >>> mask[:, 100:150, 100:150] = 1  # 病灶区域
        >>> target = mask_to_attention_target(mask, size=(32, 32))
        >>> target.shape
        torch.Size([2, 32, 32])
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, H, W)

    # 调整尺寸
    if size is not None and mask.shape[-2:] != size:
        mask = F.interpolate(
            mask.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    mask = mask.squeeze(1)  # (B, H, W)

    if normalize:
        # 归一化为概率分布
        B, H, W = mask.shape
        mask_flat = mask.view(B, -1)

        # 应用温度参数增强对比度
        mask_flat = mask_flat * temperature

        # Softmax 归一化
        target = F.softmax(mask_flat, dim=1)
        target = target.view(B, H, W)
    else:
        target = mask

    return target


class MaskSupervisedAttention(BaseAttentionSupervision):
    """
    基于分割掩码的注意力监督

    使用数据集中的分割掩码来监督模型��注意力权重，
    让模型学会关注病灶区域。

    Args:
        loss_weight: 注意力损失的权重
        loss_type: 损失函数类型
            - "mse": 均方误差
            - "kl": KL散度
            - "bce": 二元交叉熵
        temperature: 温度参数，用于增强掩码对比度
        add_smooth_loss: 是否添加平滑损失
        smooth_weight: 平滑损失的权重
        enabled: 是否启用

    Example:
        >>> supervision = MaskSupervisedAttention(
        ...     loss_weight=0.1,
        ...     loss_type="kl",
        ... )
        >>>
        >>> # 训练时
        >>> attention = model.get_attention_weights(images)
        >>> loss_result = supervision(
        ...     attention_weights=attention,
        ...     features=features,
        ...     targets=segmentation_masks,
        ... )
        >>> total_loss = classification_loss + loss_result.total_loss
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        loss_type: str = "kl",
        temperature: float = 10.0,
        add_smooth_loss: bool = False,
        smooth_weight: float = 0.01,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)

        self.loss_type = loss_type
        self.temperature = temperature
        self.add_smooth_loss = add_smooth_loss
        self.smooth_weight = smooth_weight

        if add_smooth_loss:
            self.smooth_loss = AttentionSmoothLoss(method="tv")

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
            attention_weights: 模型的注意力权重 (B, H, W) 或 (B, 1, H, W)
            features: 特征图 (B, C, H, W)
            targets: 分割掩码 (B, H, W) 或 (B, 1, H, W)
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        if targets is None:
            # 没有掩码，返回零损失
            return AttentionLoss(
                total_loss=torch.tensor(0.0, device=attention_weights.device),
                components={},
                attention_weights=attention_weights,
            )

        # 确保维度正确
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.squeeze(1)  # (B, H, W)

        # 转换掩码为注意力目标
        attention_target = mask_to_attention_target(
            mask=targets,
            size=attention_weights.shape[-2:],
            temperature=self.temperature,
            normalize=(self.loss_type in ["kl", "mse"]),
        )

        # 归一化��意力权重
        if self.loss_type == "kl":
            attention_norm = self.normalize_attention(
                attention_weights, method="softmax"
            )
        elif self.loss_type == "bce":
            attention_norm = self.normalize_attention(
                attention_weights, method="sigmoid"
            )
        else:  # mse
            attention_norm = self.normalize_attention(
                attention_weights, method="minmax"
            )

        # 计算主损失
        if self.loss_type == "mse":
            main_loss = F.mse_loss(attention_norm, attention_target)

        elif self.loss_type == "kl":
            # KL散度：KL(target || pred)
            attention_norm_flat = attention_norm.view(attention_norm.size(0), -1)
            attention_target_flat = attention_target.view(attention_target.size(0), -1)

            kl_loss = F.kl_div(
                torch.log(attention_norm_flat + 1e-8),
                attention_target_flat,
                reduction="batchmean",
            )
            main_loss = kl_loss

        elif self.loss_type == "bce":
            # 二元交叉熵
            target_binary = (targets > 0.5).float()
            target_binary = self.resize_target(target_binary, attention_norm.shape[-2:])
            main_loss = F.binary_cross_entropy(attention_norm, target_binary)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 损失组件
        components = {"main": main_loss}
        total_loss = main_loss

        # 添加平滑损失（可选）
        if self.add_smooth_loss:
            smooth_loss = self.smooth_loss(attention_norm)
            components["smooth"] = smooth_loss
            total_loss = total_loss + self.smooth_weight * smooth_loss

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_norm,
            metadata={
                "target": attention_target,
                "loss_type": self.loss_type,
            },
        )


class BBoxSupervisedAttention(MaskSupervisedAttention):
    """
    基于边界框的注意力监督

    将边界框转换为掩码，然后使用掩码监督注意力。

    Args:
        loss_weight: 注意力损失的权重
        loss_type: 损失函数类型
        temperature: 温度参数
        bbox_format: 边界框格式
            - "xyxy": [x_min, y_min, x_max, y_max]
            - "xywh": [x, y, width, height]
            - "cxcywh": [center_x, center_y, width, height]
        enabled: 是否启用

    Example:
        >>> supervision = BBoxSupervisedAttention(
        ...     loss_weight=0.1,
        ...     bbox_format="xyxy",
        ... )
        >>>
        >>> # 边界框: [x_min, y_min, x_max, y_max]
        >>> bboxes = torch.tensor([
        ...     [100, 100, 200, 200],  # 第一个样本
        ...     [150, 150, 250, 250],  # 第二个样本
        ... ])
        >>>
        >>> loss_result = supervision(
        ...     attention_weights=attention,
        ...     features=features,
        ...     targets=bboxes,
        ...     image_size=(512, 512),
        ... )
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        loss_type: str = "kl",
        temperature: float = 10.0,
        bbox_format: str = "xyxy",
        enabled: bool = True,
    ):
        super().__init__(
            loss_weight=loss_weight,
            loss_type=loss_type,
            temperature=temperature,
            enabled=enabled,
        )
        self.bbox_format = bbox_format

    def bbox_to_mask(
        self,
        bboxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        将边界框转换为掩码

        Args:
            bboxes: 边界框 (B, 4)
            image_size: 图像尺寸 (H, W)

        Returns:
            掩码 (B, H, W)
        """
        B = bboxes.size(0)
        H, W = image_size
        device = bboxes.device

        masks = torch.zeros(B, H, W, device=device)

        for i in range(B):
            bbox = bboxes[i]

            # 转换为 xyxy 格式
            if self.bbox_format == "xyxy":
                x_min, y_min, x_max, y_max = bbox
            elif self.bbox_format == "xywh":
                x, y, w, h = bbox
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h
            elif self.bbox_format == "cxcywh":
                cx, cy, w, h = bbox
                x_min = cx - w / 2
                y_min = cy - h / 2
                x_max = cx + w / 2
                y_max = cy + h / 2
            else:
                raise ValueError(f"Unknown bbox format: {self.bbox_format}")

            # 转换为整数索引
            x_min = int(torch.clamp(x_min, 0, W - 1))
            y_min = int(torch.clamp(y_min, 0, H - 1))
            x_max = int(torch.clamp(x_max, 0, W - 1))
            y_max = int(torch.clamp(y_max, 0, H - 1))

            # 填充掩码
            masks[i, y_min : y_max + 1, x_min : x_max + 1] = 1.0

        return masks

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> AttentionLoss:
        """
        计算注意力监督损失

        Args:
            attention_weights: 注意力权重
            features: 特征图
            targets: 边界框 (B, 4)
            image_size: 图像尺寸 (H, W)
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        if targets is None:
            return AttentionLoss(
                total_loss=torch.tensor(0.0, device=attention_weights.device),
                components={},
                attention_weights=attention_weights,
            )

        # 推断图像尺寸
        if image_size is None:
            # 假设特征图是原图的下采样
            _, _, fH, fW = features.shape
            # 通常是 16 倍下采样（如 ResNet）
            image_size = (fH * 16, fW * 16)

        # 转换边界框为掩码
        masks = self.bbox_to_mask(targets, image_size)

        # 使用父类的掩码监督
        return super().compute_attention_loss(
            attention_weights=attention_weights,
            features=features,
            targets=masks,
            **kwargs: Any,
        )


class KeypointSupervisedAttention(MaskSupervisedAttention):
    """
    基于关键点的注意力监督

    将关键点转换为高斯掩码，然后使用掩码监督注意力。

    Args:
        loss_weight: 注意力损失的权重
        loss_type: 损失函数类型
        temperature: 温度参数
        gaussian_sigma: 高斯核的标准差（像素）
        enabled: 是否启用

    Example:
        >>> supervision = KeypointSupervisedAttention(
        ...     loss_weight=0.1,
        ...     gaussian_sigma=20,
        ... )
        >>>
        >>> # 关键点: [x, y]
        >>> keypoints = torch.tensor([
        ...     [150, 150],  # 第一个样本的病灶中心
        ...     [200, 200],  # 第二个样本的病灶中心
        ... ])
        >>>
        >>> loss_result = supervision(
        ...     attention_weights=attention,
        ...     features=features,
        ...     targets=keypoints,
        ...     image_size=(512, 512),
        ... )
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        loss_type: str = "kl",
        temperature: float = 10.0,
        gaussian_sigma: float = 20.0,
        enabled: bool = True,
    ):
        super().__init__(
            loss_weight=loss_weight,
            loss_type=loss_type,
            temperature=temperature,
            enabled=enabled,
        )
        self.gaussian_sigma = gaussian_sigma

    def keypoint_to_mask(
        self,
        keypoints: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        将关键点转换为高斯掩码

        Args:
            keypoints: 关键点 (B, 2) 或 (B, N, 2)，格式为 [x, y]
            image_size: 图像尺寸 (H, W)

        Returns:
            掩码 (B, H, W)
        """
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(1)  # (B, 1, 2)

        B, N, _ = keypoints.shape
        H, W = image_size
        device = keypoints.device

        # 创建坐标网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        y_grid = y_grid.float()
        x_grid = x_grid.float()

        masks = torch.zeros(B, H, W, device=device)

        for i in range(B):
            for j in range(N):
                kp = keypoints[i, j]  # (2,)
                x, y = kp[0], kp[1]

                # 跳过无效关键点
                if x < 0 or y < 0:
                    continue

                # 计算高斯分布
                distance_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
                gaussian = torch.exp(-distance_sq / (2 * self.gaussian_sigma**2))

                # 累加（支持多个关键点）
                masks[i] = torch.maximum(masks[i], gaussian)

        return masks

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> AttentionLoss:
        """
        计算注意力监督损失

        Args:
            attention_weights: 注意力权重
            features: 特征图
            targets: 关键点 (B, 2) 或 (B, N, 2)
            image_size: 图像尺寸 (H, W)
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        if targets is None:
            return AttentionLoss(
                total_loss=torch.tensor(0.0, device=attention_weights.device),
                components={},
                attention_weights=attention_weights,
            )

        # 推断图像尺寸
        if image_size is None:
            _, _, fH, fW = features.shape
            image_size = (fH * 16, fW * 16)

        # 转换关键点为掩码
        masks = self.keypoint_to_mask(targets, image_size)

        # 使用父类的掩码监督
        return super().compute_attention_loss(
            attention_weights=attention_weights,
            features=features,
            targets=masks,
            **kwargs: Any,
        )
