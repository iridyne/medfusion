"""
基于类激活图（CAM）的自监督注意力

使用类激活图来自监督注意力，无需分割掩码标注。
"""

import torch
import torch.nn.functional as F

from med_core.attention_supervision.base import (
    AttentionConsistencyLoss,
    AttentionLoss,
    BaseAttentionSupervision,
)


def generate_cam(
    feature_maps: torch.Tensor,
    classifier_weights: torch.Tensor,
    predicted_class: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    生成类激活图（Class Activation Map）

    Args:
        feature_maps: 特征图 (B, C, H, W)
        classifier_weights: 分类器权重 (num_classes, C)
        predicted_class: 预测的类别 (B,)，如果为 None 则使用最高激活的类别

    Returns:
        CAM (B, H, W)

    Example:
        >>> features = torch.randn(2, 512, 16, 16)
        >>> classifier = nn.Linear(512, 2)
        >>> cam = generate_cam(features, classifier.weight)
        >>> cam.shape
        torch.Size([2, 16, 16])
    """
    B, C, H, W = feature_maps.shape

    if predicted_class is None:
        # 计算每个类别的全局平均池化
        pooled = (
            F.adaptive_avg_pool2d(feature_maps, 1).squeeze(-1).squeeze(-1)
        )  # (B, C)
        logits = F.linear(pooled, classifier_weights)  # (B, num_classes)
        predicted_class = logits.argmax(dim=1)  # (B,)

    cam = torch.zeros(B, H, W, device=feature_maps.device)

    for i in range(B):
        # 获取预测类别的权重
        class_weights = classifier_weights[predicted_class[i]]  # (C,)

        # 加权求和特征图
        cam[i] = (class_weights.view(C, 1, 1) * feature_maps[i]).sum(0)

    # ReLU 激活（只保留正激活）
    cam = F.relu(cam)

    # 归一化到 [0, 1]
    for i in range(B):
        max_val = cam[i].max()
        if max_val > 0:
            cam[i] = cam[i] / max_val

    return cam


class CAMSelfSupervision(BaseAttentionSupervision):
    """
    基于 CAM 的自监督注意力

    使用类激活图（CAM）来自监督注意力权重，让模型关注判别性区域。
    无需分割掩码标注。

    Args:
        loss_weight: 注意力损失的权重
        consistency_method: 一致性度量方法 ("entropy", "variance", "gini")
        consistency_weight: 一致性损失的权重
        alignment_weight: 对齐损失的权重（注意力与CAM的对齐）
        cam_threshold: CAM 阈值，用于生成二值掩码
        enabled: 是否启用

    Example:
        >>> supervision = CAMSelfSupervision(
        ...     loss_weight=0.1,
        ...     consistency_method="entropy",
        ... )
        >>>
        >>> # 训练时
        >>> features = backbone(images)  # (B, C, H, W)
        >>> attention = attention_module(features)  # (B, 1, H, W)
        >>>
        >>> loss_result = supervision(
        ...     attention_weights=attention,
        ...     features=features,
        ...     classifier_weights=model.classifier.weight,
        ...     predicted_class=predictions.argmax(dim=1),
        ... )
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        consistency_method: str = "entropy",
        consistency_weight: float = 1.0,
        alignment_weight: float = 0.5,
        cam_threshold: float = 0.5,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)

        self.consistency_weight = consistency_weight
        self.alignment_weight = alignment_weight
        self.cam_threshold = cam_threshold

        # 一致性损失
        self.consistency_loss = AttentionConsistencyLoss(method=consistency_method)

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        classifier_weights: torch.Tensor | None = None,
        predicted_class: torch.Tensor | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算注意力自监督损失

        Args:
            attention_weights: 注意力权重 (B, H, W) 或 (B, 1, H, W)
            features: 特征图 (B, C, H, W)
            targets: 不使用（为了接口一致性）
            classifier_weights: 分类器权重 (num_classes, C)
            predicted_class: 预测的类别 (B,)
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        if classifier_weights is None:
            raise ValueError("classifier_weights is required for CAM supervision")

        # 确保维度正确
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.squeeze(1)  # (B, H, W)

        # 归一化注意力权重
        attention_norm = self.normalize_attention(attention_weights, method="softmax")

        # 生成 CAM
        cam = generate_cam(
            feature_maps=features,
            classifier_weights=classifier_weights,
            predicted_class=predicted_class,
        )

        # 调整 CAM 尺寸以匹配注意力
        if cam.shape[-2:] != attention_norm.shape[-2:]:
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=attention_norm.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        components = {}

        # 1. 一致性损失：鼓励注意力集中
        consistency_loss = self.consistency_loss(attention_norm)
        components["consistency"] = consistency_loss

        # 2. 对齐损失：注意力与 CAM 对齐
        if self.alignment_weight > 0:
            # 将 CAM 归一化为概率分布
            cam_flat = cam.view(cam.size(0), -1)
            cam_norm = F.softmax(cam_flat * 10, dim=1)  # 增强对比度
            cam_norm = cam_norm.view_as(cam)

            # KL 散度
            attention_flat = attention_norm.view(attention_norm.size(0), -1)
            cam_flat_norm = cam_norm.view(cam_norm.size(0), -1)

            alignment_loss = F.kl_div(
                torch.log(attention_flat + 1e-8),
                cam_flat_norm,
                reduction="batchmean",
            )
            components["alignment"] = alignment_loss
        else:
            alignment_loss = torch.tensor(0.0, device=attention_weights.device)

        # 总损失
        total_loss = (
            self.consistency_weight * consistency_loss
            + self.alignment_weight * alignment_loss
        )

        return AttentionLoss(
            total_loss=total_loss,
            components=components,
            attention_weights=attention_norm,
            metadata={
                "cam": cam,
                "cam_binary": (cam > self.cam_threshold).float(),
            },
        )


class GradCAMSupervision(BaseAttentionSupervision):
    """
    基于 Grad-CAM 的注意力监督

    使用 Grad-CAM 来监督注意力权重。Grad-CAM 考虑了梯度信息，
    比普通 CAM 更准确。

    Args:
        loss_weight: 注意力损失的权重
        alignment_weight: 对齐损失的权重
        enabled: 是否启用

    Example:
        >>> supervision = GradCAMSupervision(loss_weight=0.1)
        >>>
        >>> # 需要在反向传播前计算
        >>> features = backbone(images)
        >>> features.retain_grad()  # 保留梯度
        >>>
        >>> logits = classifier(features)
        >>> loss = criterion(logits, labels)
        >>> loss.backward(retain_graph=True)
        >>>
        >>> # 现在可以计算 Grad-CAM
        >>> loss_result = supervision(
        ...     attention_weights=attention,
        ...     features=features,
        ...     feature_gradients=features.grad,
        ... )
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        alignment_weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(loss_weight=loss_weight, enabled=enabled)
        self.alignment_weight = alignment_weight

    def generate_gradcam(
        self,
        feature_maps: torch.Tensor,
        feature_gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成 Grad-CAM

        Args:
            feature_maps: 特征图 (B, C, H, W)
            feature_gradients: 特征图的梯度 (B, C, H, W)

        Returns:
            Grad-CAM (B, H, W)
        """
        # 全局平均池化梯度，得到权重
        weights = feature_gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # 加权求和
        gradcam = (weights * feature_maps).sum(dim=1)  # (B, H, W)

        # ReLU 激活
        gradcam = F.relu(gradcam)

        # 归一化
        for i in range(gradcam.size(0)):
            max_val = gradcam[i].max()
            if max_val > 0:
                gradcam[i] = gradcam[i] / max_val

        return gradcam

    def compute_attention_loss(
        self,
        attention_weights: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        feature_gradients: torch.Tensor | None = None,
        **kwargs,
    ) -> AttentionLoss:
        """
        计算注意力监督损失

        Args:
            attention_weights: 注意力权重 (B, H, W) 或 (B, 1, H, W)
            features: 特征图 (B, C, H, W)
            targets: 不使用
            feature_gradients: 特征图的梯度 (B, C, H, W)
            **kwargs: 额外参数

        Returns:
            AttentionLoss: 损失对象
        """
        if feature_gradients is None:
            raise ValueError("feature_gradients is required for Grad-CAM supervision")

        # 确保维度正确
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.squeeze(1)

        # 归一化注意力
        attention_norm = self.normalize_attention(attention_weights, method="softmax")

        # 生成 Grad-CAM
        gradcam = self.generate_gradcam(features, feature_gradients)

        # 调整尺寸
        if gradcam.shape[-2:] != attention_norm.shape[-2:]:
            gradcam = F.interpolate(
                gradcam.unsqueeze(1),
                size=attention_norm.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # 归一化 Grad-CAM
        gradcam_flat = gradcam.view(gradcam.size(0), -1)
        gradcam_norm = F.softmax(gradcam_flat * 10, dim=1)
        gradcam_norm = gradcam_norm.view_as(gradcam)

        # 对齐损失
        attention_flat = attention_norm.view(attention_norm.size(0), -1)
        gradcam_flat_norm = gradcam_norm.view(gradcam_norm.size(0), -1)

        alignment_loss = F.kl_div(
            torch.log(attention_flat + 1e-8),
            gradcam_flat_norm,
            reduction="batchmean",
        )

        total_loss = self.alignment_weight * alignment_loss

        return AttentionLoss(
            total_loss=total_loss,
            components={"alignment": alignment_loss},
            attention_weights=attention_norm,
            metadata={"gradcam": gradcam},
        )
