"""
注意力监督配置

⚠️ DEPRECATED: 此模块已弃用，将在未来版本中移除。
请使用 `med_core.configs.ExperimentConfig` 替代，它现在包含了所有注意力监督配置选项。

迁移示例:
    # 旧方法（已弃用）:
    from med_core.configs.attention_config import ExperimentConfigWithAttention
    config = ExperimentConfigWithAttention(...)

    # 新方法（推荐）:
    from med_core.configs import ExperimentConfig
    config = ExperimentConfig()
    config.training.use_attention_supervision = True
    config.training.attention_supervision_method = "mask"

此模块仅为向后兼容而保留。
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

# 发出弃用警告
warnings.warn(
    "med_core.configs.attention_config is deprecated. "
    "Use med_core.configs.ExperimentConfig instead. "
    "This module will be removed in version 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class AttentionSupervisionConfig:
    """
    注意力监督配置

    Args:
        enabled: 是否启用注意力监督
        method: 监督方法
            - "mask": 基于分割掩码监督
            - "bbox": 基于边界框监督
            - "keypoint": 基于关键点监督
            - "cam": 基于 CAM 自监督
            - "gradcam": 基于 Grad-CAM 监督
            - "mil": 基于多实例学习
        loss_weight: 注意力损失的权重
        loss_type: 损失函数类型（用于掩码监督）
            - "mse": 均方误差
            - "kl": KL散度
            - "bce": 二元交叉熵
        temperature: 温度参数（用于掩码监督）
        add_smooth_loss: 是否添加平滑损失
        smooth_weight: 平滑损失权重
        consistency_method: 一致性度量方法（用于 CAM）
            - "entropy": 熵
            - "variance": 方差
            - "gini": 基尼系数
        consistency_weight: 一致性损失权重
        alignment_weight: 对齐损失权重
        cam_threshold: CAM 阈值
        patch_size: Patch 大小（用于 MIL）
        attention_dim: 注意力隐藏层维度（用于 MIL）
        diversity_weight: 多样性损失权重（用于 MIL）
        bbox_format: 边界框格式
            - "xyxy": [x_min, y_min, x_max, y_max]
            - "xywh": [x, y, width, height]
            - "cxcywh": [center_x, center_y, width, height]
        gaussian_sigma: 高斯核标准差（用于关键点）

    Example:
        >>> # 使用分割掩码监督
        >>> config = AttentionSupervisionConfig(
        ...     enabled=True,
        ...     method="mask",
        ...     loss_weight=0.1,
        ...     loss_type="kl",
        ... )
        >>>
        >>> # 使用 CAM 自监督
        >>> config = AttentionSupervisionConfig(
        ...     enabled=True,
        ...     method="cam",
        ...     loss_weight=0.1,
        ...     consistency_method="entropy",
        ... )
    """

    # 基础配置
    enabled: bool = False
    method: Literal["mask", "bbox", "keypoint", "cam", "gradcam", "mil"] = "mask"
    loss_weight: float = 0.1

    # 掩码监督配置
    loss_type: Literal["mse", "kl", "bce"] = "kl"
    temperature: float = 10.0
    add_smooth_loss: bool = False
    smooth_weight: float = 0.01

    # CAM 监督配置
    consistency_method: Literal["entropy", "variance", "gini"] = "entropy"
    consistency_weight: float = 1.0
    alignment_weight: float = 0.5
    cam_threshold: float = 0.5

    # MIL 配置
    patch_size: int = 16
    attention_dim: int = 128
    diversity_weight: float = 0.1

    # 边界框配置
    bbox_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy"

    # 关键点配置
    gaussian_sigma: float = 20.0


@dataclass
class DataConfigWithMask:
    """
    扩展的数据配置，支持掩码加载

    Args:
        data_dir: 数据目录
        csv_file: CSV 文件路径
        image_dir: 图像目录
        mask_dir: 掩码目录（可选）
        image_col: 图像路径列名
        mask_col: 掩码路径列名
        label_col: 标签列名
        tabular_cols: 表格特征列名列表
        image_format: 图像格式
        train_split: 训练集比例
        val_split: 验证集比例
        test_split: 测试集比例
        random_seed: 随机种子
        return_mask: 是否返回掩码
        return_bbox: 是否返回边界框
        return_keypoint: 是否返回关键点

    Example:
        >>> config = DataConfigWithMask(
        ...     data_dir="data/",
        ...     csv_file="annotations.csv",
        ...     image_dir="images/",
        ...     mask_dir="masks/",
        ...     return_mask=True,
        ... )
    """

    # 基础路径
    data_dir: str = "data"
    csv_file: str = "data.csv"
    image_dir: str = "images"
    mask_dir: str | None = None

    # 列名
    image_col: str = "image_path"
    mask_col: str = "mask_path"
    label_col: str = "label"
    tabular_cols: list[str] | None = None

    # 图像格式
    image_format: Literal["dicom", "nifti", "png", "jpg"] = "dicom"

    # 数据划分
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # 返回选项
    return_mask: bool = False
    return_bbox: bool = False
    return_keypoint: bool = False


@dataclass
class TrainingConfigWithAttention:
    """
    扩展的训练配置，支持注意力监督

    Args:
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        optimizer: 优化器类型
        scheduler: 学习率调度器类型
        weight_decay: 权重衰减
        gradient_clip: 梯度裁剪阈值
        mixed_precision: 是否使用混合精度训练
        attention_supervision: 注意力监督配置
        log_attention_every: 每隔多少步记录注意力可视化
        save_attention_maps: 是否保存注意力图

    Example:
        >>> config = TrainingConfigWithAttention(
        ...     num_epochs=100,
        ...     batch_size=32,
        ...     attention_supervision=AttentionSupervisionConfig(
        ...         enabled=True,
        ...         method="mask",
        ...         loss_weight=0.1,
        ...     ),
        ... )
    """

    # 基础训练配置
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: Literal["cosine", "step", "plateau", "none"] = "cosine"
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    # 注意力监督配置
    attention_supervision: AttentionSupervisionConfig = field(
        default_factory=AttentionSupervisionConfig
    )

    # 注意力可视化配置
    log_attention_every: int = 100
    save_attention_maps: bool = True


@dataclass
class ExperimentConfigWithAttention:
    """
    完整的实验配置，包含注意力监督

    Args:
        experiment_name: 实验名称
        output_dir: 输出目录
        data: 数据配置
        training: 训练配置
        device: 设备
        num_workers: 数据加载器工作进程数
        pin_memory: 是否固定内存
        deterministic: 是否使用确定性算法

    Example:
        >>> config = ExperimentConfigWithAttention(
        ...     experiment_name="pneumonia_detection_with_attention",
        ...     data=DataConfigWithMask(
        ...         csv_file="pneumonia.csv",
        ...         mask_dir="lesion_masks/",
        ...         return_mask=True,
        ...     ),
        ...     training=TrainingConfigWithAttention(
        ...         attention_supervision=AttentionSupervisionConfig(
        ...             enabled=True,
        ...             method="mask",
        ...             loss_weight=0.1,
        ...         ),
        ...     ),
        ... )
    """

    # 实验基础信息
    experiment_name: str = "experiment"
    output_dir: str = "outputs"

    # 子配置
    data: DataConfigWithMask = field(default_factory=DataConfigWithMask)
    training: TrainingConfigWithAttention = field(
        default_factory=TrainingConfigWithAttention
    )

    # 硬件配置
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    deterministic: bool = False


# 预设配置工厂函数


def create_mask_supervised_config(
    loss_weight: float = 0.1,
    loss_type: Literal["mse", "kl", "bce"] = "kl",
) -> AttentionSupervisionConfig:
    """
    创建分割掩码监督配置

    Args:
        loss_weight: 损失权重
        loss_type: 损失类型

    Returns:
        配置对象

    Example:
        >>> config = create_mask_supervised_config(
        ...     loss_weight=0.1,
        ...     loss_type="kl",
        ... )
    """
    return AttentionSupervisionConfig(
        enabled=True,
        method="mask",
        loss_weight=loss_weight,
        loss_type=loss_type,
        temperature=10.0,
        add_smooth_loss=False,
    )


def create_cam_supervised_config(
    loss_weight: float = 0.1,
    consistency_method: Literal["entropy", "variance", "gini"] = "entropy",
) -> AttentionSupervisionConfig:
    """
    创建 CAM 自监督配置

    Args:
        loss_weight: 损失权重
        consistency_method: 一致性度量方法

    Returns:
        配置对象

    Example:
        >>> config = create_cam_supervised_config(
        ...     loss_weight=0.1,
        ...     consistency_method="entropy",
        ... )
    """
    return AttentionSupervisionConfig(
        enabled=True,
        method="cam",
        loss_weight=loss_weight,
        consistency_method=consistency_method,
        consistency_weight=1.0,
        alignment_weight=0.5,
    )


def create_mil_config(
    loss_weight: float = 0.1,
    patch_size: int = 16,
) -> AttentionSupervisionConfig:
    """
    创建多实例学习配置

    Args:
        loss_weight: 损失权重
        patch_size: Patch 大小

    Returns:
        配置对象

    Example:
        >>> config = create_mil_config(
        ...     loss_weight=0.1,
        ...     patch_size=16,
        ... )
    """
    return AttentionSupervisionConfig(
        enabled=True,
        method="mil",
        loss_weight=loss_weight,
        patch_size=patch_size,
        attention_dim=128,
        diversity_weight=0.1,
    )


def create_bbox_supervised_config(
    loss_weight: float = 0.1,
    bbox_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
) -> AttentionSupervisionConfig:
    """
    创建边界框监督配置

    Args:
        loss_weight: 损失权重
        bbox_format: 边界框格式

    Returns:
        配置对象

    Example:
        >>> config = create_bbox_supervised_config(
        ...     loss_weight=0.1,
        ...     bbox_format="xyxy",
        ... )
    """
    return AttentionSupervisionConfig(
        enabled=True,
        method="bbox",
        loss_weight=loss_weight,
        loss_type="kl",
        bbox_format=bbox_format,
    )
