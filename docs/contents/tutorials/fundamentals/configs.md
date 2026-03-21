# 配置文件详解

**预计时间：20分钟**

本教程详细讲解 MedFusion 的配置系统，帮助你理解每个参数的含义和最佳实践。

## 配置文件结构

MedFusion 使用 YAML 格式的配置文件，主要包含五个部分：

```yaml
# 1. 实验元数据
project_name: "medical-multimodal"
experiment_name: "resnet18_mlp_gated_v1"
seed: 42

# 2. 数据配置 (data)
# 3. 模型配置 (model)
# 4. 训练配置 (training)
# 5. 日志配置 (logging)
```

## 1. 实验元数据

### 基本参数

```yaml
project_name: "medical-multimodal"
experiment_name: "resnet18_mlp_gated_v1"
description: "使用 ResNet18 和 MLP 配合门控融合的基线实验"
tags: ["baseline", "resnet", "multimodal"]
```

**参数说明：**
- `project_name`: 项目名称，用于组织多个实验
- `experiment_name`: 实验名称，建议包含模型架构信息
- `description`: 实验描述（可选）
- `tags`: 标签列表，便于筛选和管理（可选）

### 全局设置

```yaml
seed: 42
deterministic: true
device: "auto"  # "auto", "cuda", "cpu", "mps"
```

**参数说明：**
- `seed`: 随机种子，确保实验可复现（默认：42）
- `deterministic`: 是否使用确定性算法（默认：true）
- `device`: 计算设备
  - `"auto"`: 自动检测（优先级：CUDA > MPS > CPU）
  - `"cuda"`: 使用 NVIDIA GPU
  - `"cpu"`: 使用 CPU
  - `"mps"`: 使用 Apple Silicon GPU

## 2. 数据配置 (data)

### 路径配置

```yaml
data:
  data_root: "data"
  csv_path: "data/mock/metadata.csv"
  image_dir: "data/mock"
```

**参数说明：**
- `data_root`: 数据根目录（默认：`"data"`）
- `csv_path`: CSV 元数据文件路径（必需）
- `image_dir`: 图像文件根目录（必需）

### 列映射

```yaml
data:
  image_path_column: "image_path"
  target_column: "diagnosis"
  patient_id_column: "patient_id"
```

**参数说明：**
- `image_path_column`: CSV 中图像路径列名（默认：`"image_path"`）
- `target_column`: 标签列名（默认：`"label"`）
- `patient_id_column`: 患者 ID 列名（可选，用于患者级别划分）

### 特征选择

```yaml
data:
  numerical_features:
    - "age"
    - "bmi"
    - "blood_pressure"
  categorical_features:
    - "gender"
    - "smoking_status"
```

**参数说明：**
- `numerical_features`: 数值型特征列表
- `categorical_features`: 类别型特征列表

**注意事项：**
- 列名必须与 CSV 文件中的列名完全匹配
- 类别型特征会自动进行 one-hot 编码
- 数值型特征会自动进行标准化

### 数据划分

```yaml
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
```

**参数说明：**
- `train_ratio`: 训练集比例（默认：0.7）
- `val_ratio`: 验证集比例（默认：0.15）
- `test_ratio`: 测试集比例（默认：0.15）
- `random_seed`: 划分随机种子（默认：42）

**有效范围：**
- 三个比例之和必须等于 1.0
- 每个比例范围：0.0 ~ 1.0

### 图像处理

```yaml
data:
  image_size: 224
  image_channels: 3
  image_view: "default"  # "coronal", "axial", "sagittal"
```

**参数说明：**
- `image_size`: 图像尺寸（默认：224）
  - 常用值：224（ResNet）、384（ViT）、512（高分辨率）
- `image_channels`: 图像通道数（默认：3）
  - RGB 图像：3
  - 灰度图像：1
  - CT/MRI：1 或多通道
- `image_view`: 图像视图类型（可选）

### 数据加载器

```yaml
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
```

**参数说明：**
- `batch_size`: 批次大小（默认：16）
  - 建议根据 GPU 显存调整：8GB → 16-32，16GB → 32-64
- `num_workers`: 数据加载线程数（默认：4）
  - 建议值：CPU 核心数的 1/2 到 1/4
  - 设为 0 可避免多进程问题
- `pin_memory`: 是否使用锁页内存（默认：true）
  - GPU 训练时建议开启，可加速数据传输

### 数据增强

```yaml
data:
  use_augmentation: true
  augmentation_strength: "medium"  # "light", "medium", "heavy"
```

**参数说明：**
- `use_augmentation`: 是否使用数据增强（默认：true）
- `augmentation_strength`: 增强强度
  - `"light"`: 轻度增强（旋转 ±10°，轻微缩放）
  - `"medium"`: 中度增强（旋转 ±15°，中等缩放和翻转）
  - `"heavy"`: 重度增强（旋转 ±30°，强烈变换）

## 3. 模型配置 (model)

### 基本设置

```yaml
model:
  num_classes: 2
  use_auxiliary_heads: true
```

**参数说明：**
- `num_classes`: 分类类别数（必需）
- `use_auxiliary_heads`: 是否为各模态添加辅助分类器（默认：true）
  - 辅助分类器可提供额外的监督信号

### 视觉骨干网络 (vision)

```yaml
model:
  vision:
    backbone: "resnet18"
    pretrained: true
    freeze_backbone: true
    freeze_strategy: "progressive"
    unfreeze_last_n_layers: 2
    feature_dim: 128
    dropout: 0.3
    attention_type: "cbam"
    enable_attention_supervision: false
```

**参数说明：**

**backbone**: 骨干网络类型
- 2D 网络：`resnet18`, `resnet34`, `resnet50`, `resnet101`, `efficientnet_b0`-`b7`, `vit_b_16`, `swin_t`, `swin_s`
- 3D 网络：`resnet3d18`, `swin3d_tiny`, `swin3d_small`

**pretrained**: 是否使用预训练权重（默认：true）
- ImageNet 预训练可显著提升性能

**freeze_backbone**: 是否冻结骨干网络（默认：true）
- 小数据集建议冻结，大数据集可解冻微调

**freeze_strategy**: 冻结策略
- `"full"`: 完全冻结
- `"partial"`: 部分冻结（冻结前 N 层）
- `"progressive"`: 渐进式解冻（推荐）
- `"none"`: 不冻结

**unfreeze_last_n_layers**: 解冻最后 N 层（默认：2）

**feature_dim**: 特征维度（默认：128）
- 建议范围：64-512

**dropout**: Dropout 比例（默认：0.3）
- 范围：0.0-0.5

**attention_type**: 注意力机制类型
- `"cbam"`: CBAM（通道+空间注意力）
- `"se"`: Squeeze-and-Excitation
- `"eca"`: Efficient Channel Attention
- `"none"`: 不使用注意力

**enable_attention_supervision**: 是否启用注意力监督（默认：false）

### 表格骨干网络 (tabular)

```yaml
model:
  tabular:
    hidden_dims: [64, 64]
    output_dim: 32
    dropout: 0.2
    use_batch_norm: true
    activation: "relu"
```

**参数说明：**
- `hidden_dims`: 隐藏层维度列表（默认：`[64, 64]`）
  - 例如：`[128, 64, 32]` 表示 3 层 MLP
- `output_dim`: 输出特征维度（默认：32）
- `dropout`: Dropout 比例（默认：0.2）
- `use_batch_norm`: 是否使用 Batch Normalization（默认：true）
- `activation`: 激活函数
  - `"relu"`: ReLU（默认）
  - `"gelu"`: GELU
  - `"silu"`: SiLU/Swish

### 融合模块 (fusion)

```yaml
model:
  fusion:
    fusion_type: "gated"
    hidden_dim: 96
    dropout: 0.4
    num_heads: 4
    initial_image_weight: 0.3
    initial_tabular_weight: 0.7
    learnable_weights: true
```

**参数说明：**

**fusion_type**: 融合策略
- `"concatenate"`: 简单拼接
- `"gated"`: 门控融合（推荐）
- `"attention"`: 注意力融合
- `"cross_attention"`: 交叉注意力
- `"bilinear"`: 双线性融合
- `"kronecker"`: Kronecker 积融合
- `"fused_attention"`: 融合注意力（SMuRF 使用）

**hidden_dim**: 融合层隐藏维度（默认：96）

**dropout**: Dropout 比例（默认：0.4）

**num_heads**: 注意力头数（仅用于注意力融合，默认：4）

**initial_image_weight**: 图像模态初始权重（默认：0.3）

**initial_tabular_weight**: 表格模态初始权重（默认：0.7）

**learnable_weights**: 权重是否可学习（默认：true）

## 4. 训练配置 (training)

### 基本训练参数

```yaml
training:
  num_epochs: 50
  mixed_precision: true
  gradient_clip: 1.0
  accumulation_steps: 1
  label_smoothing: 0.1
  class_weights: null
```

**参数说明：**
- `num_epochs`: 训练轮数（默认：100）
- `mixed_precision`: 是否使用混合精度训练（默认：true）
  - 可节省显存并加速训练
- `gradient_clip`: 梯度裁剪阈值（默认：1.0）
  - 设为 `null` 禁用梯度裁剪
- `accumulation_steps`: 梯度累积步数（默认：1）
  - 用于模拟更大的 batch size
- `label_smoothing`: 标签平滑系数（默认：0.1）
  - 范围：0.0-0.2
- `class_weights`: 类别权重（可选）
  - 例如：`[1.0, 2.0]` 表示第二类权重为第一类的 2 倍

### 渐进式训练

```yaml
training:
  use_progressive_training: true
  stage1_epochs: 10
  stage2_epochs: 20
  stage3_epochs: 20
```

**参数说明：**
- `use_progressive_training`: 是否使用渐进式训练（默认：true）
- `stage1_epochs`: 阶段 1 轮数（训练单个流）
- `stage2_epochs`: 阶段 2 轮数（完整微调）
- `stage3_epochs`: 阶段 3 轮数（仅微调融合层）

**训练策略：**
1. 阶段 1：冻结其他部分，训练单个模态
2. 阶段 2：解冻所有层，端到端微调
3. 阶段 3：冻结骨干网络，仅微调融合层

### 早停机制

```yaml
training:
  early_stopping: true
  patience: 15
  min_delta: 0.001
  monitor: "val_auc"
  mode: "max"
```

**参数说明：**
- `early_stopping`: 是否启用早停（默认：true）
- `patience`: 容忍轮数（默认：20）
- `min_delta`: 最小改善阈值（默认：0.001）
- `monitor`: 监控指标
  - `"val_loss"`: 验证损失
  - `"val_auc"`: 验证 AUC（推荐）
  - `"val_acc"`: 验证准确率
- `mode`: 优化方向
  - `"min"`: 越小越好（用于 loss）
  - `"max"`: 越大越好（用于 AUC/ACC）

### 检查点保存

```yaml
training:
  save_top_k: 3
  save_last: true
```

**参数说明：**
- `save_top_k`: 保存最佳的 K 个检查点（默认：3）
- `save_last`: 是否保存最后一个检查点（默认：true）

### 优化器配置

```yaml
training:
  optimizer:
    optimizer: "adamw"
    learning_rate: 1.0e-4
    weight_decay: 0.01
    momentum: 0.9
    use_differential_lr: true
    lr_backbone: 1.0e-5
    lr_tabular: 1.0e-3
    lr_fusion: 5.0e-5
    lr_classifier: 1.0e-4
```

**参数说明：**

**optimizer**: 优化器类型
- `"adam"`: Adam
- `"adamw"`: AdamW（推荐）
- `"sgd"`: SGD with momentum

**learning_rate**: 基础学习率（默认：1e-4）

**weight_decay**: 权重衰减（默认：0.01）

**momentum**: 动量系数（仅用于 SGD，默认：0.9）

**use_differential_lr**: 是否使用差异化学习率（默认：true）
- 不同组件使用不同学习率

**差异化学习率：**
- `lr_backbone`: 骨干网络学习率（默认：1e-5）
- `lr_tabular`: 表格网络学习率（默认：1e-4）
- `lr_fusion`: 融合层学习率（默认：5e-5）
- `lr_classifier`: 分类器学习率（默认：1e-4）

### 学习率调度器

```yaml
training:
  scheduler:
    scheduler: "cosine"
    warmup_epochs: 5
    min_lr: 1.0e-6
    step_size: 10
    gamma: 0.1
    patience: 5
    factor: 0.5
```

**参数说明：**

**scheduler**: 调度器类型
- `"cosine"`: 余弦退火（推荐）
- `"step"`: 阶梯式衰减
- `"plateau"`: 自适应衰减
- `"onecycle"`: One Cycle 策略
- `"none"`: 不使用调度器

**warmup_epochs**: 预热轮数（默认：5）

**min_lr**: 最小学习率（默认：1e-7）

**StepLR 参数：**
- `step_size`: 衰减步长（默认：10）
- `gamma`: 衰减系数（默认：0.1）

**ReduceLROnPlateau 参数：**
- `patience`: 容忍轮数（默认：5）
- `factor`: 衰减系数（默认：0.5）

### 注意力监督（高级）

```yaml
training:
  use_attention_supervision: false
  attention_loss_weight: 0.1
  attention_supervision_method: "none"  # "mask", "cam", "none"
```

**参数说明：**
- `use_attention_supervision`: 是否使用注意力监督（默认：false）
- `attention_loss_weight`: 注意力损失权重（默认：0.1）
- `attention_supervision_method`: 监督方法
  - `"mask"`: 基于掩码的监督（需要数据集提供掩码）
  - `"cam"`: 基于 CAM 的监督（自动生成）
  - `"none"`: 不使用

## 5. 日志配置 (logging)

```yaml
logging:
  output_dir: "outputs"
  experiment_name: "experiment"
  use_tensorboard: true
  use_wandb: false
  wandb_project: "med-core"
  wandb_entity: null
  log_every_n_steps: 10
  val_check_interval: 1.0
  save_visualizations: true
  gradcam_samples: 10
```

**参数说明：**

**output_dir**: 输出目录（默认：`"outputs"`）
- 自动创建子目录：`checkpoints/`, `logs/`, `results/`

**experiment_name**: 实验名称（默认：`"experiment"`）

**use_tensorboard**: 是否使用 TensorBoard（默认：true）

**use_wandb**: 是否使用 Weights & Biases（默认：false）

**wandb_project**: W&B 项目名称（默认：`"med-core"`）

**wandb_entity**: W&B 团队名称（可选）

**log_every_n_steps**: 日志记录频率（默认：10）

**val_check_interval**: 验证检查间隔（默认：1.0）
- 1.0 表示每个 epoch 验证一次
- 0.5 表示每半个 epoch 验证一次

**save_visualizations**: 是否保存可视化结果（默认：true）

**gradcam_samples**: Grad-CAM 可视化样本数（默认：10）

## 常见配置模式

### 模式 1：快速测试

```yaml
# 最小化配置，用于快速验证
data:
  batch_size: 4
  num_workers: 0
  image_size: 224

model:
  vision:
    backbone: "resnet18"
    feature_dim: 64
  tabular:
    hidden_dims: [32]
    output_dim: 16
  fusion:
    fusion_type: "concatenate"

training:
  num_epochs: 3
  mixed_precision: false
  use_progressive_training: false

logging:
  use_tensorboard: false
  use_wandb: false
```

### 模式 2：高性能训练

```yaml
# 大数据集 + 强大 GPU
data:
  batch_size: 64
  num_workers: 8
  image_size: 384

model:
  vision:
    backbone: "swin_s"
    feature_dim: 512
  tabular:
    hidden_dims: [256, 128, 64]
    output_dim: 128
  fusion:
    fusion_type: "fused_attention"
    num_heads: 8

training:
  num_epochs: 100
  mixed_precision: true
  use_progressive_training: true
  optimizer:
    optimizer: "adamw"
    learning_rate: 5.0e-5
  scheduler:
    scheduler: "cosine"
    warmup_epochs: 10
```

### 模式 3：小数据集微调

```yaml
# 小数据集 + 预训练模型
model:
  vision:
    backbone: "resnet50"
    pretrained: true
    freeze_backbone: true
    freeze_strategy: "progressive"
  fusion:
    fusion_type: "gated"

training:
  num_epochs: 50
  label_smoothing: 0.1
  optimizer:
    use_differential_lr: true
    lr_backbone: 1.0e-6  # 很小的学习率
    lr_fusion: 1.0e-4
  early_stopping: true
  patience: 10
```

## 配置验证

MedFusion 提供自动配置验证，会在训练前检查配置的有效性：

```python
from med_core.configs.validation import validate_config_or_exit

# 自动验证并报告错误
config = validate_config_or_exit(config_dict)
```

**常见验证错误：**

- **E001**: 缺少必需字段
- **E002**: 字段类型错误
- **E003**: 数值超出有效范围
- **E004**: 数据划分比例之和不等于 1.0
- **E005**: CSV 文件不存在
- **E006**: 图像目录不存在
- **E007**: 不支持的骨干网络类型
- **E008**: 不支持的融合策略

## 最佳实践

1. **从默认配置开始**：复制 `configs/starter/default.yaml` 并修改
2. **使用有意义的命名**：`experiment_name` 应包含关键信息
3. **记录实验**：使用 `description` 和 `tags` 记录实验目的
4. **渐进式调整**：先用小模型快速验证，再用大模型训练
5. **监控指标**：使用 TensorBoard 或 W&B 跟踪训练过程
6. **保存配置**：每次实验保存完整配置文件
7. **版本控制**：将配置文件纳入 Git 管理

## 下一步

- [数据准备指南](04_data_preparation.md) - 学习如何准备训练数据
- [模型构建器 API](05_builder_api.md) - 使用代码构建模型
