# 注意力监督训练使用指南

> **状态更新（2026-02-18）**: ✅ 功能已完全实现并集成到框架中，可以直接使用。

本指南介绍如何在 MedFusion 中使用注意力监督功能，在训练阶段引导模型关注正确的区域。

---

## 📋 目录

1. [快速开始](#快速开始)
2. [方法选择](#方法选择)
3. [详细示例](#详细示例)
4. [配置说明](#配置说明)
5. [最佳实践](#最佳实践)

---

## ⚠️ 重要前提

**注意力监督只支持 CBAM 注意力机制**，因为只有 CBAM 具有空间注意力权重。

- ✅ 支持：`attention_type="cbam"`
- ❌ 不支持：`attention_type="se"` 或 `"eca"`（只有通道注意力）
- ❌ 不支持：Transformer 架构（ViT、Swin）

---

## 🚀 快速开始

### 方法1: 使用配置系统（推荐）⭐

```python
from med_core.configs import ExperimentConfig
from med_core.fusion import create_fusion_model
from med_core.trainers import create_trainer
from med_core.datasets import MedicalMultimodalDataset

# 1. 配置
config = ExperimentConfig()

# 启用注意力监督
config.model.vision.attention_type = "cbam"  # 必须使用 CBAM
config.model.vision.enable_attention_supervision = True

config.training.use_attention_supervision = True
config.training.attention_loss_weight = 0.1
config.training.attention_supervision_method = "mask"  # 或 "cam"

# 2. 数据集（如果使用 mask 方法，CSV 需要包含掩码路径）
# CSV 格式: patient_id,image_path,mask_path,age,gender,label
dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data.csv",
    image_dir="images/",
    numerical_features=["age"],
    categorical_features=["gender"],
    target_column="label",
)

# 3. 模型
model = create_fusion_model(
    vision_backbone_name="resnet50",
    tabular_input_dim=2,
    fusion_type="gated",
    num_classes=2,
    config=config.model,
)

# 4. 训练器（自动处理注意力监督）
trainer = create_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

# 5. 训练
trainer.train()  # 注意力监督会自动应用
```

### 方法2: 使用 CAM 自监督（无需掩码标注）

```python
from med_core.configs import ExperimentConfig

# 配置 CAM 自监督
config = ExperimentConfig()

config.model.vision.attention_type = "cbam"
config.model.vision.enable_attention_supervision = True

config.training.use_attention_supervision = True
config.training.attention_loss_weight = 0.1
config.training.attention_supervision_method = "cam"  # 👈 使用 CAM 方法

# 数据集不需要掩码
# CSV 格式: patient_id,image_path,age,gender,label
dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data.csv",
    image_dir="images/",
    numerical_features=["age"],
    categorical_features=["gender"],
    target_column="label",
)

# 其余步骤相同
model = create_fusion_model(...)
trainer = create_trainer(model, train_loader, val_loader, config)
trainer.train()  # CAM 会自动生成并用于监督
```

---

## 🎯 方法选择

根据你的数据集标注情况选择合适的方法：

| 数据集标注 | 推荐方法 | 配置 | 效果 |
|-----------|---------|------|------|
| ✅ 有分割掩码 | Mask 监督 | `method="mask"` | ⭐⭐⭐⭐⭐ 最好 |
| ❌ 只有图像标签 | CAM 自监督 | `method="cam"` | ⭐⭐⭐⭐ 好 |

### 当前支持的方法

**1. Mask-based supervision**
- 需要：分割掩码标注
- 精度：最高
- 成本：需要人工标注

**2. CAM-based supervision**
- 需要：仅图像标签
- 精度：���高
- 成本：无需额外标注

### 检查数据集

```python
import pandas as pd

# 检查 CSV 是否包含掩码列
df = pd.read_csv("data.csv")
print("列名:", df.columns.tolist())

# 如果有 mask_path 列，可以使用 mask 方法
if "mask_path" in df.columns:
    print("✅ 可以使用 mask 监督")
    config.training.attention_supervision_method = "mask"
else:
    print("⚠️ 只能使用 CAM 监督")
    config.training.attention_supervision_method = "cam"
```

---

## 📚 详细示例

### 示例1: 肺炎检测（Mask 监督）

```python
from med_core.configs import ExperimentConfig
from med_core.datasets import MedicalMultimodalDataset
from med_core.fusion import create_fusion_model
from med_core.trainers import create_trainer
from torch.utils.data import DataLoader

# 1. 配置
config = ExperimentConfig(
    project_name="pneumonia-detection",
    experiment_name="with-attention-supervision",
)

config.model.vision.backbone = "resnet50"
config.model.vision.attention_type = "cbam"
config.model.vision.enable_attention_supervision = True

config.training.num_epochs = 100
config.training.batch_size = 32
config.training.use_attention_supervision = True
config.training.attention_loss_weight = 0.1
config.training.attention_supervision_method = "mask"

# 2. 数据集（CSV 包含掩码路径）
# CSV 格式: patient_id,image_path,mask_path,age,gender,fever,cough,label
train_dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data/pneumonia_train.csv",
    image_dir="data/chest_xrays/",
    numerical_features=["age"],
    categorical_features=["gender", "fever", "cough"],
    target_column="has_pneumonia",
)

val_dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data/pneumonia_val.csv",
    image_dir="data/chest_xrays/",
    numerical_features=["age"],
    categorical_features=["gender", "fever", "cough"],
    target_column="has_pneumonia",
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 模型
model = create_fusion_model(
    vision_backbone_name="resnet50",
    tabular_input_dim=4,  # age + gender + fever + cough
    fusion_type="gated",
    num_classes=2,
    config=config.model,
)

# 4. 训练
trainer = create_trainer(model, train_loader, val_loader, config)
trainer.train()

print(f"训练完成！最佳模型保存在: {trainer.best_model_path}")
```

### 示例2: 肺结节检测（CAM 自监督）

```python
from med_core.configs import ExperimentConfig

# 配置（使用 CAM，无需掩码）
config = ExperimentConfig(
    project_name="lung-nodule-detection",
    experiment_name="cam-supervision",
)

config.model.vision.backbone = "efficientnet_b0"
config.model.vision.attention_type = "cbam"
config.model.vision.enable_attention_supervision = True

config.training.use_attention_supervision = True
config.training.attention_loss_weight = 0.1
config.training.attention_supervision_method = "cam"  # 👈 CAM 方法

# 数据集（不需要掩码）
# CSV 格式: patient_id,image_path,age,smoking_history,label
dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data/nodules.csv",
    image_dir="data/ct_scans/",
    numerical_features=["age"],
    categorical_features=["smoking_history"],
    target_column="has_nodule",
)

# 其余步骤相同
model = create_fusion_model(...)
trainer = create_trainer(model, train_loader, val_loader, config)
trainer.train()
```

---

## ⚙️ 配置说明

### 使用主配置系统（推荐）

```python
from med_core.configs import ExperimentConfig

# 创建配置
config = ExperimentConfig()

# 模型配置
config.model.vision.backbone = "resnet50"
config.model.vision.attention_type = "cbam"  # 必须
config.model.vision.enable_attention_supervision = True  # 启用

# 训练配置
config.training.num_epochs = 100
config.training.batch_size = 32
config.training.learning_rate = 1e-4

# 注意力监督配置
config.training.use_attention_supervision = True
config.training.attention_loss_weight = 0.1  # 损失权重
config.training.attention_supervision_method = "mask"  # 或 "cam"

# 保存配置
config.save("config.yaml")
```

### 从 YAML 加载配置

```yaml
# config.yaml
model:
  vision:
    backbone: resnet50
    attention_type: cbam
    enable_attention_supervision: true
  
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  use_attention_supervision: true
  attention_loss_weight: 0.1
  attention_supervision_method: mask  # 或 cam
```

```python
from med_core.configs import ExperimentConfig

config = ExperimentConfig.from_yaml("config.yaml")
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model.vision.attention_type` | str | "cbam" | 必须使用 "cbam" |
| `model.vision.enable_attention_supervision` | bool | False | 启用注意力监督 |
| `training.use_attention_supervision` | bool | False | 在训练中使用 |
| `training.attention_loss_weight` | float | 0.1 | 注意力损失权重 |
| `training.attention_supervision_method` | str | "mask" | "mask" 或 "cam" |

---

## 💡 最佳实践

### 1. 损失权重调整

```python
# 开始时使用较小的权重
config.training.attention_loss_weight = 0.01  # 👈 从小开始

# 训练稳定后可以增加
# 在训练过程中观察：
# - 如果注意力损失远大于分类损失，减小权重
# - 如果注意力损失太小，增加权重
# 建议范围：0.05 - 0.2
```

### 2. 渐进式训练

```python
# 前几个 epoch 不使用注意力监督
config.training.use_attention_supervision = False

# 训练 10-20 个 epoch 后再启用
# 这样可以让模型先学习基本特征
```

### 3. 数据增强注意事项

如果使用 mask 方法，图像和掩码需要使用相同的变换：

```python
from torchvision import transforms

# 定义变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 图像和掩码都会翻转
    transforms.ToTensor(),
])

# 数据集会自动对图像和掩码应用相同的变换
dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data.csv",
    image_dir="images/",
    transform=transform,
)
```

### 4. 监控训练过程

```python
# 训练器会自动记录注意力损失
# 在 TensorBoard 中查看：
# - train/cls_loss: 分类损失
# - train/att_loss: 注意力损失
# - train/loss: 总损失

# 启动 TensorBoard
# tensorboard --logdir outputs/logs
```

### 5. 选择合适的方法

```python
# 决策树
if has_mask_annotations:
    method = "mask"  # 最佳精度
else:
    method = "cam"   # 无需标注
```

---

## 🔍 故障排除

### 问题1: 注意力损失过大

**原因**: 损失权重太大

**解决方案**:
```python
# 减小损失权重
config.training.attention_loss_weight = 0.01  # 从 0.1 降到 0.01
```

### 问题2: 训练不稳定

**原因**: 注意力监督过早引入

**解决方案**:
```python
# 延迟启用注意力监督
# 先训练 10-20 个 epoch，让模型学习基本特征
# 然后再启用注意力监督
```

### 问题3: 警告信息

```
WARNING: use_attention_supervision=True but vision.enable_attention_supervision=False
```

**解决方案**:
```python
# 确保两个配置都启用
config.model.vision.enable_attention_supervision = True
config.training.use_attention_supervision = True
```

### 问题4: 不支持的注意力类型

```
WARNING: Attention supervision only works with CBAM, but attention_type=se
```

**解决方案**:
```python
# 必须使用 CBAM
config.model.vision.attention_type = "cbam"
```

---

## 📖 参考资料

- [注意力机制指南](./mechanism.md) - CBAM/SE/ECA 使用方法
- [架构分析报告](../../architecture/analysis.md) - 框架整体架构
- [注意力监督审查报告](../../reviews/attention_supervision.md) - 功能实现验证

---

## 📝 更新日志

- **2026-02-18**: 更新文档以反映实际实现，简化使用方法
- **2026-02-13**: 初始版本

**版本**: v1.1  
**状态**: ✅ 功能已完全实现并可用
