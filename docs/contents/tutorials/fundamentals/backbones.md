# 选择骨干网络

**预计时间：15分钟**

本教程帮助你为医学影像任务选择合适的骨干网络（Backbone）。

## 骨干网络概述

MedFusion 支持 29+ 种视觉骨干网络，分为三大类：

### 1. CNN 架构

**ResNet 系列**
- `resnet18/34/50/101/152`
- 经典架构，稳定可靠
- 适合：通用医学影像任务

**EfficientNet 系列**
- `efficientnet_b0` 到 `efficientnet_b7`
- `efficientnet_v2_s/m/l`
- 高效率，精度与速度平衡
- 适合：资源受限环境

**MobileNet 系列**
- `mobilenetv2/mobilenetv3_small/large`
- 轻量级，适合移动端
- 适合：边缘设备部署

**ConvNeXt 系列**
- `convnext_tiny/small/base/large`
- 现代 CNN 设计
- 适合：追求高精度

### 2. Transformer 架构

**Vision Transformer (ViT)**
- `vit_b_16/vit_b_32/vit_l_16`
- 全局感受野
- 适合：大规模数据集

**Swin Transformer**
- `swin_t/swin_s/swin_b`
- 层次化设计
- 适合：多尺度特征

### 3. 3D 架构

**3D ResNet**
- `resnet3d_18/34/50`
- 处理体积数据
- 适合：CT/MRI 体积扫描

**3D Swin Transformer**
- `swin3d_tiny/small/base`
- 3D 注意力机制
- 适合：高分辨率 3D 数据

## 决策树

```
你的数据类型？
├─ 2D 图像（X光、病理切片）
│   ├─ 数据量大（>10k）？
│   │   ├─ 是 → ViT 或 Swin Transformer
│   │   └─ 否 → ResNet50 + 预训练
│   ├─ 需要移动端部署？
│   │   └─ 是 → MobileNetV3 或 EfficientNet-B0
│   └─ 追求最高精度？
│       └─ 是 → ConvNeXt-Base 或 EfficientNetV2-L
│
└─ 3D 体积数据（CT、MRI）
    ├─ 内存受限？
    │   └─ 是 → ResNet3D-18
    └─ 追求精度？
        └─ 是 → Swin3D-Small
```

## 性能对比

### 2D 骨干网络

| 骨干网络 | 参数量 | FLOPs | ImageNet Top-1 | 推理速度 |
|---------|--------|-------|----------------|---------|
| MobileNetV3-Large | 5.4M | 0.2G | 75.2% | 快 |
| EfficientNet-B0 | 5.3M | 0.4G | 77.1% | 快 |
| ResNet50 | 25.6M | 4.1G | 80.4% | 中 |
| ConvNeXt-Tiny | 28.6M | 4.5G | 82.1% | 中 |
| ViT-B/16 | 86.6M | 17.6G | 81.8% | 慢 |
| Swin-Base | 88M | 15.4G | 83.5% | 慢 |

### 3D 骨干网络

| 骨干网络 | 参数量 | 内存占用 | 适用场景 |
|---------|--------|---------|---------|
| ResNet3D-18 | 33M | ~4GB | 快速原型 |
| ResNet3D-50 | 46M | ~8GB | 通用任务 |
| Swin3D-Tiny | 28M | ~6GB | 高精度 |

## 代码示例

### 使用 ResNet50（推荐起点）

```python
from med_core.models import MultiModalModelBuilder

builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality(
    "ct",
    backbone="resnet50",
    pretrained=True,  # 使用 ImageNet 预训练
    input_channels=3
)
model = builder.build()
```

### 使用 EfficientNet（效率优先）

```python
builder.add_modality(
    "xray",
    backbone="efficientnet_b0",
    pretrained=True,
    input_channels=1  # 灰度图像
)
```

### 使用 Swin Transformer（精度优先）

```python
builder.add_modality(
    "pathology",
    backbone="swin_t",
    pretrained=True,
    input_channels=3
)
```

### 使用 3D 骨干网络

```python
builder.add_modality(
    "ct_volume",
    backbone="swin3d_tiny",
    pretrained=False,  # 3D 模型通常无预训练
    input_channels=1,
    input_size=(96, 96, 96)  # 3D 输入尺寸
)
```

## 预训练权重

### 何时使用预训练？

**推荐使用预训练：**
- 数据量 < 10,000 样本
- 2D 自然图像相似任务（X光、病理）
- 快速原型开发

**不推荐预训练：**
- 3D 医学影像（无可用预训练）
- 特殊模态（超声、OCT）
- 数据量 > 100,000 样本

### 加载预训练权重

```python
# 方法 1：通过 builder
builder.add_modality(
    "ct",
    backbone="resnet50",
    pretrained=True  # 自动下载 ImageNet 权重
)

# 方法 2：手动加载
from med_core.backbones import create_vision_backbone

backbone = create_vision_backbone(
    name="resnet50",
    pretrained=True,
    in_channels=1  # 自动调整第一层
)
```

### 微调策略

```python
# 冻结骨干网络，只训练分类头
for param in model.backbones['ct'].parameters():
    param.requires_grad = False

# 或使用渐进式解冻（推荐）
config.training.progressive_unfreezing = True
config.training.unfreeze_schedule = [
    {"epoch": 0, "modules": ["head"]},
    {"epoch": 5, "modules": ["head", "fusion"]},
    {"epoch": 10, "modules": ["all"]}
]
```

## 内存和速度权衡

### 减少内存占用

```python
# 1. 使用更小的骨干网络
backbone="resnet18"  # 而非 resnet50

# 2. 减小输入尺寸
input_size=(224, 224)  # 而非 (512, 512)

# 3. 启用混合精度训练
config.training.mixed_precision = True

# 4. 减小批次大小
config.data.batch_size = 16  # 而非 32
```

### 提升推理速度

```python
# 1. 使用轻量级骨干网络
backbone="mobilenetv3_large"

# 2. 导出为 ONNX
from med_core.utils.export import export_model
export_model(model, "model.onnx", format="onnx")

# 3. 使用 TorchScript
model_scripted = torch.jit.script(model)
```

## 实际案例

### 案例 1：肺部 X 光分类

```python
# 数据：5000 张 X 光图像
# 目标：分类肺炎 vs 正常
# 推荐：ResNet50 + 预训练

builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality(
    "xray",
    backbone="resnet50",
    pretrained=True,
    input_channels=1
)
builder.set_fusion("none")  # 单模态
builder.set_head("classification")
model = builder.build()
```

### 案例 2：病理切片分析

```python
# 数据：50000 张高分辨率病理图像
# 目标：肿瘤分级
# 推荐：Swin Transformer

builder.add_modality(
    "pathology",
    backbone="swin_t",
    pretrained=True,
    input_channels=3
)
```

### 案例 3：CT 体积分析

```python
# 数据：1000 个 CT 体积
# 目标：肺结节检测
# 推荐：Swin3D-Tiny

builder.add_modality(
    "ct_volume",
    backbone="swin3d_tiny",
    pretrained=False,
    input_channels=1,
    input_size=(96, 96, 96)
)
```

### 案例 4：移动端部署

```python
# 目标：在移动设备上实时推理
# 推荐：MobileNetV3

builder.add_modality(
    "image",
    backbone="mobilenetv3_large",
    pretrained=True,
    input_channels=3
)
```

## 常见问题

### Q1: 如何选择 ResNet 的深度？

**A:** 根据数据量选择：
- < 5k 样本：ResNet18
- 5k-20k 样本：ResNet34 或 ResNet50
- > 20k 样本：ResNet50 或更深

### Q2: Transformer 比 CNN 更好吗？

**A:** 不一定：
- 数据量大（>10k）：Transformer 通常更好
- 数据量小（<5k）：CNN + 预训练更稳定
- 推理速度要求高：CNN 更快

### Q3: 3D 骨干网络内存占用太大怎么办？

**A:** 三种方法：
1. 使用更小的模型（ResNet3D-18）
2. 减小输入尺寸（96³ → 64³）
3. 使用 2.5D 方法（多切片 2D 模型）

### Q4: 如何处理灰度医学图像？

**A:** 两种方法：
```python
# 方法 1：复制通道（推荐）
input_channels=1  # 自动调整第一层卷积

# 方法 2：手动复制
image = image.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
```

## 下一步

- [融合策略选择](fusion.md) - 学习如何组合多模态
- [训练工作流](../training/workflow.md) - 开始训练模型
- [超参数调优](../training/tuning.md) - 优化模型性能

## 参考资源

- [骨干网络 API 文档](../../api/backbones.md)
- [模型导出指南](../deployment/model-export.md)
- [性能基准测试](../../guides/advanced-features/performance-benchmarking.md)
