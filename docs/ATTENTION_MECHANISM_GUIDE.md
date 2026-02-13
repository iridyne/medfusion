# 注意力机制使用指南

## 概述

Med-Framework 支持在 Vision Backbone 中集成多种注意力机制，并提供可选的注意力监督训练功能。本文档详细说明了注意力机制的使用方法、限制和最佳实践。

## 支持的注意力机制

### 1. CBAM (Convolutional Block Attention Module)

**特点：**
- 结合通道注意力和空间注意力
- 可以返回空间注意力权重
- **唯一支持注意力监督训练的模块**

**参数量：** 中等  
**计算量：** 中等  
**适用场景：** 需要空间注意力和监督训练的任务

```python
backbone = ResNetBackbone(
    variant="resnet50",
    attention_type="cbam",
    enable_attention_supervision=True  # 支持
)
```

### 2. SE (Squeeze-and-Excitation)

**特点：**
- 只有通道注意力
- 轻量级设计
- **不支持注意力监督训练**（无空间权重）

**参数量：** 少  
**计算量：** 少  
**适用场景：** 轻量级模型，只需通道注意力

```python
backbone = MobileNetBackbone(
    variant="mobilenetv2",
    attention_type="se",
    enable_attention_supervision=True  # ⚠️ 无效，SE 不返回权重
)
```

### 3. ECA (Efficient Channel Attention)

**特点：**
- 只有通道注意力（使用 1D 卷积）
- 极致效率
- **不支持注意力监督训练**（无空间权重）

**参数量：** 极少  
**计算量：** 极少  
**适用场景：** 移动端部署，追求极致效率

```python
backbone = EfficientNetBackbone(
    variant="efficientnet_b0",
    attention_type="eca",
    enable_attention_supervision=True  # ⚠️ 无效，ECA 不返回权重
)
```

### 4. None (无注意力)

**特点：**
- 不添加额外注意力模块
- 零额外开销

**适用场景：** Backbone 本身已足够强（如 ConvNeXt），或 Transformer 架构

```python
backbone = ConvNeXtBackbone(
    variant="convnext_tiny",
    attention_type="none"  # ConvNeXt 内部设计已经很好
)
```

## Backbone 与注意力模块的兼容性

### 支持矩阵

| Backbone 类型 | CBAM | SE | ECA | None | 注意力监督 |
|--------------|------|----|----|------|-----------|
| **ResNet** | ✅ | ✅ | ✅ | ✅ | ✅ (仅 CBAM) |
| **MobileNet** | ✅ | ✅ | ✅ | ✅ | ✅ (仅 CBAM) |
| **EfficientNet** | ✅ | ✅ | ✅ | ✅ | ✅ (仅 CBAM) |
| **EfficientNetV2** | ✅ | ✅ | ✅ | ✅ | ✅ (仅 CBAM) |
| **ConvNeXt** | ✅ | ✅ | ✅ | ✅ | ✅ (仅 CBAM) |
| **RegNet** | ✅ | ✅ | ✅ | ✅ | ✅ (仅 CBAM) |
| **ViT** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Swin** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **MaxViT** | ❌ | ❌ | ❌ | ✅ | ❌ |

### 重要限制

#### ⚠️ 限制 1：Transformer 架构不支持外部注意力模块

**原因：**
- CNN 输出 **(B, C, H, W)** 特征图，可以应用 CBAM/SE/ECA
- Transformer 输出 **(B, hidden_dim)** 向量，已经过全局注意力处理

**受影响的 Backbone：**
- ViTBackbone
- SwinBackbone
- MaxViTBackbone

```python
# ❌ 错误用法
backbone = ViTBackbone(attention_type="cbam")  # 会被忽略或报错

# ✅ 正确用法
backbone = ViTBackbone()  # Transformer 内部已有自注意力机制
```

#### ⚠️ 限制 2：只有 CBAM 支持注意力监督训练

**原因：**
- 注意力监督需要**空间注意力权重** (spatial attention weights)
- SE 和 ECA 只做通道注意力，没有空间维度的权重

**影响：**
```python
# ✅ 支持注意力监督
backbone = ResNetBackbone(
    attention_type="cbam",
    enable_attention_supervision=True  # 有效
)

# ❌ 不支持注意力监督
backbone = ResNetBackbone(
    attention_type="se",
    enable_attention_supervision=True  # 无效，不会报错但不返回权重
)

backbone = ResNetBackbone(
    attention_type="eca",
    enable_attention_supervision=True  # 无效，不会报错但不返回权重
)
```

## 注意力监督训练

### 什么是注意力监督？

注意力监督训练是指在训练过程中，引导模型关注正确的区域（如病灶位置）。

### 使用方法

#### 1. 启用注意力监督

```python
from med_core.backbones import create_vision_backbone

backbone = create_vision_backbone(
    name="resnet50",
    attention_type="cbam",  # 必须使用 CBAM
    enable_attention_supervision=True,  # 启用监督
    pretrained=True
)
```

#### 2. 获取注意力权重

```python
# 训练时获取中间结果
outputs = backbone(images, return_intermediates=True)

# outputs 是一个字典
{
    "features": torch.Tensor,           # (B, feature_dim) 最终特征
    "feature_maps": torch.Tensor,       # (B, C, H, W) 池化前的特征图
    "attention_weights": torch.Tensor,  # (B, 1, H, W) 空间注意力权重
}
```

#### 3. 使用注意力权重进行监督

```python
# 假设有分割掩码标注
masks = ...  # (B, 1, H, W)

# 计算注意力与掩码的对齐损失
attention_weights = outputs["attention_weights"]
attention_loss = F.binary_cross_entropy(
    attention_weights,
    masks,
    reduction="mean"
)

# 总损失
total_loss = classification_loss + 0.1 * attention_loss
```

### 注意力监督的两种方法

#### 方法 1：直接使用 CBAM 权重（推荐）

**优点：**
- 简单直接
- 实时监督
- 适合有掩码标注的数据集

**缺点：**
- 需要人工标注的掩码

```python
backbone = ResNetBackbone(
    attention_type="cbam",
    enable_attention_supervision=True
)

outputs = backbone(images, return_intermediates=True)
attention_weights = outputs["attention_weights"]  # 直接使用
```

#### 方法 2：使用 CAM (Class Activation Map)

**优点：**
- 无需掩码标注
- 自动生成热力图

**缺点：**
- 需要额外计算
- 精度不如人工标注

```python
# CAM 方法需要 feature_maps 和 classifier 权重
feature_maps = outputs["feature_maps"]  # (B, C, H, W)
classifier_weights = model.classifier.weight  # (num_classes, C)

# 生成 CAM
cam = torch.einsum('bchw,nc->bnhw', feature_maps, classifier_weights)
cam = F.relu(cam)  # 只保留正值
cam = cam / (cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
```

## 推荐搭配

### 场景 1：医学影像诊断（需要可解释性）

```python
backbone = create_vision_backbone(
    name="resnet50",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)
```

**理由：**
- CBAM 提供空间注意力
- 可以监督模型关注病灶区域
- 可视化注意力权重提高可解释性

### 场景 2：移动端部署（追求效率）

```python
backbone = create_vision_backbone(
    name="mobilenetv3_large",
    attention_type="eca",  # 最轻量
    pretrained=True
)
```

**理由：**
- MobileNet 本身轻量
- ECA 额外开销极小
- 适合资源受限环境

### 场景 3：高精度任务（不在乎计算量）

```python
backbone = create_vision_backbone(
    name="efficientnet_v2_l",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)
```

**理由：**
- EfficientNetV2 精度高
- CBAM 进一步提升性能
- 注意力监督提高鲁棒性

### 场景 4：已经很强的模型

```python
backbone = create_vision_backbone(
    name="convnext_base",
    attention_type="none",  # 不需要额外注意力
    pretrained=True
)
```

**理由：**
- ConvNeXt 内部设计已经很好
- 额外注意力模块收益有限

### 场景 5：Transformer 架构

```python
backbone = create_vision_backbone(
    name="vit_b_16",
    # 不支持 attention_type 参数
    pretrained=True
)
```

**理由：**
- Transformer 内部已有自注意力机制
- 不需要也不支持外部注意力模块

## 性能对比

| 注意力模块 | 参数增加 | FLOPs 增加 | 精度提升 | 支持监督 |
|-----------|---------|-----------|---------|---------|
| **CBAM** | ~0.5% | ~1% | +1-2% | ✅ |
| **SE** | ~0.1% | ~0.5% | +0.5-1% | ❌ |
| **ECA** | <0.01% | <0.1% | +0.3-0.8% | ❌ |
| **None** | 0% | 0% | 0% | ❌ |

*注：具体数值取决于 Backbone 和数据集*

## 常见问题

### Q1: 为什么 SE 和 ECA 不支持注意力监督？

**A:** SE 和 ECA 只做通道注意力，输出的权重是 **(B, C, 1, 1)** 形状，表示每个通道的重要性。注意力监督需要空间维度的权重 **(B, 1, H, W)**，表示图像中每个位置的重要性。只有 CBAM 同时具有通道和空间注意力。

### Q2: 可以同时使用多个注意力模块吗？

**A:** 当前实现不支持。每个 Backbone 只能选择一种注意力类型。如果需要多种注意力，可以修改 Backbone 代码手动添加。

### Q3: Transformer 为什么不支持外部注意力模块？

**A:** Transformer 架构（ViT/Swin/MaxViT）内部已经使用了自注意力机制，输出的是全局特征向量而非特征图。外部的 CBAM/SE/ECA 是为 CNN 特征图设计的，无法应用到 Transformer 的输出上。

### Q4: 如何选择注意力模块？

**决策树：**
```
需要注意力监督？
├─ 是 → 使用 CBAM
└─ 否 → 追求效率？
    ├─ 是 → 使用 ECA
    └─ 否 → 追求精度？
        ├─ 是 → 使用 CBAM
        └─ 否 → 使用 SE 或 None
```

### Q5: enable_attention_supervision 对性能有影响吗？

**A:** 有轻微影响：
- **训练时：** 需要额外返回注意力权重，增加约 5-10% 内存占用
- **推理时：** 如果设置 `return_intermediates=False`（默认），没有额外开销

## 代码示例

### 完整训练示例

```python
from med_core.backbones import create_vision_backbone
from med_core.trainers import MultimodalTrainer
import torch.nn.functional as F

# 1. 创建支持注意力监督的 Backbone
backbone = create_vision_backbone(
    name="resnet50",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)

# 2. 训练循环
for images, masks, labels in dataloader:
    # 获取中间结果
    outputs = backbone(images, return_intermediates=True)
    
    # 分类损失
    logits = classifier(outputs["features"])
    cls_loss = F.cross_entropy(logits, labels)
    
    # 注意力监督损失
    attention_weights = outputs["attention_weights"]
    attention_loss = F.binary_cross_entropy(
        attention_weights,
        F.interpolate(masks, size=attention_weights.shape[-2:]),
        reduction="mean"
    )
    
    # 总损失
    total_loss = cls_loss + 0.1 * attention_loss
    total_loss.backward()
    optimizer.step()
```

### 推理示例

```python
# 推理时不需要中间结果
backbone.eval()
with torch.no_grad():
    features = backbone(images)  # 只返回特征，零额外开销
    logits = classifier(features)
```

## 最佳实践

1. **医学影像任务优先使用 CBAM + 注意力监督**
2. **移动端部署优先使用 ECA**
3. **Transformer 架构不要尝试添加外部注意力**
4. **推理时关闭 return_intermediates 以节省内存**
5. **注意力监督的权重系数建议在 0.05-0.2 之间**

## 更新日志

- **2026-02-13**: 初始版本，记录注意力机制的限制和使用方法
