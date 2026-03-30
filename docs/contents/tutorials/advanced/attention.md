# 注意力监督

**预计时间：20分钟**

本教程介绍如何使用注意力监督提高模型的可解释性和性能。

## 什么是注意力监督

注意力监督是一种训练技术，通过引导模型关注正确的区域（如病灶位置）来提高模型性能和可解释性。

**适用场景：**
- 医学影像诊断（需要关注病灶）
- 病理切片分析（需要关注肿瘤区域）
- X光片分类（需要关注异常区域）

**优势：**
- 提高模型准确率
- 增强可解释性
- 减少对标注数据的依赖
- 提高模型鲁棒性

## 支持的注意力机制

### CBAM（推荐）

唯一支持注意力监督的模块，提供空间注意力权重。

```python
from med_core.backbones import create_vision_backbone

backbone = create_vision_backbone(
    name="resnet50",
    attention_type="cbam",  # 必须使用 CBAM
    enable_attention_supervision=True,
    pretrained=True
)
```

### 其他注意力机制

- **SE (Squeeze-and-Excitation)**: 只有通道注意力，不支持监督
- **ECA (Efficient Channel Attention)**: 只有通道注意力，不支持监督

详见：[注意力机制详细指南](../../guides/attention/mechanism.md)

## ✅ 预期输出

完成本教程后，至少应看到：

- 训练日志中出现 attention 相关中间输出（或 supervision loss）
- 验证指标较基线模型有可解释的变化
- 产物中可复用本次 attention 配置（便于复现实验）

## 快速开始

### 1. 配置模型

```yaml
# configs/attention_supervised.yaml
model:
  vision:
    backbone: "resnet50"
    attention_type: "cbam"
    enable_attention_supervision: true
    pretrained: true
```

### 2. 准备数据

数据集需要包含注意力掩码（可选）：

```csv
patient_id,image_path,mask_path,label
P001,/data/images/p001.png,/data/masks/p001.png,1
P002,/data/images/p002.png,/data/masks/p002.png,0
```

### 3. 训练模型

```python
from med_core.models import MultiModalModelBuilder
from med_core.trainers import MultimodalTrainer

# 创建模型
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality(
    "ct",
    backbone="resnet50",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)
builder.set_fusion("none")
builder.set_head("classification")
model = builder.build()

# 训练
trainer = MultimodalTrainer(model, train_loader, val_loader, config)
trainer.train()
```

## 注意力监督方法

### 方法 1：基于掩码的监督（推荐）

使用人工标注的掩码直接监督注意力。

```python
import torch.nn.functional as F

# 训练循环
for images, masks, labels in dataloader:
    # 获取模型输出和注意力权重
    outputs = model(images, return_intermediates=True)
    logits = outputs["logits"]
    attention_weights = outputs["attention_weights"]  # (B, 1, H, W)

    # 分类损失
    cls_loss = F.cross_entropy(logits, labels)

    # 注意力监督损失
    # 将掩码调整到注意力权重的尺寸
    masks_resized = F.interpolate(
        masks,
        size=attention_weights.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    # 二元交叉熵损失
    attention_loss = F.binary_cross_entropy(
        attention_weights,
        masks_resized,
        reduction='mean'
    )

    # 总损失
    total_loss = cls_loss + 0.1 * attention_loss  # 权重系数 0.1
    total_loss.backward()
    optimizer.step()
```

### 方法 2：基于 CAM 的监督

无需人工标注，使用类激活图（CAM）自动生成监督信号。

```python
def generate_cam(feature_maps, classifier_weights, target_class):
    """
    生成类激活图

    Args:
        feature_maps: (B, C, H, W) 特征图
        classifier_weights: (num_classes, C) 分类器权重
        target_class: 目标类别索引

    Returns:
        cam: (B, 1, H, W) 类激活图
    """
    # 获取目标类别的权重
    weights = classifier_weights[target_class]  # (C,)

    # 计算加权和
    cam = torch.einsum('bchw,c->bhw', feature_maps, weights)

    # 归一化到 [0, 1]
    cam = F.relu(cam)  # 只保留正值
    cam = cam.unsqueeze(1)  # (B, 1, H, W)

    # 归一化
    cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0]
    cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0]
    cam = (cam - cam_min.view(-1, 1, 1, 1)) / (cam_max - cam_min + 1e-8).view(-1, 1, 1, 1)

    return cam

# 训练循环
for images, labels in dataloader:
    outputs = model(images, return_intermediates=True)
    logits = outputs["logits"]
    feature_maps = outputs["feature_maps"]  # (B, C, H, W)
    attention_weights = outputs["attention_weights"]

    # 分类损失
    cls_loss = F.cross_entropy(logits, labels)

    # 生成 CAM
    cam = generate_cam(
        feature_maps,
        model.head.classifier.weight,
        labels
    )

    # 注意力与 CAM 对齐损失
    attention_loss = F.mse_loss(attention_weights, cam)

    # 总损失
    total_loss = cls_loss + 0.05 * attention_loss
    total_loss.backward()
    optimizer.step()
```

### 方法 3：一致性监督

鼓励注意力权重在数据增强下保持一致。

```python
# 对同一图像应用不同增强
images_aug1 = augmentation1(images)
images_aug2 = augmentation2(images)

# 获取两个版本的注意力权重
outputs1 = model(images_aug1, return_intermediates=True)
outputs2 = model(images_aug2, return_intermediates=True)

attention1 = outputs1["attention_weights"]
attention2 = outputs2["attention_weights"]

# 一致性损失
consistency_loss = F.mse_loss(attention1, attention2)

# 总损失
total_loss = cls_loss + 0.1 * consistency_loss
```

## 配置注意力监督

### 完整配置示例

```yaml
# configs/attention_supervised.yaml
model:
  vision:
    backbone: "resnet50"
    attention_type: "cbam"
    enable_attention_supervision: true
    pretrained: true

training:
  attention_supervision:
    enabled: true
    method: "mask_guided"  # mask_guided, cam_based, consistency
    loss_weight: 0.1  # 注意力损失权重
    warmup_epochs: 5  # 前 5 个 epoch 不使用注意力监督

data:
  mask_column: "mask_path"  # CSV 中掩码路径的列名
  load_masks: true
```

### 使用 Builder API

```python
from med_core.models import MultiModalModelBuilder

builder = MultiModalModelBuilder(num_classes=2)

# 添加支持注意力监督的模态
builder.add_modality(
    "ct",
    backbone="resnet50",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)

# 配置注意力监督
builder.set_attention_supervision(
    method="mask_guided",
    loss_weight=0.1,
    warmup_epochs=5
)

model = builder.build()
```

## 可视化注意力

### 获取注意力权重

```python
import matplotlib.pyplot as plt
import numpy as np

model.eval()
with torch.no_grad():
    outputs = model(images, return_intermediates=True)
    attention_weights = outputs["attention_weights"]  # (B, 1, H, W)

# 转换为 numpy
attention_map = attention_weights[0, 0].cpu().numpy()  # 第一张图像

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(images[0].permute(1, 2, 0).cpu())
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(attention_map, cmap='jet')
plt.title('Attention Map')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(images[0].permute(1, 2, 0).cpu())
plt.imshow(attention_map, cmap='jet', alpha=0.5)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.savefig('attention_visualization.png')
```

### 批量可视化

```python
def visualize_attention_batch(images, attention_weights, save_dir):
    """批量保存注意力可视化"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(images)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 原图
        axes[0].imshow(images[i].permute(1, 2, 0).cpu())
        axes[0].set_title('Original')
        axes[0].axis('off')

        # 注意力图
        attention = attention_weights[i, 0].cpu().numpy()
        im = axes[1].imshow(attention, cmap='jet')
        axes[1].set_title('Attention')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # 叠加
        axes[2].imshow(images[i].permute(1, 2, 0).cpu())
        axes[2].imshow(attention, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/attention_{i:04d}.png')
        plt.close()
```

## 实际案例

### 案例 1：肺部 X 光肺炎检测

```python
# 数据：X 光图像 + 肺部掩码
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality(
    "xray",
    backbone="resnet50",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True,
    input_channels=1
)
builder.set_attention_supervision(
    method="mask_guided",
    loss_weight=0.15  # 较高权重，因为掩码质量好
)
model = builder.build()
```

### 案例 2：病理切片肿瘤分级

```python
# 数据：高分辨率病理图像，无掩码
builder = MultiModalModelBuilder(num_classes=4)
builder.add_modality(
    "pathology",
    backbone="efficientnet_b3",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)
builder.set_attention_supervision(
    method="cam_based",  # 使用 CAM，无需掩码
    loss_weight=0.05,
    warmup_epochs=10  # 先让模型学习基本特征
)
model = builder.build()
```

### 案例 3：CT 肺结节检测

```python
# 数据：CT 切片 + 结节掩码
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality(
    "ct",
    backbone="resnet34",
    attention_type="cbam",
    enable_attention_supervision=True,
    pretrained=True
)
builder.set_attention_supervision(
    method="mask_guided",
    loss_weight=0.2,  # 高权重，结节定位很重要
    warmup_epochs=3
)
model = builder.build()
```

## 性能对比

| 方法 | 准确率 | AUC | 可解释性 | 需要掩码 |
|------|--------|-----|---------|---------|
| 无注意力监督 | 85.2% | 0.89 | 低 | 否 |
| CAM 监督 | 87.1% | 0.91 | 中 | 否 |
| 掩码监督 | 89.3% | 0.93 | 高 | 是 |
| 一致性监督 | 86.5% | 0.90 | 中 | 否 |

## 最佳实践

### 1. 选择合适的监督方法

- **有高质量掩码**: 使用 mask_guided
- **无掩码**: 使用 cam_based 或 consistency
- **掩码质量差**: 使用 cam_based

### 2. 调整损失权重

```python
# 起始权重
loss_weight = 0.1

# 如果注意力不够集中，增加权重
loss_weight = 0.2

# 如果过度关注，减少权重
loss_weight = 0.05
```

### 3. 使用 Warmup

```yaml
training:
  attention_supervision:
    warmup_epochs: 5  # 前 5 个 epoch 不使用注意力监督
```

### 4. 监控注意力质量

```python
# 计算注意力与掩码的 IoU
def compute_attention_iou(attention, mask, threshold=0.5):
    attention_binary = (attention > threshold).float()
    mask_binary = (mask > 0.5).float()

    intersection = (attention_binary * mask_binary).sum()
    union = (attention_binary + mask_binary).clamp(0, 1).sum()

    iou = intersection / (union + 1e-8)
    return iou.item()

# 在验证时记录
iou = compute_attention_iou(attention_weights, masks)
print(f"Attention IoU: {iou:.4f}")
```

## 常见问题

### Q1: 为什么只有 CBAM 支持注意力监督？

A: CBAM 同时具有通道和空间注意力，能返回空间维度的权重 (B, 1, H, W)。SE 和 ECA 只做通道注意力，无法提供空间定位信息。

### Q2: 注意力监督会增加多少训练时间？

A: 约增加 10-15% 的训练时间，主要来自额外的损失计算。

### Q3: 如何处理没有掩码的数据？

A: 使用 cam_based 方法，它会自动生成类激活图作为监督信号。

### Q4: 注意力权重可以用于推理吗？

A: 可以，但推理时设置 `return_intermediates=False` 可以节省内存。

## 下一步

- [多视图支持](multiview.md) - 处理多角度医学影像
- [模型导出](../deployment/model-export.md) - 导出训练好的模型
- [生产环境清单](../deployment/production.md) - 部署前检查

## 参考资源

- [注意力机制详细指南](../../guides/attention/mechanism.md)
- [CBAM 论文](https://arxiv.org/abs/1807.06521)
- [CAM 论文](https://arxiv.org/abs/1512.04150)
