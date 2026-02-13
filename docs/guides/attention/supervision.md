# 离线注意力引导使用指南

本指南介绍如何在 Med-Framework 中使用离线注意力引导功能，在训练阶段就让模型学会关注正确的区域。

---

## 📋 目录

1. [快速开始](#快速开始)
2. [方法选择](#方法选择)
3. [详细示例](#详细示例)
4. [配置说明](#配置说明)
5. [可视化](#可视化)
6. [最佳实践](#最佳实践)

---

## 🚀 快速开始

### 方法1: 使用分割掩码监督（推荐，如果有掩码）

```python
import torch
from med_core.datasets.attention_supervised import MedicalAttentionSupervisedDataset
from med_core.attention_supervision import MaskSupervisedAttention
from med_core.configs.attention_config import create_mask_supervised_config

# 1. 创建数据集（支持掩码加载）
dataset = MedicalAttentionSupervisedDataset.from_csv(
    csv_path="data/annotations.csv",
    image_dir="data/images/",
    mask_dir="data/masks/",  # 👈 掩码目录
    image_col="scan_path",
    mask_col="lesion_mask",
    label_col="diagnosis",
    tabular_cols=["age", "gender", "symptoms"],
    return_mask=True,  # 👈 返回掩码
)

# 2. 创建注意力监督模块
attention_supervision = MaskSupervisedAttention(
    loss_weight=0.1,
    loss_type="kl",
    temperature=10.0,
)

# 3. 训练循环
for images, tabular, labels, masks in dataloader:
    # 前向传播
    outputs = model(images, tabular)
    attention = outputs["attention_weights"]  # 模型的注意力权重
    
    # 计算分类损失
    classification_loss = criterion(outputs["logits"], labels)
    
    # 计算注意力监督损失
    attention_loss_result = attention_supervision(
        attention_weights=attention,
        features=outputs["features"],
        targets=masks,  # 👈 使用掩码监督
    )
    
    # 总损失
    total_loss = classification_loss + attention_loss_result.total_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
```

### 方法2: 使用 CAM 自监督（推荐，只有图像标签）

```python
from med_core.attention_supervision import CAMSelfSupervision

# 创建 CAM 自监督模块
attention_supervision = CAMSelfSupervision(
    loss_weight=0.1,
    consistency_method="entropy",
    alignment_weight=0.5,
)

# 训练循环
for images, tabular, labels in dataloader:  # 👈 不需要掩码
    outputs = model(images, tabular)
    
    classification_loss = criterion(outputs["logits"], labels)
    
    # CAM 自监督
    attention_loss_result = attention_supervision(
        attention_weights=outputs["attention_weights"],
        features=outputs["features"],
        classifier_weights=model.classifier.weight,  # 👈 分类器权重
        predicted_class=outputs["logits"].argmax(dim=1),
    )
    
    total_loss = classification_loss + attention_loss_result.total_loss
    total_loss.backward()
    optimizer.step()
```

---

## 🎯 方法选择

根据你的���据集标注情况选择合适的方法：

| 数据集标注 | 推荐方法 | 优先级 | 效果 |
|-----------|---------|--------|------|
| ✅ 有分割掩码 | 分割掩码监督 | ⭐⭐⭐⭐⭐ | 最好 |
| ✅ 有边界框 | 边界框监督 | ⭐⭐⭐⭐ | 很好 |
| ✅ 有关键点 | 关键点监督 | ⭐⭐⭐ | 好 |
| ❌ 只有图像标签 | CAM 自监督 | ⭐⭐⭐⭐ | 好 |
| ❌ 只有图像标签 | 多实例学习 | ⭐⭐⭐⭐ | 好 |

### 检查数据集标注

```python
# 检查你的数据集有什么标注
dataset = MedicalAttentionSupervisedDataset.from_csv(...)

print(f"有分割掩码: {dataset.has_masks()}")
print(f"有边界框: {dataset.has_bboxes()}")
print(f"有关键点: {dataset.has_keypoints()}")

if dataset.has_masks():
    print(f"掩码覆盖率: {dataset.get_mask_coverage():.1%}")
```

---

## 📚 详细示例

### 示例1: 肺炎检测（分割掩码监督）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from med_core.datasets.attention_supervised import MedicalAttentionSupervisedDataset
from med_core.attention_supervision import MaskSupervisedAttention
from med_core.visualization.attention_viz import visualize_attention_supervision_loss

# 1. 数据准备
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = MedicalAttentionSupervisedDataset.from_csv(
    csv_path="data/pneumonia.csv",
    image_dir="data/chest_xrays/",
    mask_dir="data/lesion_masks/",
    image_col="xray_path",
    mask_col="lesion_mask",
    label_col="has_pneumonia",
    tabular_cols=["age", "gender", "fever", "cough"],
    image_format="png",
    transform=transform,
    return_mask=True,
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 模型定义（假设你已有模型）
class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 你的模型定义
        self.attention_module = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, images, tabular):
        features = self.backbone(images)  # (B, 512, 7, 7)
        attention = self.attention_module(features)  # (B, 1, 7, 7)
        
        # 加权特征
        weighted_features = features * attention
        pooled = F.adaptive_avg_pool2d(weighted_features, 1).flatten(1)
        
        # 融合表格数据
        combined = torch.cat([pooled, tabular], dim=1)
        logits = self.classifier(combined)
        
        return {
            "logits": logits,
            "attention_weights": attention.squeeze(1),
            "features": features,
        }

model = PneumoniaModel()

# 3. 注意力监督
attention_supervision = MaskSupervisedAttention(
    loss_weight=0.1,
    loss_type="kl",
    temperature=10.0,
    add_smooth_loss=True,
    smooth_weight=0.01,
)

# 4. 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    for batch_idx, (images, tabular, labels, masks) in enumerate(dataloader):
        images = images.cuda()
        tabular = tabular.cuda()
        labels = labels.cuda()
        masks = masks.cuda()
        
        # 前向传播
        outputs = model(images, tabular)
        
        # 分类损失
        classification_loss = criterion(outputs["logits"], labels)
        
        # 注意力监督损失
        attention_loss_result = attention_supervision(
            attention_weights=outputs["attention_weights"],
            features=outputs["features"],
            targets=masks,
        )
        
        # 总损失
        total_loss = classification_loss + attention_loss_result.total_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 日志
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"  分类损失: {classification_loss.item():.4f}")
            print(f"  注意力损失: {attention_loss_result.total_loss.item():.4f}")
            for name, value in attention_loss_result.components.items():
                print(f"    {name}: {value.item():.4f}")
        
        # 可视化（每100步）
        if batch_idx % 100 == 0:
            fig = visualize_attention_supervision_loss(
                image=images[0],
                attention=attention_loss_result.attention_weights[0],
                target=attention_loss_result.metadata["target"][0],
                loss_components={k: v.item() for k, v in attention_loss_result.components.items()},
                save_path=f"outputs/attention_epoch{epoch}_batch{batch_idx}.png",
            )
            plt.close(fig)
```

### 示例2: 肺结节检测（CAM 自监督）

```python
from med_core.attention_supervision import CAMSelfSupervision

# 数据集（只有图像标签，没有掩码）
dataset = MedicalAttentionSupervisedDataset.from_csv(
    csv_path="data/nodules.csv",
    image_dir="data/ct_scans/",
    mask_dir=None,  # 👈 没有掩码
    label_col="has_nodule",
    tabular_cols=["age", "smoking_history"],
    return_mask=False,  # 👈 不返回掩码
)

# CAM 自监督
attention_supervision = CAMSelfSupervision(
    loss_weight=0.1,
    consistency_method="entropy",
    consistency_weight=1.0,
    alignment_weight=0.5,
)

# 训练
for images, tabular, labels in dataloader:  # 👈 没有掩码
    outputs = model(images, tabular)
    
    classification_loss = criterion(outputs["logits"], labels)
    
    # CAM 自监督
    attention_loss_result = attention_supervision(
        attention_weights=outputs["attention_weights"],
        features=outputs["features"],
        classifier_weights=model.classifier.weight,
        predicted_class=outputs["logits"].argmax(dim=1),
    )
    
    total_loss = classification_loss + attention_loss_result.total_loss
    total_loss.backward()
    optimizer.step()
    
    # 可视化 CAM
    if batch_idx % 100 == 0:
        cam = attention_loss_result.metadata["cam"][0]
        fig = visualize_attention_overlay(
            image=images[0],
            attention=cam,
            title="CAM 可视化",
            save_path=f"outputs/cam_batch{batch_idx}.png",
        )
        plt.close(fig)
```

### 示例3: 多实例学习（MIL）

```python
from med_core.attention_supervision import AttentionMIL, MILSupervision
from med_core.visualization.attention_viz import visualize_mil_attention

# 创建 MIL 模型
from torchvision.models import resnet18
backbone = resnet18(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-2])  # 移除分类层

mil_model = AttentionMIL(
    backbone=backbone,
    feature_dim=512,
    num_classes=2,
    patch_size=16,
    attention_dim=128,
    pooling_mode="attention",
)

# MIL 监督
mil_supervision = MILSupervision(
    loss_weight=0.1,
    patch_size=16,
    diversity_weight=0.1,
)

# 训练
for images, tabular, labels in dataloader:
    # MIL 前向传播
    mil_outputs = mil_model(images)
    
    # 分类损失
    classification_loss = criterion(mil_outputs["logits"], labels)
    
    # MIL 监督损失
    attention_loss_result = mil_supervision(
        attention_weights=mil_outputs["attention_weights"],
        features=mil_outputs["patch_features"],
        grid_size=mil_outputs["grid_size"],
    )
    
    total_loss = classification_loss + attention_loss_result.total_loss
    total_loss.backward()
    optimizer.step()
    
    # 可视化 MIL 注意力
    if batch_idx % 100 == 0:
        fig = visualize_mil_attention(
            image=images[0],
            patch_attention=mil_outputs["attention_weights"][0],
            grid_size=mil_outputs["grid_size"],
            top_k=5,
            save_path=f"outputs/mil_batch{batch_idx}.png",
        )
        plt.close(fig)
```

### 示例4: 边界框监督

```python
from med_core.attention_supervision import BBoxSupervisedAttention

# 数据集（有边界框标注）
dataset = AttentionSupervisedDataset(
    image_paths=image_paths,
    tabular_data=tabular_data,
    labels=labels,
    bboxes=bboxes,  # 👈 边界框列表 [[x_min, y_min, x_max, y_max], ...]
    return_bbox=True,
)

# 边界框监督
attention_supervision = BBoxSupervisedAttention(
    loss_weight=0.1,
    bbox_format="xyxy",
)

# 训练
for images, tabular, labels, bboxes in dataloader:
    outputs = model(images, tabular)
    
    classification_loss = criterion(outputs["logits"], labels)
    
    # 边界框监督
    attention_loss_result = attention_supervision(
        attention_weights=outputs["attention_weights"],
        features=outputs["features"],
        targets=bboxes,  # 👈 边界框
        image_size=(512, 512),  # 原图尺寸
    )
    
    total_loss = classification_loss + attention_loss_result.total_loss
    total_loss.backward()
    optimizer.step()
```

---

## ⚙️ 配置说明

### 使用配置文件

```python
from med_core.configs.attention_config import (
    ExperimentConfigWithAttention,
    DataConfigWithMask,
    TrainingConfigWithAttention,
    AttentionSupervisionConfig,
)

# 创建完整配置
config = ExperimentConfigWithAttention(
    experiment_name="pneumonia_detection_with_attention",
    output_dir="outputs/",
    
    # 数据配置
    data=DataConfigWithMask(
        csv_file="data/pneumonia.csv",
        image_dir="data/images/",
        mask_dir="data/masks/",
        return_mask=True,
    ),
    
    # 训练配置
    training=TrainingConfigWithAttention(
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        
        # 注意力监督配置
        attention_supervision=AttentionSupervisionConfig(
            enabled=True,
            method="mask",
            loss_weight=0.1,
            loss_type="kl",
            temperature=10.0,
        ),
        
        log_attention_every=100,
        save_attention_maps=True,
    ),
)

# 保存配置
import yaml
with open("config.yaml", "w") as f:
    yaml.dump(config.__dict__, f)
```

### 预设配置

```python
from med_core.configs.attention_config import (
    create_mask_supervised_config,
    create_cam_supervised_config,
    create_mil_config,
    create_bbox_supervised_config,
)

# 分割掩码监督配置
mask_config = create_mask_supervised_config(
    loss_weight=0.1,
    loss_type="kl",
)

# CAM 自监督配置
cam_config = create_cam_supervised_config(
    loss_weight=0.1,
    consistency_method="entropy",
)

# MIL 配置
mil_config = create_mil_config(
    loss_weight=0.1,
    patch_size=16,
)

# 边界框监督配置
bbox_config = create_bbox_supervised_config(
    loss_weight=0.1,
    bbox_format="xyxy",
)
```

---

## 📊 可视化

### 可视化注意力叠加

```python
from med_core.visualization.attention_viz import visualize_attention_overlay

fig = visualize_attention_overlay(
    image=image,
    attention=attention_weights,
    alpha=0.5,
    cmap="jet",
    title="注意力可视化",
    save_path="attention.png",
)
```

### 可视化监督效果

```python
from med_core.visualization.attention_viz import visualize_attention_comparison

fig = visualize_attention_comparison(
    image=image,
    attention_before=attention_before_supervision,
    attention_after=attention_after_supervision,
    target=mask,
    titles=["原图", "监督前", "监督后", "目标掩码"],
    save_path="comparison.png",
)
```

### 可视化损失组件

```python
from med_core.visualization.attention_viz import visualize_attention_supervision_loss

fig = visualize_attention_supervision_loss(
    image=image,
    attention=attention_weights,
    target=mask,
    loss_components={"main": 0.5, "smooth": 0.1},
    save_path="loss.png",
)
```

### 可视化 MIL 注意力

```python
from med_core.visualization.attention_viz import visualize_mil_attention

fig = visualize_mil_attention(
    image=image,
    patch_attention=patch_attention_weights,
    grid_size=(14, 14),
    top_k=5,
    save_path="mil_attention.png",
)
```

---

## 💡 最佳实践

### 1. 损失权重调整

```python
# 开始时使用较小的权重
attention_supervision = MaskSupervisedAttention(
    loss_weight=0.01,  # 👈 从小开始
)

# 训练稳定后逐渐增加
for epoch in range(100):
    if epoch > 20:
        attention_supervision.loss_weight = 0.1  # 增加权重
```

### 2. 渐进式训练

```python
# 前几个 epoch 不使用注意力监督
attention_supervision = MaskSupervisedAttention(
    loss_weight=0.1,
    enabled=False,  # 👈 先禁用
)

for epoch in range(100):
    if epoch >= 10:
        attention_supervision.enabled = True  # 👈 10 个 epoch 后启用
    
    # 训练...
```

### 3. 监控注意力质量

```python
from med_core.visualization.attention_viz import plot_attention_statistics

attention_history = []

for epoch in range(100):
    for batch in dataloader:
        # 训练...
        attention_history.append(outputs["attention_weights"][0])
    
    # 每个 epoch 结束后绘制统计
    if epoch % 10 == 0:
        fig = plot_attention_statistics(
            attention_history[-100:],  # 最近 100 步
            save_path=f"outputs/stats_epoch{epoch}.png",
        )
        plt.close(fig)
```

### 4. 数据增强注意事项

```python
# 图像和掩码需要使用相同的变换
from torchvision import transforms

# 定义变换
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 掩码变换（不包括归一化）
mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 👈 与图像相同的随机变换
    transforms.ToTensor(),
])

dataset = MedicalAttentionSupervisedDataset(
    ...,
    transform=image_transform,
    mask_transform=mask_transform,
)
```

### 5. 混合使用多种方法

```python
# 如果部分样本有掩码，部分没有
for images, tabular, labels, masks in dataloader:
    outputs = model(images, tabular)
    classification_loss = criterion(outputs["logits"], labels)
    
    # 检查哪些样本有掩码
    has_mask = masks is not None and not torch.all(masks == 0)
    
    if has_mask:
        # 使用掩码监督
        attention_loss = mask_supervision(
            attention_weights=outputs["attention_weights"],
            features=outputs["features"],
            targets=masks,
        )
    else:
        # 使用 CAM 自监督
        attention_loss = cam_supervision(
            attention_weights=outputs["attention_weights"],
            features=outputs["features"],
            classifier_weights=model.classifier.weight,
        )
    
    total_loss = classification_loss + attention_loss.total_loss
    total_loss.backward()
    optimizer.step()
```

---

## 🔍 故障排除

### 问题1: 注意力损失过大

**原因**: 损失权重太大或温度参数不合适

**解决方案**:
```python
# 减小损失权重
attention_supervision.loss_weight = 0.01  # 从 0.1 降到 0.01

# 调整温度参数
attention_supervision.temperature = 5.0  # 从 10.0 降到 5.0
```

### 问题2: 注意力不集中

**原因**: 一致性损失权重太小

**解决方案**:
```python
# 增加一致性损失权重
cam_supervision = CAMSelfSupervision(
    consistency_weight=2.0,  # 从 1.0 增加到 2.0
)
```

### 问题3: 训练不稳定

**原因**: 注意力监督过早引入

**解决方案**:
```python
# 延迟启用注意力监督
for epoch in range(100):
    if epoch < 20:
        attention_supervision.enabled = False
    else:
        attention_supervision.enabled = True
```

---

## 📖 参考资料

- [离线注意力引导方案文档](./offline-attention-guidance.md)
- [交互式引导路线图](./interactive-guidance-roadmap.md)
- [决策链研究报告](./decision-chain-research.md)

---

**最后更新**: 2026-02-13  
**版本**: v1.0
