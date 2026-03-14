# 多视图支持

**预计时间：25分钟**

本教程介绍如何使用 MedFusion 处理多视图医学影像数据。

## 什么是多视图

多视图是指同一患者的多张图像，常见于：

- **多角度 CT 扫描**: 轴位（axial）、冠状位（coronal）、矢状位（sagittal）
- **时间序列影像**: 治疗前后对比、疾病进展追踪
- **多模态影像**: 不同成像方式的组合
- **多切片影像**: 同一组织的多个切片

## 快速开始

### 1. 准备数据

CSV 格式（推荐）：

```csv
patient_id,axial_path,coronal_path,sagittal_path,age,gender,label
P001,/data/p001_axial.png,/data/p001_coronal.png,/data/p001_sagittal.png,45,M,1
P002,/data/p002_axial.png,/data/p002_coronal.png,,50,F,0
```

### 2. 配置模型

```yaml
# configs/multiview_config.yaml
data:
  enable_multiview: true
  view_names: ["axial", "coronal", "sagittal"]
  view_path_columns:
    axial: "axial_path"
    coronal: "coronal_path"
    sagittal: "sagittal_path"
  missing_view_strategy: "zero"  # zero, skip, duplicate
  require_all_views: false

model:
  vision:
    enable_multiview: true
    aggregator_type: "attention"  # mean, max, attention, cross_view_attention
    share_backbone_weights: true
    backbone: "resnet50"
```

### 3. 训练模型

```python
from med_core.configs import create_ct_multiview_config
from med_core.fusion import create_multiview_fusion_model
from med_core.trainers import create_multiview_trainer

# 创建配置
config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet50"
)

# 创建模型
model = create_multiview_fusion_model(
    vision_backbone_name="resnet50",
    tabular_input_dim=10,
    fusion_type="gated",
    num_classes=2,
    aggregator_type="attention",
    view_names=config.data.view_names
)

# 训练
trainer = create_multiview_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)
trainer.train()
```

## 视图聚合策略

### 1. Mean Pooling（平均池化）

最简单的方法，对所有视图取平均。

```yaml
model:
  vision:
    aggregator_type: "mean"
```

**优点**: 简单快速
**缺点**: 所有视图权重相同

### 2. Max Pooling（最大池化）

取每个特征维度的最大值。

```yaml
model:
  vision:
    aggregator_type: "max"
```

**优点**: 保留最显著特征
**缺点**: 可能丢失重要信息

### 3. Attention（注意力聚合，推荐）

自动学习每个视图的重要性。

```yaml
model:
  vision:
    aggregator_type: "attention"
```

**优点**: 自适应权重，性能最好
**缺点**: 增加少量参数

### 4. Cross-View Attention（跨视图注意力）

视图之间相互关注。

```yaml
model:
  vision:
    aggregator_type: "cross_view_attention"
```

**优点**: 捕获视图间关系
**缺点**: 计算量较大

### 5. Learned Weight（可学习权重）

每个视图固定权重。

```yaml
model:
  vision:
    aggregator_type: "learned_weight"
```

**优点**: 参数少，可解释
**缺点**: 不够灵活

## 处理缺失视图

### 策略 1: Zero（零填充，默认）

用零张量填充缺失视图。

```yaml
data:
  missing_view_strategy: "zero"
```

### 策略 2: Skip（跳过）

跳过包含缺失视图的样本。

```yaml
data:
  missing_view_strategy: "skip"
  require_all_views: true
```

### 策略 3: Duplicate（复制）

复制可用视图填充缺失视图。

```yaml
data:
  missing_view_strategy: "duplicate"
```

## 权重共享

### 共享权重（推荐）

所有视图使用同一个 backbone。

```yaml
model:
  vision:
    share_backbone_weights: true
```

**优点**: 参数少，训练快
**适用**: 视图相似（如 CT 多角度）

### 独立权重

每个视图使用独立 backbone。

```yaml
model:
  vision:
    share_backbone_weights: false
```

**优点**: 表达能力强
**适用**: 视图差异大（如不同模态）

## 实际案例

### 案例 1: CT 多角度扫描

```python
from med_core.configs import create_ct_multiview_config

config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet50"
)

# 数据配置
config.data.csv_path = "data/ct_multiview.csv"
config.data.view_path_columns = {
    "axial": "axial_path",
    "coronal": "coronal_path",
    "sagittal": "sagittal_path"
}
config.data.missing_view_strategy = "zero"

# 模型配置
config.model.vision.share_backbone_weights = True  # CT 视图相似
```

### 案例 2: 时间序列影像

```python
from med_core.configs import create_temporal_multiview_config

config = create_temporal_multiview_config(
    num_timepoints=3,  # 基线、中期、终点
    aggregator_type="cross_view_attention",  # 捕获时间关系
    backbone="resnet50"
)

config.data.view_names = ["baseline", "midpoint", "endpoint"]
config.data.view_path_columns = {
    "baseline": "baseline_path",
    "midpoint": "midpoint_path",
    "endpoint": "endpoint_path"
}
```

### 案例 3: 多模态影像

```python
# CT + MRI + PET
config = MultiViewExperimentConfig()
config.data.enable_multiview = True
config.data.view_names = ["ct", "mri", "pet"]
config.data.view_path_columns = {
    "ct": "ct_path",
    "mri": "mri_path",
    "pet": "pet_path"
}

# 不同模态使用独立权重
config.model.vision.share_backbone_weights = False
config.model.vision.aggregator_type = "attention"
```

## 渐进式视图训练

从少量视图开始，逐步增加视图数量。

```yaml
model:
  vision:
    use_progressive_view_training: true
    initial_views: ["axial"]  # 从单个视图开始
    add_views_every_n_epochs: 10  # 每 10 个 epoch 添加一个视图
```

```python
# 训练过程
# Epoch 0-9: 只使用 axial
# Epoch 10-19: 使用 axial + coronal
# Epoch 20+: 使用 axial + coronal + sagittal
```

## 视图特定预处理

为不同视图设置不同的预处理参数。

```yaml
model:
  vision:
    view_specific_preprocessing:
      axial:
        normalize: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      coronal:
        normalize: true
        mean: [0.500, 0.500, 0.500]
        std: [0.250, 0.250, 0.250]
```

## 获取注意力权重

用于可解释性分析。

```python
model.eval()
with torch.no_grad():
    outputs = model(images, return_intermediates=True)

# 获取视图注意力权重
view_attention = outputs.get("view_attention")  # (B, num_views)

# 可视化
import matplotlib.pyplot as plt

view_names = ["axial", "coronal", "sagittal"]
weights = view_attention[0].cpu().numpy()

plt.bar(view_names, weights)
plt.title("View Attention Weights")
plt.ylabel("Weight")
plt.savefig("view_attention.png")
```

## 性能优化

### 1. 减少内存占用

```yaml
# 使用更小的 backbone
model:
  vision:
    backbone: "resnet18"  # 而非 resnet50

# 减小批次大小
data:
  batch_size: 16  # 多视图会增加内存使用

# 启用混合精度
training:
  mixed_precision: true
```

### 2. 提升训练速度

```yaml
# 共享权重
model:
  vision:
    share_backbone_weights: true

# 使用更快的聚合策略
model:
  vision:
    aggregator_type: "mean"  # 而非 cross_view_attention
```

## 数据加载

### 使用 MultiViewDataset

```python
from med_core.datasets import MultiViewDataset

dataset = MultiViewDataset(
    csv_path="data/multiview.csv",
    view_names=["axial", "coronal", "sagittal"],
    view_path_columns={
        "axial": "axial_path",
        "coronal": "coronal_path",
        "sagittal": "sagittal_path"
    },
    label_column="label",
    missing_view_strategy="zero",
    transform=transform
)

# 数据格式
sample = dataset[0]
# sample = {
#     "images": {
#         "axial": Tensor(3, 224, 224),
#         "coronal": Tensor(3, 224, 224),
#         "sagittal": Tensor(3, 224, 224)
#     },
#     "label": 1,
#     "view_mask": Tensor([1, 1, 0])  # 第三个视图缺失
# }
```

### 自定义数据加载

```python
from torch.utils.data import Dataset

class CustomMultiViewDataset(Dataset):
    def __init__(self, data_dir, view_names):
        self.data_dir = data_dir
        self.view_names = view_names
        # ... 初始化代码 ...

    def __getitem__(self, idx):
        images = {}
        view_mask = []

        for view_name in self.view_names:
            image_path = self.get_image_path(idx, view_name)
            if os.path.exists(image_path):
                images[view_name] = self.load_image(image_path)
                view_mask.append(1)
            else:
                images[view_name] = torch.zeros(3, 224, 224)
                view_mask.append(0)

        return {
            "images": images,
            "label": self.labels[idx],
            "view_mask": torch.tensor(view_mask)
        }
```

## 向后兼容性

多视图模型完全兼容单视图输入。

```python
# 单视图输入
single_image = torch.randn(B, 3, 224, 224)
output = model(single_image, tabular)  # 正常工作

# 多视图输入
multi_images = {
    "axial": torch.randn(B, 3, 224, 224),
    "coronal": torch.randn(B, 3, 224, 224),
    "sagittal": torch.randn(B, 3, 224, 224)
}
output = model(multi_images, tabular)  # 也正常工作
```

## 常见问题

### Q1: 视图名称不匹配怎么办？

A: 确保数据集和模型使用相同的 view_names。

```python
# 错误
ValueError: Unexpected view names: {'view4'}. Expected: ['axial', 'coronal', 'sagittal']

# 解决
config.data.view_names = ["axial", "coronal", "sagittal"]
model = create_multiview_fusion_model(
    view_names=["axial", "coronal", "sagittal"]  # 必须匹配
)
```

### Q2: 内存不足怎么办？

A: 三种方法：
1. 减小批次大小
2. 使用更小的 backbone
3. 启用混合精度训练

### Q3: 如何处理视图数量不同的样本？

A: 使用 view_mask 标记可用视图。

```python
# 模型会自动处理
output = model(images, tabular, view_mask=view_mask)
```

### Q4: 哪种聚合策略最好？

A: 推荐顺序：
1. attention（最佳性能）
2. mean（快速原型）
3. cross_view_attention（视图相关性强）

## 性能对比

| 聚合策略 | 准确率 | 参数量 | 推理速度 |
|---------|--------|--------|---------|
| 单视图 | 85.2% | 基准 | 基准 |
| Mean | 87.5% | +0% | 1.0x |
| Max | 87.1% | +0% | 1.0x |
| Attention | 89.3% | +0.1% | 0.98x |
| Cross-View | 90.1% | +0.5% | 0.85x |

## 下一步

- [模型导出](13_model_export.md) - 导出多视图模型
- [Docker 部署](14_docker_deployment.md) - 容器化部署
- [生产环境清单](15_production_checklist.md) - 部署前检查

## 参考资源

- [多视图详细指南](/home/yixian/Projects/med-ml/medfusion/docs/guides/multiview/overview.md)
- [多视图集成测试](../../tests/test_multiview_integration.py)
- [API 文档](../../api/multiview.md)
