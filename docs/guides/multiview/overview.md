# Multi-View Support in Med-Framework

Med-Framework 现已支持多视图（Multi-View）医学影像处理，允许单个患者使用多张图片进行训练和推理。

## 概述

多视图支持使框架能够处理以下场景：

- **多角度 CT 扫描**：轴位（axial）、冠状位（coronal）、矢状位（sagittal）
- **时间序列影像**：治疗前后对比、疾病进展追踪
- **多模态影像组合**：不同成像方式的组合
- **多切片/多层影像**：同一组织的多个切片

## 核心特性

### 1. 灵活的数据结构

支持三种输入格式：

```python
# 格式 1: 字典格式（推荐）
images = {
    "axial": torch.Tensor(B, 3, 224, 224),
    "coronal": torch.Tensor(B, 3, 224, 224),
    "sagittal": torch.Tensor(B, 3, 224, 224),
}

# 格式 2: 堆叠张量
images = torch.Tensor(B, N, 3, 224, 224)  # N = 视图数量

# 格式 3: 单视图（向后兼容）
images = torch.Tensor(B, 3, 224, 224)
```

### 2. 视图聚合策略

提供 5 种视图聚合方法：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `max` | 最大池化 | 简单快速，适合初步实验 |
| `mean` | 平均池化（支持 mask） | 处理缺失视图 |
| `attention` | 可学习注意力权重 | **推荐**，自动学习视图重要性 |
| `cross_view_attention` | 跨视图自注意力 | 视图间有强相关性 |
| `learned_weight` | 每个视图独立权重 | 视图重要性固定 |

### 3. 缺失视图处理

支持三种策略处理缺失视图：

- **`skip`**：跳过缺失视图的样本
- **`zero`**：用零张量填充（默认）
- **`duplicate`**：复制可用视图

### 4. 权重共享选项

- **共享权重**（默认）：所有视图使用同一个 backbone，参数更少
- **独立权重**：每个视图使用独立 backbone，表达能力更强

## 快速开始

### 基本用法

```python
from med_core.configs import create_ct_multiview_config
from med_core.fusion import create_multiview_fusion_model
from med_core.trainers import create_multiview_trainer

# 1. 创建配置
config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet18",
)

# 2. 创建模型
model = create_multiview_fusion_model(
    vision_backbone_name="resnet18",
    tabular_input_dim=10,
    fusion_type="gated",
    num_classes=2,
    aggregator_type="attention",
    view_names=config.data.view_names,
)

# 3. 创建训练器
trainer = create_multiview_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

# 4. 训练
trainer.train()
```

### 数据集准备

#### CSV 格式（推荐）

```csv
patient_id,axial_path,coronal_path,sagittal_path,age,gender,label
P001,/path/to/p001_axial.png,/path/to/p001_coronal.png,/path/to/p001_sagittal.png,45,M,1
P002,/path/to/p002_axial.png,/path/to/p002_coronal.png,,50,F,0
```

#### 配置数据路径

```python
from med_core.configs import MultiViewDataConfig

data_config = MultiViewDataConfig(
    enable_multiview=True,
    view_names=["axial", "coronal", "sagittal"],
    view_path_columns={
        "axial": "axial_path",
        "coronal": "coronal_path",
        "sagittal": "sagittal_path",
    },
    missing_view_strategy="zero",
    require_all_views=False,
)
```

## 高级功能

### 1. 渐进式视图训练

从少量视图开始，逐步增加视图数量：

```python
from med_core.configs import MultiViewVisionConfig

vision_config = MultiViewVisionConfig(
    enable_multiview=True,
    aggregator_type="attention",
    use_progressive_view_training=True,
    initial_views=["axial"],  # 从单个视图开始
    add_views_every_n_epochs=10,  # 每 10 个 epoch 添加一个视图
)
```

### 2. 视图特定预处理

为不同视图设置不同的预处理参数：

```python
vision_config.view_specific_preprocessing = {
    "axial": {"normalize": True, "mean": [0.485, 0.456, 0.406]},
    "coronal": {"normalize": True, "mean": [0.500, 0.500, 0.500]},
}
```

### 3. 获取注意力权重

用于可解释性分析：

```python
# 训练后获取注意力权重
attention_weights = model.get_attention_weights()

# 包含：
# - fusion_attention: 融合层注意力
# - view_attention: 视图聚合注意力
```

### 4. 时间序列配置

用于治疗前后对比：

```python
from med_core.configs import create_temporal_multiview_config

config = create_temporal_multiview_config(
    num_timepoints=2,  # 治疗前 + 治疗后
    aggregator_type="attention",
    backbone="resnet18",
)
```

## 配置预设

### CT 多角度扫描

```python
config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet18",
)
```

### 时间序列影像

```python
config = create_temporal_multiview_config(
    num_timepoints=3,  # 基线、中期、终点
    aggregator_type="cross_view_attention",
    backbone="resnet50",
)
```

### 自定义配置

```python
from med_core.configs import MultiViewExperimentConfig

config = MultiViewExperimentConfig(
    project_name="my-multiview-project",
    experiment_name="custom-experiment",
)

# 数据配置
config.data.enable_multiview = True
config.data.view_names = ["view1", "view2", "view3"]
config.data.view_path_columns = {
    "view1": "path_col1",
    "view2": "path_col2",
    "view3": "path_col3",
}

# 模型配置
config.model.vision.enable_multiview = True
config.model.vision.aggregator_type = "attention"
config.model.vision.share_backbone_weights = True
```

## 架构细节

### 数据流

```
多视图输入 (dict or tensor)
    ↓
MultiViewVisionBackbone
    ├─ 每个视图通过 backbone 提取特征
    ├─ 视图聚合器（Attention/Mean/Max/etc.）
    └─ 输出聚合特征 (B, feature_dim)
    ↓
Fusion Module (与 tabular 特征融合)
    ↓
Classifier
    ↓
输出 (logits, 辅助输出)
```

### 关键组件

1. **MultiViewVisionBackbone** (`med_core.backbones.multiview_vision`)
   - 包装任意视觉 backbone
   - 处理多视图输入
   - 集成视图聚合器

2. **ViewAggregator** (`med_core.backbones.view_aggregator`)
   - 5 种聚合策略
   - 支持 view_mask
   - 返回注意力权重

3. **MultiViewMultiModalFusionModel** (`med_core.fusion.multiview_model`)
   - 完整的多模态融合模型
   - 支持辅助头
   - 可解释性输出

4. **MultiViewMultimodalTrainer** (`med_core.trainers.multiview_trainer`)
   - 处理多视图批次
   - 渐进式视图训练
   - 注意力权重日志

## 性能优化建议

### 1. 权重共享

```python
# 推荐：共享权重（更少参数，更快训练）
share_backbone_weights=True

# 仅在视图差异极大时使用独立权重
share_backbone_weights=False
```

### 2. 聚合策略选择

- **快速原型**：使用 `mean` 或 `max`
- **生产环境**：使用 `attention`（最佳性能）
- **视图相关性强**：使用 `cross_view_attention`

### 3. 批次大小

多视图会增加内存使用，建议：

```python
# 单视图批次大小 32 → 多视图批次大小 16-24
config.data.batch_size = 16
```

### 4. 混合精度训练

```python
config.training.mixed_precision = True  # 节省内存
```

## 向后兼容性

所有多视图组件完全向后兼容单视图输入：

```python
# 单视图输入仍然有效
single_image = torch.randn(B, 3, 224, 224)
output = model(single_image, tabular)  # ✓ 正常工作
```

## 示例代码

完整示例请参考：
- `tests/test_multiview_integration.py` - 集成测试
- `demos/multimodal-demo/` - 实际应用示例

## 故障排除

### 问题：视图名称不匹配

```python
# 错误
ValueError: Unexpected view names: {'view4'}. Expected: ['axial', 'coronal', 'sagittal']

# 解决：确保数据集和模型使用相同的 view_names
model = create_multiview_fusion_model(
    view_names=["axial", "coronal", "sagittal"],  # 必须匹配
    ...
)
```

### 问题：内存不足

```python
# 解决方案 1：减小批次大小
config.data.batch_size = 8

# 解决方案 2：启用混合精度
config.training.mixed_precision = True

# 解决方案 3：使用更小的 backbone
config.model.vision.backbone = "resnet18"  # 而非 resnet50
```

### 问题：缺失视图导致错误

```python
# 解决：设置缺失视图策略
config.data.missing_view_strategy = "zero"  # 或 "skip" 或 "duplicate"
config.data.require_all_views = False
```

## 下一步

- 查看 `AGENTS.md` 了解框架整体架构
- 运行 `tests/test_multiview_integration.py` 验证安装
- 参考 `demos/` 目录中的实际应用示例
- 阅读各模块的 docstring 了解详细 API

## 更新日志

- **2026-02-13**: 初始多视图支持发布
  - 数据层、模型层、配置层���训练层完整实现
  - 5 种视图聚合策略
  - 渐进式视图训练
  - 完整的向后兼容性
