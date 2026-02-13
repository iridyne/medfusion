# Med-Framework 多视图类型完全指南

## 目录
- [概述](#概述)
- [支持的多图片类型](#支持的多图片类型)
- [数据格式详解](#数据格式详解)
- [配置方法](#配置方法)
- [实际应用案例](#实际应用案例)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 概述

Med-Framework 的多视图支持允许为**单个患者**使用**多张图片**进行训练和推理。这是医学影像分析中的常见需求，因为：

- 医学影像通常从多个角度/时间点采集
- 多视图信息可以提供更全面的诊断依据
- 不同视图可能包含互补的临床信息

**核心优势：**
- ✅ 灵活支持任意数量和类型的视图
- ✅ 自动处理缺失视图
- ✅ 5 种视图聚合策略可选
- ✅ 完全向后兼容单视图模式

---

## 支持的多图片类型

### 1. 多角度 CT 扫描 ⭐ 最常用

**描述：** 同一患者的 CT 扫描从不同解剖平面采集的图像。

**典型视图：**
```python
view_names = ["axial", "coronal", "sagittal"]
```

- **axial（轴位）**：横断面，从头到脚的水平切片
- **coronal（冠状位）**：冠状面，从前到后的垂直切片
- **sagittal（矢状位）**：矢状面，从左到右的垂直切片

**应用场景：**
- 肺癌检测与分期
- 肝脏病变诊断
- 骨折检测
- 腹部器官评估

**配置示例：**
```python
from med_core.configs import create_ct_multiview_config

config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",  # 自动学习哪个角度更重要
    backbone="resnet50",
)
```

**CSV 数据格式：**
```csv
patient_id,axial_path,coronal_path,sagittal_path,age,gender,label
P001,/data/p001_axial.png,/data/p001_coronal.png,/data/p001_sagittal.png,55,M,1
P002,/data/p002_axial.png,/data/p002_coronal.png,/data/p002_sagittal.png,62,F,0
```

---

### 2. 时间序列影像

**描述：** 同一患者在不同时间点采集的影像，用于追踪疾病进展或治疗效果。

**典型视图：**

#### 2.1 治疗前后对比
```python
view_names = ["baseline", "followup"]
# 或
view_names = ["pre_treatment", "post_treatment"]
```

#### 2.2 疾病进展追踪
```python
view_names = ["timepoint_0", "timepoint_1", "timepoint_2"]
# 或
view_names = ["week_0", "week_4", "week_8", "week_12"]
```

**应用场景：**
- 肿瘤治疗效果评估（RECIST 标准）
- 慢性病进展监测（如阿尔茨海默病）
- 术后恢复追踪
- 药物疗效评估

**配置示例：**
```python
from med_core.configs import create_temporal_multiview_config

# 治疗前后对比
config = create_temporal_multiview_config(
    num_timepoints=2,
    aggregator_type="cross_view_attention",  # 学习时间点间的关系
    backbone="resnet50",
)

# 多时间点追踪
config = create_temporal_multiview_config(
    num_timepoints=4,  # 基线 + 3 次随访
    aggregator_type="attention",
    backbone="efficientnet_b0",
)
```

**CSV 数据格式：**
```csv
patient_id,baseline_path,followup_path,age,treatment,label
P001,/data/p001_t0.png,/data/p001_t1.png,45,chemotherapy,1
P002,/data/p002_t0.png,/data/p002_t1.png,58,radiation,0
```

---

### 3. 多模态影像组合

**描述：** 同一患者使用不同成像技术采集的图像。

**典型视图：**

#### 3.1 不同成像方式
```python
view_names = ["CT", "MRI", "PET"]
# 或
view_names = ["xray", "ultrasound"]
```

#### 3.2 MRI 不同序列
```python
view_names = ["T1", "T2", "FLAIR", "DWI"]
```

- **T1**：T1 加权像，解剖结构清晰
- **T2**：T2 加权像，水肿和病变敏感
- **FLAIR**：液体衰减反转恢复，抑制脑脊液信号
- **DWI**：弥散加权成像，检测急性梗死

**应用场景：**
- 脑肿瘤分类（多序列 MRI）
- 心脏病诊断（CT + MRI）
- 癌症分期（CT + PET）
- 多模态融合诊断

**配置示例：**
```python
from med_core.configs import MultiViewExperimentConfig

config = MultiViewExperimentConfig(
    project_name="brain-tumor-classification",
    experiment_name="multi-sequence-mri",
)

# 配置 MRI 多序列
config.data.enable_multiview = True
config.data.view_names = ["T1", "T2", "FLAIR", "DWI"]
config.data.view_path_columns = {
    "T1": "t1_path",
    "T2": "t2_path",
    "FLAIR": "flair_path",
    "DWI": "dwi_path",
}

config.model.vision.enable_multiview = True
config.model.vision.aggregator_type = "cross_view_attention"  # 学习序列间关系
config.model.vision.share_backbone_weights = False  # 不同模态用独立权重
```

**CSV 数据格式：**
```csv
patient_id,t1_path,t2_path,flair_path,dwi_path,age,tumor_type
P001,/data/p001_t1.nii,/data/p001_t2.nii,/data/p001_flair.nii,/data/p001_dwi.nii,45,glioma
P002,/data/p002_t1.nii,/data/p002_t2.nii,/data/p002_flair.nii,/data/p002_dwi.nii,62,meningioma
```

---

### 4. 多切片/多层影像

**描述：** 同一组织或器官的多个切片/层级图像。

**典型视图：**
```python
# 连续切片
view_names = ["slice_1", "slice_2", "slice_3", "slice_4", "slice_5"]

# 或关键层级
view_names = ["upper", "middle", "lower"]

# 或病灶周围切片
view_names = ["lesion_minus2", "lesion_minus1", "lesion_center", "lesion_plus1", "lesion_plus2"]
```

**应用场景：**
- 肺结节检测（多层 CT）
- 肝脏分割（连续切片）
- 脊柱病变诊断
- 3D 体积数据的 2D 切片表示

**配置示例：**
```python
config = MultiViewExperimentConfig(
    project_name="lung-nodule-detection",
    experiment_name="multi-slice",
)

config.data.enable_multiview = True
config.data.view_names = [f"slice_{i}" for i in range(1, 8)]  # 7 个切片
config.data.view_path_columns = {f"slice_{i}": f"slice_{i}_path" for i in range(1, 8)}

config.model.vision.enable_multiview = True
config.model.vision.aggregator_type = "attention"  # 自动学习哪些切片更重要
config.model.vision.share_backbone_weights = True  # 切片共享权重
```

**CSV 数据格式：**
```csv
patient_id,slice_1_path,slice_2_path,slice_3_path,slice_4_path,slice_5_path,nodule_present
P001,/data/p001_s1.png,/data/p001_s2.png,/data/p001_s3.png,/data/p001_s4.png,/data/p001_s5.png,1
P002,/data/p002_s1.png,/data/p002_s2.png,/data/p002_s3.png,/data/p002_s4.png,/data/p002_s5.png,0
```

---

### 5. 自定义任意视图

**描述：** 完全灵活的视图定义，适应任何特殊需求。

**典型视图：**
```python
# 乳腺 X 光的标准视图
view_names = ["CC", "MLO"]  # Craniocaudal, Mediolateral Oblique

# 皮肤病变的多角度照片
view_names = ["front", "back", "left", "right", "close_up"]

# 眼底照片的不同区域
view_names = ["macula", "optic_disc", "periphery"]

# 病理切片的不同染色
view_names = ["HE", "IHC_ki67", "IHC_p53"]
```

**配置示例：**
```python
# 乳腺癌检测
config.data.view_names = ["CC", "MLO"]
config.data.view_path_columns = {
    "CC": "cc_view_path",
    "MLO": "mlo_view_path",
}

# 皮肤病变分类
config.data.view_names = ["front", "back", "left", "right", "close_up"]
config.data.view_path_columns = {
    "front": "front_path",
    "back": "back_path",
    "left": "left_path",
    "right": "right_path",
    "close_up": "closeup_path",
}
```

**约束：**
- 最多支持 10 个视图（可通过 `max_views` 配置）
- 至少需要 1 个视图（可通过 `min_views` 配置）
- 视图名称可以是任意字符串

---

## 数据格式详解

### 支持的输入格式

Med-Framework 支持三种多视图数据格式：

#### 格式 1: 字典格式（推荐）⭐

```python
images = {
    "axial": torch.Tensor(B, 3, 224, 224),
    "coronal": torch.Tensor(B, 3, 224, 224),
    "sagittal": torch.Tensor(B, 3, 224, 224),
}
```

**优点：**
- 视图名称明确，可读性强
- 支持缺失视图（值为 None）
- 灵活处理不同数量的视图

**使用场景：** 所有场景，特别是视图可能缺失的情况

#### 格式 2: 堆叠张量

```python
images = torch.Tensor(B, N, 3, 224, 224)
# B = batch size
# N = 视图数量
```

**优点：**
- 内存连续，计算效率高
- 适合固定数量的视图

**使用场景：** 所有样本都有相同数量的视图

#### 格式 3: 单视图（向后兼容）

```python
images = torch.Tensor(B, 3, 224, 224)
```

**优点：**
- 与单视图模式完全兼容
- 无需修改现有代码

**使用场景：** 只有一个视图的情况

---

## 配置方法

### 方法 1: 使用预设配置（推荐）

```python
from med_core.configs import create_ct_multiview_config, create_temporal_multiview_config

# CT 多角度
config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet50",
)

# 时间序列
config = create_temporal_multiview_config(
    num_timepoints=3,
    aggregator_type="cross_view_attention",
    backbone="efficientnet_b0",
)
```

### 方法 2: 自定义配置

```python
from med_core.configs import MultiViewExperimentConfig

config = MultiViewExperimentConfig(
    project_name="my-project",
    experiment_name="my-experiment",
)

# 数据配置
config.data.enable_multiview = True
config.data.view_names = ["view1", "view2", "view3"]
config.data.view_path_columns = {
    "view1": "path_col1",
    "view2": "path_col2",
    "view3": "path_col3",
}
config.data.missing_view_strategy = "zero"  # skip, zero, duplicate
config.data.require_all_views = False

# 模型配置
config.model.vision.enable_multiview = True
config.model.vision.aggregator_type = "attention"  # max, mean, attention, cross_view_attention, learned_weight
config.model.vision.share_backbone_weights = True  # True=共享权重, False=独立权重
```

### 缺失视图处理策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `skip` | 跳过缺失视图的样本 | 所有视图都很重要，缺失会影响诊断 |
| `zero` | 用零张量填充（默认） | 模型能够学习忽略零张量 |
| `duplicate` | 复制第一个可用视图 | 缺失视图与其他视图相似 |

```python
config.data.missing_view_strategy = "zero"
```

### 视图聚合策略

| 策略 | 描述 | 参数量 | 计算复杂度 | 适用场景 |
|------|------|--------|-----------|----------|
| `max` | 最大池化 | 0 | O(N) | 快速原型，简单场景 |
| `mean` | 平均池化（支持 mask） | 0 | O(N) | 所有视图同等重要 |
| `attention` | 可学习注意力权重 | 少 | O(N) | **推荐**，自动学习重要性 |
| `cross_view_attention` | 跨视图自注意力 | 中 | O(N²) | 视图间有强相关性 |
| `learned_weight` | 每个视图独立权重 | 少 | O(N) | 视图重要性固定 |

```python
config.model.vision.aggregator_type = "attention"
```

---

## 实际应用案例

### 案例 1: 肺癌 CT 多角度诊断

```python
from med_core.configs import create_ct_multiview_config
from med_core.datasets import MedicalMultiViewDataset
from med_core.fusion import create_multiview_fusion_model
from med_core.trainers import create_multiview_trainer

# 1. 配置
config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet50",
)

# 2. 数据集
dataset = MedicalMultiViewDataset.from_csv_multiview(
    csv_path="lung_cancer_data.csv",
    view_columns={
        "axial": "axial_path",
        "coronal": "coronal_path",
        "sagittal": "sagittal_path",
    },
    tabular_columns=["age", "smoking_history", "tumor_size"],
    label_column="malignant",
    view_config=config.data,
)

# 3. 模型
model = create_multiview_fusion_model(
    vision_backbone_name="resnet50",
    tabular_input_dim=3,
    fusion_type="gated",
    num_classes=2,
    aggregator_type="attention",
    view_names=config.data.view_names,
)

# 4. 训练
trainer = create_multiview_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

trainer.train()
```

### 案例 2: 肿瘤治疗效果评估

```python
# 1. 配置
config = create_temporal_multiview_config(
    num_timepoints=2,  # 治疗前 + 治疗后
    aggregator_type="cross_view_attention",  # 学习时间点间的变化
    backbone="efficientnet_b0",
)

# 2. 数据集
dataset = MedicalMultiViewDataset.from_csv_multiview(
    csv_path="treatment_response.csv",
    view_columns={
        "pre_treatment": "baseline_path",
        "post_treatment": "followup_path",
    },
    tabular_columns=["age", "treatment_type", "dose"],
    label_column="response",  # 0=无效, 1=部分缓解, 2=完全缓解
    view_config=config.data,
)

# 3. 模型（3 分类）
model = create_multiview_fusion_model(
    vision_backbone_name="efficientnet_b0",
    tabular_input_dim=3,
    fusion_type="attention",
    num_classes=3,
    aggregator_type="cross_view_attention",
    view_names=["pre_treatment", "post_treatment"],
)
```

### 案例 3: 脑肿瘤 MRI 多序列分类

```python
# 1. 配置
config = MultiViewExperimentConfig(
    project_name="brain-tumor",
    experiment_name="multi-sequence-mri",
)

config.data.enable_multiview = True
config.data.view_names = ["T1", "T2", "FLAIR", "DWI"]
config.data.view_path_columns = {
    "T1": "t1_path",
    "T2": "t2_path",
    "FLAIR": "flair_path",
    "DWI": "dwi_path",
}
config.data.missing_view_strategy = "zero"

config.model.vision.enable_multiview = True
config.model.vision.aggregator_type = "cross_view_attention"
config.model.vision.share_backbone_weights = False  # 不同序列用独立权重

# 2. 数据集
dataset = MedicalMultiViewDataset.from_csv_multiview(
    csv_path="brain_tumor_data.csv",
    view_columns=config.data.view_path_columns,
    tabular_columns=["age", "gender", "tumor_location"],
    label_column="tumor_type",  # glioma, meningioma, pituitary
    view_config=config.data,
)

# 3. 模型
model = create_multiview_fusion_model(
    vision_backbone_name="resnet50",
    tabular_input_dim=3,
    fusion_type="gated",
    num_classes=3,
    aggregator_type="cross_view_attention",
    view_names=config.data.view_names,
    share_backbone_weights=False,
)
```

---

## 最佳实践

### 1. 选择合适的视图聚合策略

```python
# 快速原型 → max/mean
config.model.vision.aggregator_type = "max"

# 生产环境 → attention（推荐）
config.model.vision.aggregator_type = "attention"

# 视图间有时序关系 → cross_view_attention
config.model.vision.aggregator_type = "cross_view_attention"
```

### 2. 权重共享 vs 独立权重

```python
# 相似视图（如 CT 多角度）→ 共享权重
config.model.vision.share_backbone_weights = True

# 不同模态（如 CT + MRI）→ 独立权重
config.model.vision.share_backbone_weights = False
```

### 3. 处理缺失视图

```python
# 数据质量高，缺失少 → skip
config.data.missing_view_strategy = "skip"

# 数据质量一般，缺失较多 → zero（推荐）
config.data.missing_view_strategy = "zero"

# 缺失视图与其他视图相似 → duplicate
config.data.missing_view_strategy = "duplicate"
```

### 4. 渐进式视图训练

```python
# 从少量视图开始，逐步增加
config.model.vision.use_progressive_view_training = True
config.model.vision.initial_views = ["axial"]  # 先用单视图
config.model.vision.add_views_every_n_epochs = 10  # 每 10 epoch 添加一个视图
```

### 5. 数据增强

```python
# 为不同视图设置不同的增强策略
config.data.view_specific_augmentation = {
    "axial": {"rotation": 15, "flip": True},
    "coronal": {"rotation": 10, "flip": False},
    "sagittal": {"rotation": 10, "flip": False},
}
```

---

## 常见问题

### Q1: 最多支持多少个视图？

**A:** 默认最多 10 个视图，可通过配置修改：

```python
config.data.max_views = 20  # 增加到 20 个
```

### Q2: 如果某些样本缺少某些视图怎么办？

**A:** 使用 `missing_view_strategy` 配置：

```python
config.data.missing_view_strategy = "zero"  # 推荐
# 或
config.data.missing_view_strategy = "skip"  # 跳过缺失样本
```

### Q3: 不同视图的图像尺寸可以不同吗？

**A:** 不可以。所有视图必须预处理到相同尺寸（如 224×224）。

```python
config.data.image_size = (224, 224)  # 所有视图统一尺寸
```

### Q4: 如何可视化视图注意力权重？

**A:** 使用训练器的回调函数：

```python
# 训练后获取注意力权重
attention_weights = trainer.get_attention_weights()

# 可视化
import matplotlib.pyplot as plt
plt.bar(config.data.view_names, attention_weights)
plt.title("View Attention Weights")
plt.show()
```

### Q5: 单视图和多视图可以无缝切换吗？

**A:** 可以。只需修改配置：

```python
# 单视图模式
config.data.enable_multiview = False

# 多视图模式
config.data.enable_multiview = True
config.data.view_names = ["axial", "coronal", "sagittal"]
```

### Q6: 如何选择 backbone？

**A:** 根据数据量和计算资源：

```python
# 小数据集 → 轻量级模型
backbone = "resnet18" 或 "efficientnet_b0"

# 中等数据集 → 中等模型
backbone = "resnet50" 或 "efficientnet_b3"

# 大数据集 → 大模型
backbone = "resnet101" 或 "efficientnet_b7"
```

### Q7: 多视图会增加多少训练时间？

**A:** 取决于配置：

- **共享权重 + Max/Mean 聚合**：增加 10-20%
- **共享权重 + Attention 聚合**：增加 20-30%
- **独立权重 + CrossView Attention**：增加 50-100%

### Q8: 如何处理 3D 医学影像？

**A:** 将 3D 体积切片为多个 2D 图像：

```python
# 方法 1: 选择关键切片
view_names = ["slice_upper", "slice_middle", "slice_lower"]

# 方法 2: 均匀采样
view_names = [f"slice_{i}" for i in range(0, 100, 10)]  # 每 10 层采样一次
```

---

## 总结

Med-Framework 的多视图支持提供了：

✅ **5 种多图片类型**：CT 多角度、时间序列、多模态、多切片、自定义  
✅ **3 种数据格式**：字典、堆叠张量、单视图  
✅ **5 种聚合策略**：Max、Mean、Attention、CrossView、Learned  
✅ **3 种缺失处理**：Skip、Zero、Duplicate  
✅ **完全灵活配置**：支持任意视图名称和数量  

**推荐配置：**
- 聚合策略：`attention`
- 缺失处理：`zero`
- 权重共享：相似视图用 `True`，不同模态用 `False`

---

**文档版本：** 1.0  
**最后更新：** 2026-02-13  
**维护者：** Medical AI Research Team
