# Datasets API

医学多模态数据集模块，支持单视图和多视图数据加载。

## 概述

Datasets 模块提供了灵活的数据加载系统，支持：

- **单视图数据集**: 每个样本一张图像 + 表格数据
- **多视图数据集**: 每个样本多张图像（多角度、多切片、时间序列等）
- **注意力监督**: 支持注意力图监督信号
- **数据增强**: 医学影像专用的数据增强策略
- **数据清洗**: 自动检测和处理异常数据

## 基础类

### BaseMultimodalDataset

所有多模态数据集的抽象基类。

**参数：**
- `image_paths` (list[str | Path]): 图像文件路径列表
- `tabular_data` (np.ndarray | torch.Tensor): 表格特征数组 (N, num_features)
- `labels` (np.ndarray | torch.Tensor): 标签数组 (N,)
- `transform` (Any): 图像变换，默认 None
- `target_transform` (Any): 标签变换，默认 None

**方法：**
- `__len__()` - 返回数据集大小
- `__getitem__(idx)` - 获取单个样本 (image, tabular, label)
- `load_image(path)` - 加载图像（需子类实现）
- `get_tabular_dim()` - 返回表格特征维度
- `get_num_classes()` - 返回类别数
- `get_class_distribution()` - 返回类别分布
- `get_sample_weights()` - 计算样本权重（用于平衡采样）
- `subset(indices)` - 创建子集
- `get_statistics()` - 获取数据集统计信息

**示例：**
```python
from med_core.datasets import BaseMultimodalDataset

# 查看数据集统计
stats = dataset.get_statistics()
print(f"样本数: {stats['num_samples']}")
print(f"类别分布: {stats['class_distribution']}")

# 创建平衡采样器
sample_weights = dataset.get_sample_weights()
sampler = WeightedRandomSampler(sample_weights, len(dataset))
```

### MedicalMultimodalDataset

医学多模态数据集的具体实现。

**参数：**
- `image_paths` (list[str | Path]): 医学图像路径
- `tabular_data` (np.ndarray | torch.Tensor): 临床表格数据
- `labels` (np.ndarray | torch.Tensor): 诊断标签
- `transform` (Any): 图像变换
- `target_transform` (Any): 标签变换
- `feature_names` (list[str]): 特征名称列表
- `patient_ids` (list[str]): 患者 ID 列表

**特性：**
- 支持常见医学图像格式（JPEG, PNG, TIFF）
- 自动特征预处理（归一化、编码）
- 处理缺失数据
- 支持患者级别的数据划分

**示例：**
```python
from med_core.datasets import MedicalMultimodalDataset

# 从 CSV 创建数据集
dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data/patients.csv",
    image_dir="data/ct_scans",
    image_column="ct_path",
    target_column="diagnosis",
    numerical_features=["age", "bmi", "blood_pressure"],
    categorical_features=["gender", "smoking_status"],
    transform=train_transforms
)

# 访问样本
image, tabular, label = dataset[0]
print(f"图像形状: {image.shape}")
print(f"表格特征: {tabular.shape}")
print(f"标签: {label}")
```

## 多视图数据集

### BaseMultiViewDataset

多视图数据集的抽象基类。

**特性：**
- 支持每个样本多张图像
- 灵活的视图配置
- 自动处理不同数量的视图

### MedicalMultiViewDataset

医学多视图数据集实现。

**应用场景：**
- 多角度 CT 扫描
- 多切片 MRI
- 时间序列影像
- 病理切片的多视野

**示例：**
```python
from med_core.datasets import MedicalMultiViewDataset, MultiViewConfig

# 配置多视图
config = MultiViewConfig(
    max_views=8,  # 最多 8 个视图
    pad_mode="repeat",  # 不足时重复填充
    view_aggregation="attention"  # 使用注意力聚合
)

# 创建多视图数据集
dataset = MedicalMultiViewDataset(
    image_paths_list=[
        ["patient1_view1.jpg", "patient1_view2.jpg"],
        ["patient2_view1.jpg", "patient2_view2.jpg", "patient2_view3.jpg"],
    ],
    tabular_data=tabular_features,
    labels=labels,
    config=config,
    transform=train_transforms
)

# 获取样本
views, tabular, label = dataset[0]
print(f"视图数量: {len(views)}")
print(f"每个视图形状: {views[0].shape}")
```

### MultiViewConfig

多视图配置类。

**参数：**
- `max_views` (int): 最大视图数量
- `pad_mode` (str): 填充模式 ("repeat", "zero", "mean")
- `view_aggregation` (str): 聚合策略 ("mean", "max", "attention")

## 数据变换

### get_train_transforms

获取训练集数据增强。

**参数：**
- `image_size` (int | tuple): 图像大小，默认 224
- `augmentation_level` (str): 增强强度 ("light", "medium", "strong")

**返回：**
- `torchvision.transforms.Compose`: 变换组合

**示例：**
```python
from med_core.datasets import get_train_transforms

train_transforms = get_train_transforms(
    image_size=224,
    augmentation_level="medium"
)
```

### get_val_transforms

获取验证集数据变换（无增强）。

**参数：**
- `image_size` (int | tuple): 图像大小，默认 224

**示例：**
```python
from med_core.datasets import get_val_transforms

val_transforms = get_val_transforms(image_size=224)
```

### get_medical_augmentation

获取医学影像专用增强。

**特性：**
- 保持医学影像的诊断特征
- 避免过度变形
- 支持 CLAHE 对比度增强
- 支持弹性变形

**示例：**
```python
from med_core.datasets import get_medical_augmentation

medical_aug = get_medical_augmentation(
    image_size=512,
    use_clahe=True,  # 使用 CLAHE 增强对比度
    elastic_transform=True  # 弹性变形
)
```

## 数据加载工具

### create_dataloaders

创建训练、验证、测试数据加载器。

**参数：**
- `dataset`: 数据集对象
- `batch_size` (int): 批次大小
- `train_split` (float): 训练集比例，默认 0.7
- `val_split` (float): 验证集比例，默认 0.15
- `test_split` (float): 测试集比例，默认 0.15
- `num_workers` (int): 数据加载线程数，默认 4
- `pin_memory` (bool): 是否固定内存，默认 True
- `balanced_sampling` (bool): 是否使用平衡采样，默认 False

**返回：**
- `tuple[DataLoader, DataLoader, DataLoader]`: (train_loader, val_loader, test_loader)

**示例：**
```python
from med_core.datasets import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset=dataset,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_workers=8,
    balanced_sampling=True  # 处理类别不平衡
)

# 训练循环
for images, tabular, labels in train_loader:
    # 训练代码
    pass
```

### split_dataset

划分数据集为训练、验证、测试集。

**参数：**
- `dataset`: 数据集对象
- `train_ratio` (float): 训练集比例
- `val_ratio` (float): 验证集比例
- `test_ratio` (float): 测试集比例
- `stratify` (bool): 是否分层划分，默认 True
- `random_state` (int): 随机种子

**返回：**
- `tuple[Dataset, Dataset, Dataset]`: (train_set, val_set, test_set)

**示例：**
```python
from med_core.datasets import split_dataset

train_set, val_set, test_set = split_dataset(
    dataset=dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,  # 保持类别比例
    random_state=42
)
```

## 数据清洗

### DataCleaner

数据清洗工具类。

**功能：**
- 检测异常值
- 处理缺失值
- 移除重复样本
- 特征归一化

**示例：**
```python
from med_core.datasets import DataCleaner

cleaner = DataCleaner()

# 清洗表格数据
cleaned_data = cleaner.clean_tabular(
    data=raw_tabular_data,
    remove_outliers=True,
    fill_missing="median",
    normalize=True
)

# 检测图像质量
quality_scores = cleaner.assess_image_quality(image_paths)
valid_indices = quality_scores > 0.5  # 过滤低质量图像
```

## 配置示例

在 YAML 配置文件中配置数据集：

```yaml
data:
  csv_path: data/patients.csv
  image_dir: data/ct_scans
  image_column: ct_path
  target_column: diagnosis

  numerical_features:
    - age
    - bmi
    - blood_pressure

  categorical_features:
    - gender
    - smoking_status

  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

  augmentation:
    level: medium
    use_clahe: true

  dataloader:
    batch_size: 32
    num_workers: 8
    pin_memory: true
    balanced_sampling: true
```

多视图配置：

```yaml
data:
  multiview:
    enabled: true
    max_views: 8
    pad_mode: repeat
    view_aggregation: attention
```

## 最佳实践

**数据划分：**
- 使用分层划分保持类别比例
- 固定随机种子确保可复现性
- 患者级别划分避免数据泄露

**数据增强：**
- 训练集使用中等强度增强
- 验证/测试集仅做归一化
- 医学影像避免过度变形

**数据加载：**
- GPU 训练时设置 `pin_memory=True`
- 根据 CPU 核心数设置 `num_workers`
- 类别不平衡时使用 `balanced_sampling`

**性能优化：**
- 使用数据缓存加速重复加载
- 预处理数据保存为 HDF5 格式
- 多视图数据使用延迟加载

## 参考

完整实现请参考：
- `med_core/datasets/base.py` - 基础数据集类
- `med_core/datasets/medical.py` - 医学数据集实现
- `med_core/datasets/medical_multiview.py` - 多视图数据集
- `med_core/datasets/transforms.py` - 数据变换
- `med_core/datasets/data_cleaner.py` - 数据清洗
