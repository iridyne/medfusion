# 数据准备指南

**预计时间：30分钟**

本教程详细讲解如何为 MedFusion 准备训练数据，包括 CSV 格式、图像组织、数据转换和增强策略。

## 数据组织概览

MedFusion 使用 **CSV + 图像文件** 的组织方式：

```
data/
├── metadata.csv          # 元数据文件
├── images/               # 图像目录
│   ├── patient001.png
│   ├── patient002.png
│   └── ...
└── masks/                # 可选：注意力监督掩码
    ├── patient001_mask.png
    └── ...
```

## 1. CSV 元数据文件

### 基本格式要求

CSV 文件必须包含以下列：

1. **图像路径列**：指向图像文件的路径
2. **标签列**：分类标签或回归目标
3. **特征列**：数值型和类别型临床特征

### 示例 1：基础分类任务

```csv
patient_id,image_path,age,gender,diagnosis
P001,images/patient001.png,45,M,0
P002,images/patient002.png,62,F,1
P003,images/patient003.png,38,M,0
P004,images/patient004.png,71,F,1
```

**列说明：**
- `patient_id`: 患者唯一标识符（可选，用于患者级别划分）
- `image_path`: 图像文件路径（相对于 `image_dir`）
- `age`: 数值型特征（年龄）
- `gender`: 类别型特征（性别）
- `diagnosis`: 标签列（0=良性，1=恶性）

### 示例 2：多特征任务

```csv
patient_id,image_path,age,gender,bmi,smoking,hypertension,diabetes,stage
P001,ct/001.npy,45,M,24.5,never,0,0,I
P002,ct/002.npy,62,F,28.3,former,1,1,III
P003,ct/003.npy,38,M,22.1,current,0,0,II
```

**特征类型：**
- 数值型：`age`, `bmi`
- 类别型：`gender`, `smoking`, `hypertension`, `diabetes`
- 标签：`stage`（多分类：I, II, III, IV）

### 示例 3：生存分析任务

```csv
patient_id,image_path,age,gender,time,event
P001,images/001.png,45,M,365,0
P002,images/002.png,62,F,180,1
P003,images/003.png,38,M,730,0
```

**生存分析列：**
- `time`: 生存时间（天数）
- `event`: 事件发生标志（0=删失，1=事件发生）

### 示例 4：多视图数据

```csv
patient_id,view1_path,view2_path,view3_path,age,gender,label
P001,ct/001_axial.png,ct/001_coronal.png,ct/001_sagittal.png,45,M,0
P002,ct/002_axial.png,ct/002_coronal.png,ct/002_sagittal.png,62,F,1
```

**多视图配置：**
- 每个视图使用独立的列
- 配置文件中需要指定多个 `image_path_column`

## 2. 图像文件组织

### 支持的图像格式

- **2D 图像**：PNG, JPG, JPEG, BMP, TIFF
- **医学图像**：DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`)
- **NumPy 数组**：`.npy`, `.npz`

### 组织方式 1：扁平结构

```
data/images/
├── patient001.png
├── patient002.png
├── patient003.png
└── ...
```

**CSV 配置：**
```csv
image_path
patient001.png
patient002.png
patient003.png
```

**Config 配置：**
```yaml
data:
  image_dir: "data/images"
  image_path_column: "image_path"
```

### 组织方式 2：分层结构

```
data/
├── train/
│   ├── benign/
│   │   ├── 001.png
│   │   └── 002.png
│   └── malignant/
│       ├── 003.png
│       └── 004.png
└── test/
    └── ...
```

**CSV 配置：**
```csv
image_path,label
train/benign/001.png,0
train/benign/002.png,0
train/malignant/003.png,1
train/malignant/004.png,1
```

### 组织方式 3：患者分组

```
data/images/
├── patient001/
│   ├── slice_001.png
│   ├── slice_002.png
│   └── slice_003.png
├── patient002/
│   └── ...
```

**用于多切片/多实例学习（MIL）**

## 3. 数据转换

### DICOM 转 PNG

```python
import pydicom
from PIL import Image
import numpy as np

def dicom_to_png(dicom_path, output_path):
    """将 DICOM 文件转换为 PNG"""
    # 读取 DICOM
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array

    # 归一化到 0-255
    pixel_array = pixel_array.astype(float)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    pixel_array = (pixel_array * 255).astype(np.uint8)

    # 保存为 PNG
    Image.fromarray(pixel_array).save(output_path)

# 批量转换
import glob
from pathlib import Path

dicom_dir = Path("data/raw_dicom")
output_dir = Path("data/images")
output_dir.mkdir(exist_ok=True)

for dicom_file in dicom_dir.glob("*.dcm"):
    output_file = output_dir / f"{dicom_file.stem}.png"
    dicom_to_png(dicom_file, output_file)
    print(f"Converted: {dicom_file.name} -> {output_file.name}")
```

### DICOM 转 NumPy（保留原始值）

```python
def dicom_to_npy(dicom_path, output_path):
    """将 DICOM 转换为 NumPy 数组（保留 HU 值）"""
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array

    # 应用 Rescale Slope 和 Intercept（转换为 HU 值）
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    # 保存为 .npy
    np.save(output_path, pixel_array)

# 批量转换
for dicom_file in dicom_dir.glob("*.dcm"):
    output_file = output_dir / f"{dicom_file.stem}.npy"
    dicom_to_npy(dicom_file, output_file)
```

### NIfTI 转 PNG（提取切片）

```python
import nibabel as nib

def nifti_to_png_slices(nifti_path, output_dir, axis=2):
    """将 NIfTI 3D 体积转换为 2D 切片"""
    # 读取 NIfTI
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # 提取切片
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    num_slices = data.shape[axis]
    for i in range(num_slices):
        if axis == 0:
            slice_data = data[i, :, :]
        elif axis == 1:
            slice_data = data[:, i, :]
        else:
            slice_data = data[:, :, i]

        # 归一化
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        slice_data = (slice_data * 255).astype(np.uint8)

        # 保存
        output_file = output_dir / f"slice_{i:03d}.png"
        Image.fromarray(slice_data).save(output_file)

# 使用示例
nifti_to_png_slices("data/ct_scan.nii.gz", "data/slices/patient001", axis=2)
```

### 使用 MedFusion 预处理工具

```bash
# 使用内置预处理命令
uv run medfusion-preprocess \
    --input-dir data/raw_dicom \
    --output-dir data/processed \
    --format png \
    --normalize true \
    --resize 224
```

## 4. 数据增强策略

### 配置文件中的增强

```yaml
data:
  use_augmentation: true
  augmentation_strength: "medium"  # "light", "medium", "heavy"
```

### 增强强度对比

**Light（轻度）：**
- 随机水平翻转（p=0.5）
- 随机旋转（±10°）
- 轻微亮度/对比度调整（±10%）

**Medium（中度，推荐）：**
- 随机水平/垂直翻转（p=0.5）
- 随机旋转（±15°）
- 随机缩放（0.9-1.1）
- 亮度/对比度/饱和度调整（±20%）
- 轻微高斯模糊

**Heavy（重度）：**
- 所有 Medium 增强
- 随机旋转（±30°）
- 随机缩放（0.8-1.2）
- 随机裁剪和调整大小
- 弹性变换
- 网格扭曲

### 自定义增强管道

```python
from torchvision import transforms
from med_core.datasets import MedicalDataset

# 自定义增强
custom_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 应用到数据集
dataset = MedicalDataset(
    csv_path="data/metadata.csv",
    image_dir="data/images",
    transform=custom_transform
)
```

### 医学图像特定增强

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 医学图像增强（使用 Albumentations）
medical_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
    ], p=0.3),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Equalize(),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

## 5. 处理类别不平衡

### 方法 1：类别权重

```yaml
training:
  class_weights: [1.0, 3.0]  # 第二类权重是第一类的 3 倍
```

**自动计算类别权重：**
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# 读取数据
df = pd.read_csv("data/metadata.csv")
labels = df["diagnosis"].values

# 计算权重
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

print(f"Class weights: {class_weights.tolist()}")
# 输出: Class weights: [0.6, 2.4]
```

### 方法 2：过采样

```python
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

# 读取数据
df = pd.read_csv("data/metadata.csv")

# 过采样
ros = RandomOverSampler(random_state=42)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_resampled, y_resampled = ros.fit_resample(X, y)

# 保存平衡后的数据
df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
df_balanced.to_csv("data/metadata_balanced.csv", index=False)

print(f"Original: {y.value_counts().to_dict()}")
print(f"Resampled: {y_resampled.value_counts().to_dict()}")
```

### 方法 3：欠采样

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

### 方法 4：SMOTE（合成少数类过采样）

```python
from imblearn.over_sampling import SMOTE

# 注意：SMOTE 仅适用于表格特征，不适用于图像
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

## 6. 数据集划分

### 方法 1：随机划分（配置文件）

```yaml
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
```

MedFusion 会自动按比例划分数据集。

### 方法 2：患者级别划分

```yaml
data:
  patient_id_column: "patient_id"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

**重要**：指定 `patient_id_column` 后，划分会确保同一患者的所有样本在同一集合中，避免数据泄漏。

### 方法 3：预定义划分

```csv
patient_id,image_path,age,gender,diagnosis,split
P001,images/001.png,45,M,0,train
P002,images/002.png,62,F,1,train
P003,images/003.png,38,M,0,val
P004,images/004.png,71,F,1,test
```

```python
# 在代码中使用预定义划分
import pandas as pd

df = pd.read_csv("data/metadata.csv")
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]
```

### 方法 4：分层划分（保持类别比例）

```python
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/metadata.csv")

# 第一次划分：train + (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["diagnosis"],
    random_state=42
)

# 第二次划分：val + test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["diagnosis"],
    random_state=42
)

# 添加 split 列
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

# 合并并保存
df_final = pd.concat([train_df, val_df, test_df])
df_final.to_csv("data/metadata_split.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

## 7. 多模态数据组织

### 场景 1：图像 + 表格

```csv
patient_id,image_path,age,gender,bmi,diagnosis
P001,images/001.png,45,M,24.5,0
P002,images/002.png,62,F,28.3,1
```

**配置：**
```yaml
data:
  csv_path: "data/metadata.csv"
  image_dir: "data"
  image_path_column: "image_path"
  numerical_features: ["age", "bmi"]
  categorical_features: ["gender"]
  target_column: "diagnosis"
```

### 场景 2：多模态图像

```csv
patient_id,ct_path,xray_path,age,gender,diagnosis
P001,ct/001.npy,xray/001.png,45,M,0
P002,ct/002.npy,xray/002.png,62,F,1
```

**配置：**
```yaml
data:
  csv_path: "data/metadata.csv"
  image_dir: "data"
  modalities:
    ct:
      image_path_column: "ct_path"
      image_channels: 1
    xray:
      image_path_column: "xray_path"
      image_channels: 1
```

### 场景 3：多视图 CT

```csv
patient_id,axial_path,coronal_path,sagittal_path,diagnosis
P001,ct/001_ax.png,ct/001_cor.png,ct/001_sag.png,0
P002,ct/002_ax.png,ct/002_cor.png,ct/002_sag.png,1
```

### 场景 4：病理切片（MIL）

```
data/pathology/
├── patient001/
│   ├── patch_001.png
│   ├── patch_002.png
│   └── ...
├── patient002/
│   └── ...
```

```csv
patient_id,patch_dir,num_patches,diagnosis
P001,pathology/patient001,50,0
P002,pathology/patient002,45,1
```

## 8. 数据验证

### 检查 CSV 完整性

```python
import pandas as pd
from pathlib import Path

def validate_csv(csv_path, image_dir):
    """验证 CSV 文件和图像文件"""
    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)

    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # 检查缺失值
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values:")
        print(missing[missing > 0])

    # 检查图像文件是否存在
    if "image_path" in df.columns:
        missing_images = []
        for idx, row in df.iterrows():
            img_path = image_dir / row["image_path"]
            if not img_path.exists():
                missing_images.append(row["image_path"])

        if missing_images:
            print(f"\nMissing images: {len(missing_images)}")
            print(missing_images[:5])  # 显示前 5 个
        else:
            print("\nAll images exist!")

    # 检查标签分布
    if "diagnosis" in df.columns:
        print("\nLabel distribution:")
        print(df["diagnosis"].value_counts())

# 使用示例
validate_csv("data/metadata.csv", "data/images")
```

### 检查图像质量

```python
from PIL import Image
import numpy as np

def check_image_quality(image_dir):
    """检查图像质量"""
    image_dir = Path(image_dir)

    sizes = []
    corrupted = []

    for img_path in image_dir.glob("*.png"):
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
        except Exception as e:
            corrupted.append(str(img_path))

    if corrupted:
        print(f"Corrupted images: {len(corrupted)}")
        print(corrupted[:5])

    # 统计图像尺寸
    unique_sizes = set(sizes)
    print(f"\nUnique image sizes: {unique_sizes}")

    if len(unique_sizes) > 1:
        print("Warning: Images have different sizes!")

check_image_quality("data/images")
```

## 9. 完整示例：准备肺癌数据集

```python
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import pydicom

# 1. 转换 DICOM 到 PNG
def prepare_lung_cancer_dataset():
    raw_dir = Path("data/raw_dicom")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    metadata = []

    for dicom_file in raw_dir.glob("*.dcm"):
        # 读取 DICOM
        ds = pydicom.dcmread(dicom_file)
        pixel_array = ds.pixel_array

        # 归一化
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        pixel_array = (pixel_array * 255).astype(np.uint8)

        # 保存 PNG
        output_file = output_dir / f"{dicom_file.stem}.png"
        Image.fromarray(pixel_array).save(output_file)

        # 提取元数据
        patient_id = ds.PatientID if hasattr(ds, 'PatientID') else dicom_file.stem
        age = ds.PatientAge if hasattr(ds, 'PatientAge') else None
        gender = ds.PatientSex if hasattr(ds, 'PatientSex') else None

        metadata.append({
            "patient_id": patient_id,
            "image_path": f"processed/{output_file.name}",
            "age": age,
            "gender": gender,
        })

    # 2. 创建 CSV
    df = pd.DataFrame(metadata)

    # 添加模拟标签（实际应从临床数据获取）
    df["diagnosis"] = np.random.randint(0, 2, len(df))

    # 3. 划分数据集
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["diagnosis"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["diagnosis"], random_state=42)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    df_final = pd.concat([train_df, val_df, test_df])
    df_final.to_csv("data/lung_cancer_metadata.csv", index=False)

    print(f"Dataset prepared: {len(df_final)} samples")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

prepare_lung_cancer_dataset()
```

## 最佳实践

1. **使用患者级别划分**：避免同一患者的数据出现在训练集和测试集中
2. **保留原始数据**：转换后保留原始 DICOM 文件
3. **记录预处理步骤**：在 README 中记录所有预处理操作
4. **验证数据质量**：训练前检查图像完整性和标签分布
5. **版本控制元数据**：将 CSV 文件纳入 Git 管理
6. **使用相对路径**：CSV 中使用相对路径，便于迁移
7. **标准化命名**：使用一致的文件命名规范

## 常见问题

**Q: 图像尺寸不一致怎么办？**
A: MedFusion 会自动调整图像大小到 `image_size`，无需手动处理。

**Q: 如何处理 3D CT 体积？**
A: 可以提取 2D 切片，或使用 3D 骨干网络（如 `swin3d_tiny`）。

**Q: 类别严重不平衡怎么办？**
A: 使用类别权重、过采样或欠采样。

**Q: 如何处理缺失的临床特征？**
A: 使用均值/中位数填充，或使用专门的缺失值处理方法。

## 下一步

- [模型构建器 API](05_builder_api.md) - 学习如何构建模型
- [配置文件详解](03_understanding_configs.md) - 深入理解配置系统
