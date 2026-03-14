# Preprocessing API

医学图像预处理模块，提供专业的图像处理工具。

## 概述

Preprocessing 模块提供了医学影像专用的预处理功能：

- **强度归一化**: 标准化图像强度范围
- **对比度增强**: CLAHE 自适应直方图均衡化
- **ROI 提取**: 感兴趣区域裁剪
- **质量评估**: 图像质量检测和伪影识别
- **批量处理**: 高效的批量预处理流程

## 图像预处理

### ImagePreprocessor

图像预处理器类。

**功能：**
- 强度归一化
- 对比度增强
- 尺寸调整
- 中心裁剪
- 批量处理

**示例：**
```python
from med_core.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(
    target_size=(512, 512),
    normalize=True,
    apply_clahe=True
)

# 预处理单张图像
processed_image = preprocessor.process(image)

# 批量预处理
processed_images = preprocessor.process_batch(image_list)
```

### normalize_intensity

强度归一化函数。

**参数：**
- `image` (np.ndarray): 输入图像
- `method` (str): 归一化方法
  - `"minmax"` - 最小-最大归一化到 [0, 1]
  - `"zscore"` - Z-score 标准化
  - `"percentile"` - 百分位数归一化
- `clip_percentile` (tuple): 裁剪百分位数，默认 (1, 99)

**返回：**
- `np.ndarray`: 归一化后的图像

**示例：**
```python
from med_core.preprocessing import normalize_intensity

# 最小-最大归一化
normalized = normalize_intensity(image, method="minmax")

# Z-score 标准化
standardized = normalize_intensity(image, method="zscore")

# 百分位数归一化（去除异常值）
robust_normalized = normalize_intensity(
    image,
    method="percentile",
    clip_percentile=(2, 98)
)
```

### apply_clahe

应用 CLAHE（对比度受限自适应直方图均衡化）。

**参数：**
- `image` (np.ndarray): 输入图像
- `clip_limit` (float): 对比度限制，默认 2.0
- `tile_grid_size` (tuple): 网格大小，默认 (8, 8)

**返回：**
- `np.ndarray`: 增强后的图像

**特性：**
- 局部对比度增强
- 避免过度增强
- 适用于医学影像

**示例：**
```python
from med_core.preprocessing import apply_clahe

# 基础 CLAHE
enhanced = apply_clahe(image)

# 自定义参数
enhanced = apply_clahe(
    image,
    clip_limit=3.0,  # 更强的增强
    tile_grid_size=(16, 16)  # 更细的网格
)
```

### crop_center

中心裁剪图像。

**参数：**
- `image` (np.ndarray): 输入图像
- `crop_size` (tuple): 裁剪尺寸 (height, width)

**返回：**
- `np.ndarray`: 裁剪后的图像

**示例：**
```python
from med_core.preprocessing import crop_center

# 裁剪中心 224x224 区域
cropped = crop_center(image, crop_size=(224, 224))
```

## 质量评估

### QualityMetrics

图像质量指标类。

**计算指标：**
- `sharpness` - 清晰度（拉普拉斯方差）
- `contrast` - 对比度（标准差）
- `brightness` - 亮度（均值）
- `snr` - 信噪比
- `entropy` - 信息熵

**示例：**
```python
from med_core.preprocessing import QualityMetrics

metrics = QualityMetrics()

# 计算单张图像质量
quality = metrics.compute(image)
print(f"清晰度: {quality['sharpness']:.2f}")
print(f"对比度: {quality['contrast']:.2f}")
print(f"信噪比: {quality['snr']:.2f}")

# 批量计算
quality_scores = metrics.compute_batch(image_list)
```

### assess_image_quality

评估图像质量。

**参数：**
- `image` (np.ndarray): 输入图像
- `return_details` (bool): 是否返回详细指标，默认 False

**返回：**
- `float` 或 `dict`: 质量分数（0-1）或详细指标字典

**示例：**
```python
from med_core.preprocessing import assess_image_quality

# 获取质量分数
score = assess_image_quality(image)
print(f"质量分数: {score:.3f}")

# 获取详细指标
details = assess_image_quality(image, return_details=True)
print(details)
```

### detect_artifacts

检测图像伪影。

**参数：**
- `image` (np.ndarray): 输入图像
- `artifact_types` (list[str]): 要检测的伪影类型
  - `"motion"` - 运动伪影
  - `"noise"` - 噪声
  - `"truncation"` - 截断伪影
  - `"metal"` - 金属伪影

**返回：**
- `dict[str, bool]`: 伪影检测结果

**示例：**
```python
from med_core.preprocessing import detect_artifacts

# 检测所有伪影
artifacts = detect_artifacts(image)
print(f"运动伪影: {artifacts['motion']}")
print(f"噪声: {artifacts['noise']}")

# 检测特定伪影
artifacts = detect_artifacts(
    image,
    artifact_types=["motion", "metal"]
)
```

## 批量预处理

### 预处理流程

完整的批量预处理流程示例：

```python
from pathlib import Path
from med_core.preprocessing import (
    ImagePreprocessor,
    assess_image_quality,
    detect_artifacts
)
from PIL import Image
import numpy as np

# 1. 创建预处理器
preprocessor = ImagePreprocessor(
    target_size=(512, 512),
    normalize=True,
    apply_clahe=True
)

# 2. 批量处理
input_dir = Path("data/raw_images")
output_dir = Path("data/processed_images")
output_dir.mkdir(exist_ok=True)

quality_log = []

for img_path in input_dir.glob("*.png"):
    # 加载图像
    image = np.array(Image.open(img_path))

    # 质量检查
    quality_score = assess_image_quality(image)
    artifacts = detect_artifacts(image)

    # 记录质量信息
    quality_log.append({
        'filename': img_path.name,
        'quality_score': quality_score,
        'has_artifacts': any(artifacts.values())
    })

    # 跳过低质量图像
    if quality_score < 0.5:
        print(f"跳过低质量图像: {img_path.name}")
        continue

    # 预处理
    processed = preprocessor.process(image)

    # 保存
    output_path = output_dir / img_path.name
    Image.fromarray(processed).save(output_path)

# 3. 保存质量报告
import json
with open(output_dir / "quality_report.json", "w") as f:
    json.dump(quality_log, f, indent=2)

print(f"处理完成！共处理 {len(quality_log)} 张图像")
```

## 配置示例

在 YAML 配置文件中配置预处理：

```yaml
preprocessing:
  image:
    target_size: [512, 512]
    normalize: true
    normalization_method: percentile
    clip_percentile: [1, 99]

  enhancement:
    apply_clahe: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]

  quality_control:
    min_quality_score: 0.5
    detect_artifacts: true
    artifact_types:
      - motion
      - noise
      - truncation

  output:
    format: png
    compression: 9
```

## 使用场景

### CT 图像预处理

```python
from med_core.preprocessing import ImagePreprocessor, apply_clahe

# CT 专用预处理
preprocessor = ImagePreprocessor(
    target_size=(512, 512),
    normalize=True,
    apply_clahe=True
)

# 窗宽窗位调整（CT 特有）
def apply_window(image, window_center, window_width):
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed = np.clip(image, min_val, max_val)
    return (windowed - min_val) / (max_val - min_val)

# 肺窗
lung_window = apply_window(ct_image, window_center=-600, window_width=1500)
processed = preprocessor.process(lung_window)
```

### 病理切片预处理

```python
from med_core.preprocessing import normalize_intensity, apply_clahe

# 病理切片预处理
def preprocess_pathology(image):
    # 1. 色彩归一化
    normalized = normalize_intensity(image, method="percentile")

    # 2. 对比度增强
    enhanced = apply_clahe(normalized, clip_limit=2.0)

    # 3. 尺寸调整
    from PIL import Image
    pil_image = Image.fromarray(enhanced)
    resized = pil_image.resize((224, 224), Image.LANCZOS)

    return np.array(resized)

processed = preprocess_pathology(pathology_image)
```

### X 光图像预处理

```python
from med_core.preprocessing import (
    normalize_intensity,
    apply_clahe,
    assess_image_quality
)

# X 光预处理
def preprocess_xray(image):
    # 1. 质量检查
    quality = assess_image_quality(image)
    if quality < 0.6:
        print("警告: 图像质量较低")

    # 2. 强度归一化
    normalized = normalize_intensity(image, method="minmax")

    # 3. 强对比度增强
    enhanced = apply_clahe(
        normalized,
        clip_limit=3.0,
        tile_grid_size=(16, 16)
    )

    return enhanced

processed = preprocess_xray(xray_image)
```

## 最佳实践

**归一化方法选择：**
- CT/MRI: 使用 percentile 方法去除异常值
- X 光: 使用 minmax 方法
- 病理切片: 使用 zscore 方法

**CLAHE 参数：**
- 自然图像: clip_limit=2.0
- 低对比度图像: clip_limit=3.0-4.0
- 高对比度图像: clip_limit=1.0-1.5

**质量控制：**
- 设置最低质量阈值（0.5-0.6）
- 检测并标记伪影
- 记录质量报告用于审查

**批量处理：**
- 使用多进程加速
- 保存预处理参数
- 记录处理日志

## 参考

完整实现请参考：
- `/home/yixian/Projects/med-ml/medfusion/med_core/preprocessing/image.py` - 图像预处理
- `/home/yixian/Projects/med-ml/medfusion/med_core/preprocessing/quality.py` - 质量评估
