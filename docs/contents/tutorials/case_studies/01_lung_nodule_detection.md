# 案例研究 1：肺结节检测

**预计时间：60 分钟**

## 1. 医学背景

肺结节是肺部组织中的小圆形或椭圆形病变，直径通常小于 3 厘米。早期检测肺结节对肺癌的诊断和治疗至关重要，因为早期肺癌的 5 年生存率可达 70-90%，而晚期仅为 10-20%。

**临床挑战：**
- 结节大小差异大（3-30mm）
- 形态多样（实性、磨玻璃、混合）
- 与血管、支气管等结构易混淆
- 需要 3D 空间信息进行准确判断

## 2. 数据集介绍

本案例使用 LIDC-IDRI（Lung Image Database Consortium）数据集的子集。该数据集包含 1018 例胸部 CT 扫描，由 4 位放射科医师标注。

**数据特点：**
- 图像格式：DICOM
- 分辨率：512×512×(100-500) 切片
- 层厚：1-5mm
- 标注信息：结节位置、大小、恶性程度评分

## 3. 环境准备

```bash
# 安装依赖
uv sync

# 下载示例数据（使用合成数据演示）
uv run python scripts/generate_synthetic_lung_data.py --output-dir data/lung_nodule
```

## 4. 数据预处理

### 4.1 创建预处理脚本

```python
# scripts/preprocess_lung_ct.py
import numpy as np
import torch
from pathlib import Path
from med_core.preprocessing import CTPreprocessor
from med_core.utils.io import load_dicom_series, save_nifti

def preprocess_lung_ct(input_dir: Path, output_dir: Path):
    """预处理肺部 CT 数据"""
    preprocessor = CTPreprocessor(
        target_spacing=(1.0, 1.0, 1.0),  # 重采样到 1mm 各向同性
        window_center=-600,  # 肺窗窗位
        window_width=1500,   # 肺窗窗宽
        clip_range=(-1000, 400),
        normalize=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for patient_dir in input_dir.glob("*"):
        if not patient_dir.is_dir():
            continue

        print(f"Processing {patient_dir.name}...")

        # 加载 DICOM 序列
        volume, spacing = load_dicom_series(patient_dir)

        # 预处理
        processed = preprocessor(volume, spacing)

        # 保存为 NIfTI 格式
        output_path = output_dir / f"{patient_dir.name}.nii.gz"
        save_nifti(processed, output_path, spacing=(1.0, 1.0, 1.0))

if __name__ == "__main__":
    preprocess_lung_ct(
        input_dir=Path("data/lung_nodule/raw"),
        output_dir=Path("data/lung_nodule/processed")
    )
```

运行预处理：

```bash
uv run python scripts/preprocess_lung_ct.py
```

### 4.2 创建数据集类

```python
# examples/lung_nodule_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import nibabel as nib

class LungNoduleDataset(Dataset):
    """肺结节检测数据集"""

    def __init__(self, data_dir: Path, split: str = "train", patch_size: tuple = (64, 64, 64)):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size

        # 加载标注文件
        annotations = np.load(self.data_dir / f"{split}_annotations.npy", allow_pickle=True)
        self.samples = annotations.item()
        self.patient_ids = list(self.samples.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # 加载 CT 体积
        ct_path = self.data_dir / "processed" / f"{patient_id}.nii.gz"
        ct_volume = nib.load(ct_path).get_fdata()

        # 获取结节位置
        nodule_info = self.samples[patient_id]
        center = nodule_info['center']  # (z, y, x)
        label = nodule_info['malignancy']  # 0: 良性, 1: 恶性

        # 提取 patch
        patch = self._extract_patch(ct_volume, center)

        # 转换为 tensor
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # (1, D, H, W)
        label = torch.tensor(label, dtype=torch.long)

        return {
            'image': patch,
            'label': label,
            'patient_id': patient_id,
            'center': center
        }

    def _extract_patch(self, volume, center):
        """提取以结节为中心的 3D patch"""
        d, h, w = self.patch_size
        z, y, x = center

        # 计算边界
        z_start = max(0, z - d // 2)
        y_start = max(0, y - h // 2)
        x_start = max(0, x - w // 2)

        z_end = min(volume.shape[0], z_start + d)
        y_end = min(volume.shape[1], y_start + h)
        x_end = min(volume.shape[2], x_start + w)

        # 提取 patch
        patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]

        # 填充到目标大小
        if patch.shape != self.patch_size:
            padded = np.zeros(self.patch_size, dtype=patch.dtype)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded

        return patch
```

## 5. 模型构建

### 5.1 使用 MedFusion 构建 3D CNN

```python
# examples/train_lung_nodule.py
import torch
import torch.nn as nn
from med_core.models import MultiModalModelBuilder
from med_core.trainers import BaseTrainer
from torch.utils.data import DataLoader

# 构建模型
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality(
    name="ct",
    backbone="resnet3d_18",  # 3D ResNet-18
    input_channels=1,
    pretrained=False
)
builder.set_head("classification")
model = builder.build()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

### 5.2 配置文件

创建 `configs/lung_nodule_config.yaml`：

```yaml
data:
  dataset_type: "custom"
  data_dir: "data/lung_nodule"
  batch_size: 8
  num_workers: 4
  pin_memory: true

  augmentation:
    enabled: true
    random_flip: true
    random_rotation: 15
    random_scale: [0.9, 1.1]
    random_noise: 0.01

model:
  num_classes: 2
  modalities:
    ct:
      backbone: "resnet3d_18"
      input_channels: 1
      pretrained: false

  head:
    type: "classification"
    dropout: 0.5

training:
  epochs: 50
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001

  scheduler:
    type: "cosine"
    T_max: 50
    eta_min: 0.00001

  loss:
    type: "cross_entropy"
    class_weights: [1.0, 2.0]  # 恶性样本权重更高

  mixed_precision: true
  gradient_clip: 1.0

  early_stopping:
    patience: 10
    metric: "val_auc"
    mode: "max"

logging:
  output_dir: "outputs/lung_nodule"
  tensorboard: true
  log_interval: 10
  save_top_k: 3
```

## 6. 训练模型

### 6.1 完整训练脚本

```python
# examples/train_lung_nodule.py (完整版)
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from med_core.models import build_model_from_config
from med_core.trainers import BaseTrainer
from med_core.configs import load_config
from lung_nodule_dataset import LungNoduleDataset

def main():
    # 加载配置
    config = load_config("configs/lung_nodule_config.yaml")

    # 创建数据集
    train_dataset = LungNoduleDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        patch_size=(64, 64, 64)
    )

    val_dataset = LungNoduleDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        patch_size=(64, 64, 64)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    # 构建模型
    model = build_model_from_config(config['model'])

    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # 开始训练
    trainer.train()

    print(f"训练完成！最佳模型保存在: {trainer.best_checkpoint_path}")

if __name__ == "__main__":
    main()
```

运行训练：

```bash
uv run python examples/train_lung_nodule.py
```

### 6.2 使用命令行工具

```bash
# 使用 MedFusion CLI
uv run medfusion-train --config configs/lung_nodule_config.yaml
```

## 7. 模型评估

### 7.1 评估脚本

```python
# examples/evaluate_lung_nodule.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs, _ = model({'ct': images})
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 恶性概率
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 计算指标
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=['良性', '恶性'])

    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(report)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['良性', '恶性'],
                yticklabels=['良性', '恶性'])
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title(f'混淆矩阵 (AUC={auc:.4f})')
    plt.savefig('outputs/lung_nodule/confusion_matrix.png', dpi=300, bbox_inches='tight')

    return auc, cm, report

# 加载模型并评估
checkpoint = torch.load('outputs/lung_nodule/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = LungNoduleDataset(data_dir='data/lung_nodule', split='test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

evaluate_model(model, test_loader)
```

## 8. 结果可视化

### 8.1 Grad-CAM 可视化

```python
# examples/visualize_lung_nodule.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from med_core.visualization import GradCAM3D

def visualize_prediction(model, sample, device='cuda'):
    """可视化模型预测和注意力"""
    model.eval()
    model.to(device)

    image = sample['image'].unsqueeze(0).to(device)  # (1, 1, D, H, W)
    label = sample['label'].item()

    # 预测
    with torch.no_grad():
        output, _ = model({'ct': image})
        prob = torch.softmax(output, dim=1)[0, 1].item()
        pred = torch.argmax(output, dim=1).item()

    # 生成 Grad-CAM
    gradcam = GradCAM3D(model, target_layer='backbones.ct.layer4')
    cam = gradcam(image, target_class=1)  # 恶性类别

    # 可视化中间切片
    slice_idx = image.shape[2] // 2
    img_slice = image[0, 0, slice_idx].cpu().numpy()
    cam_slice = cam[0, 0, slice_idx].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('原始 CT 切片')
    axes[0].axis('off')

    # Grad-CAM
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(cam_slice, cmap='jet', alpha=0.5)
    axes[1].set_title('Grad-CAM 热图')
    axes[1].axis('off')

    # 预测结果
    axes[2].text(0.5, 0.6, f"真实标签: {'恶性' if label == 1 else '良性'}",
                ha='center', fontsize=14)
    axes[2].text(0.5, 0.4, f"预测标签: {'恶性' if pred == 1 else '良性'}",
                ha='center', fontsize=14)
    axes[2].text(0.5, 0.2, f"恶性概率: {prob:.2%}",
                ha='center', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'outputs/lung_nodule/visualization_{sample["patient_id"]}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# 可视化几个测试样本
for i in range(5):
    sample = test_dataset[i]
    visualize_prediction(model, sample)
```

## 9. 预期结果

在 LIDC-IDRI 数据集上，使用 3D ResNet-18 模型，预期可达到：

- **AUC**: 0.85-0.90
- **准确率**: 80-85%
- **敏感度**: 85-90%（检出恶性结节）
- **特异度**: 75-80%（排除良性结节）

**训练时间**（单 GPU）：
- NVIDIA RTX 3090: ~2 小时
- NVIDIA V100: ~1.5 小时

## 10. 进阶优化

### 10.1 数据增强

```python
# 在配置文件中启用更多增强
augmentation:
  enabled: true
  random_flip: true
  random_rotation: 15
  random_scale: [0.9, 1.1]
  random_noise: 0.01
  elastic_deformation: true  # 弹性形变
  random_crop: true
```

### 10.2 模型集成

```python
# 训练多个模型并集成
models = [
    build_model_from_config({...}),  # ResNet3D-18
    build_model_from_config({...}),  # DenseNet3D-121
    build_model_from_config({...}),  # Swin3D-Tiny
]

# 集成预测
ensemble_probs = []
for model in models:
    with torch.no_grad():
        output, _ = model({'ct': image})
        prob = torch.softmax(output, dim=1)
        ensemble_probs.append(prob)

final_prob = torch.stack(ensemble_probs).mean(dim=0)
```

## 11. 常见问题

**Q: 内存不足怎么办？**
- 减小 batch_size
- 减小 patch_size（如 48×48×48）
- 启用梯度累积

**Q: 训练不收敛？**
- 检查数据预处理（窗位窗宽）
- 降低学习率
- 增加训练轮数
- 检查类别平衡

**Q: 如何处理不平衡数据？**
- 使用类别权重
- 过采样少数类
- 使用 Focal Loss

## 12. 总结

本案例展示了如何使用 MedFusion 框架进行肺结节检测：

1. ✅ 医学影像预处理（窗位窗宽、重采样）
2. ✅ 3D CNN 模型构建
3. ✅ 端到端训练流程
4. ✅ 模型评估和可视化

**下一步：** 尝试 [案例 2：乳腺癌分类](02_breast_cancer_classification.md)，学习多模态融合技术。
