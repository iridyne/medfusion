# 你的第一个模型

> 文档状态：**Beta**

**预计时间：30 分钟** ⭐

本教程将带你完成第一个多模态医学模型的端到端训练流程。我们将使用合成数据，因此无需准备真实医学数据即可快速上手。

## 学习目标

完成本教程后，你将学会：

- 准备多模态医学数据（图像 + 表格）
- 创建和理解配置文件
- 训练一个多模态融合模型
- 评估模型性能并解读结果

## 步骤 1：生成合成数据（5 分钟）

我们将创建一个简单的脚本来生成合成的医学图像和临床数据。

创建文件 `generate_data.py`：

```python
"""Generate synthetic medical data for training demo."""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def generate_synthetic_data(output_dir: Path, num_samples: int = 200):
    """Generate synthetic medical images and clinical data."""
    print(f"Generating {num_samples} synthetic samples...")

    # Create directories
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    data = []

    for i in range(num_samples):
        # Generate random medical image (noise + shapes)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Add a "tumor" (white circle) for positive class
        label = np.random.randint(0, 2)
        if label == 1:
            center = (np.random.randint(50, 174), np.random.randint(50, 174))
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:224, :224]
            mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
            img_array[mask] = 255  # White spot

        # Save image
        img = Image.fromarray(img_array)
        img_name = f"sample_{i:04d}.png"
        img.save(image_dir / img_name)

        # Generate clinical data (correlated with label)
        age = np.random.normal(60, 10) + (5 if label == 1 else 0)
        marker = np.random.normal(0.5, 0.2) + (0.3 if label == 1 else 0)

        record = {
            "patient_id": f"P{i:04d}",
            "image_path": img_name,
            "age": age,
            "marker": marker,
            "sex": np.random.choice(["M", "F"]),
            "smoking": np.random.choice(["Yes", "No"]),
            "diagnosis": label,
        }
        data.append(record)

    # Save CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    print(f"✓ Generated {num_samples} samples")
    print(f"✓ Images saved to: {image_dir}")
    print(f"✓ Metadata saved to: {csv_path}")

    return csv_path

if __name__ == "__main__":
    output_dir = Path("demo_data")
    generate_synthetic_data(output_dir, num_samples=200)
```

运行脚本生成数据：

```bash
uv run python generate_data.py
```

**预期输出：**

```
Generating 200 synthetic samples...
✓ Generated 200 samples
✓ Images saved to: demo_data/images
✓ Metadata saved to: demo_data/dataset.csv
```

## 步骤 2：创建配置文件（5 分钟）

创建配置文件 `demo_config.yaml`：

```yaml
# 实验基本信息
experiment_name: "first_model_demo"
seed: 42
device: "auto"  # 自动选择 GPU 或 CPU

# 数据配置
data:
  csv_path: "demo_data/dataset.csv"
  image_dir: "demo_data/images"
  image_path_column: "image_path"
  target_column: "diagnosis"

  # 特征列
  numerical_features:
    - "age"
    - "marker"
  categorical_features:
    - "sex"
    - "smoking"

  # 数据划分
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

  # 数据加载
  image_size: 224
  batch_size: 16
  num_workers: 0  # 设为 0 避免多进程问题

# 模型配置
model:
  num_classes: 2

  # 视觉骨干网络
  vision:
    backbone: "resnet18"
    pretrained: true
    feature_dim: 128
    attention_type: "cbam"  # 使用 CBAM 注意力机制
    dropout: 0.3

  # 表格骨干网络
  tabular:
    hidden_dims: [32, 16]
    output_dim: 16
    dropout: 0.2

  # 融合策略
  fusion:
    fusion_type: "gated"  # 门控融合
    hidden_dim: 32

# 训练配置
training:
  num_epochs: 5

  # 渐进式训练（分阶段解冻）
  use_progressive_training: true
  stage1_epochs: 1  # 只训练融合层和分类头
  stage2_epochs: 2  # 解冻表格网络
  stage3_epochs: 2  # 解冻所有层

  # 优化器
  optimizer:
    optimizer: "adamw"
    learning_rate: 0.001
    weight_decay: 0.01

  # 学习率调度器
  scheduler:
    scheduler: "cosine"
    T_max: 5

# 日志配置
logging:
  output_dir: "demo_output"
  use_tensorboard: true
  use_wandb: false
```

**配置说明：**

- **数据部分**：指定数据路径、特征列、数据划分比例
- **模型部分**：定义视觉网络（ResNet18）、表格网络（MLP）、融合策略（门控融合）
- **训练部分**：使用渐进式训练，分 3 个阶段逐步解冻网络层
- **日志部分**：保存训练日志和模型检查点

## 步骤 3：训练模型（15 分钟）

创建训练脚本 `train_first_model.py`：

```python
"""Train your first multimodal model."""
import logging
import sys
from pathlib import Path

import torch
import torch.optim as optim

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from med_core.backbones import create_vision_backbone, create_tabular_backbone
from med_core.configs import ExperimentConfig
from med_core.datasets import (
    MedicalMultimodalDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    split_dataset,
)
from med_core.fusion import MultiModalFusionModel, create_fusion_module
from med_core.trainers import MultimodalTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = ExperimentConfig.from_yaml("demo_config.yaml")
    logger.info(f"Loaded config: {config.experiment_name}")

    # Load dataset
    logger.info("Loading dataset...")
    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.csv_path,
        image_dir=config.data.image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
        handle_missing="fill_mean",
    )

    # Split dataset
    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )

    # Add transforms
    train_ds.transform = get_train_transforms(image_size=config.data.image_size)
    val_ds.transform = get_val_transforms(image_size=config.data.image_size)
    test_ds.transform = get_val_transforms(image_size=config.data.image_size)

    logger.info(f"Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Build model
    logger.info("Building model...")
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        pretrained=config.model.vision.pretrained,
        feature_dim=config.model.vision.feature_dim,
        attention_type=config.model.vision.attention_type,
    )

    tabular_backbone = create_tabular_backbone(
        input_dim=train_ds.get_tabular_dim(),
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
    )

    fusion_module = create_fusion_module(
        fusion_type=config.model.fusion.fusion_type,
        vision_dim=config.model.vision.feature_dim,
        tabular_dim=config.model.tabular.output_dim,
        output_dim=config.model.fusion.hidden_dim,
    )

    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs
    )

    trainer = MultimodalTrainer(
        config=config,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Training complete! Results saved to: {config.logging.output_dir}")

if __name__ == "__main__":
    main()
```

运行训练：

```bash
uv run python train_first_model.py
```

**预期输出：**

```
2026-03-14 10:00:00 - INFO - Loaded config: first_model_demo
2026-03-14 10:00:01 - INFO - Loading dataset...
2026-03-14 10:00:02 - INFO - Dataset split: train=140, val=30, test=30
2026-03-14 10:00:03 - INFO - Building model...
2026-03-14 10:00:04 - INFO - Model parameters: 11,234,567
2026-03-14 10:00:05 - INFO - Starting training...

[Stage 1/3] Training fusion layer only
Epoch 1/1: 100%|████████| 9/9 [00:15<00:00, 1.67s/it]
Train Loss: 0.6234, Val Loss: 0.5891, Val Acc: 0.6333

[Stage 2/3] Training fusion + tabular
Epoch 1/2: 100%|████████| 9/9 [00:16<00:00, 1.78s/it]
Train Loss: 0.5123, Val Loss: 0.4567, Val Acc: 0.7667
Epoch 2/2: 100%|████████| 9/9 [00:15<00:00, 1.72s/it]
Train Loss: 0.4234, Val Loss: 0.3891, Val Acc: 0.8333

[Stage 3/3] Training all layers
Epoch 1/2: 100%|████████| 9/9 [00:18<00:00, 2.01s/it]
Train Loss: 0.3456, Val Loss: 0.3123, Val Acc: 0.8667
Epoch 2/2: 100%|████████| 9/9 [00:17<00:00, 1.95s/it]
Train Loss: 0.2789, Val Loss: 0.2891, Val Acc: 0.9000

2026-03-14 10:02:30 - INFO - Training complete! Results saved to: demo_output
```

## 步骤 4：评估模型（5 分钟）

创建评估脚本 `evaluate_model.py`：

```python
"""Evaluate the trained model."""
import logging
from pathlib import Path

import torch
import numpy as np

from med_core.configs import ExperimentConfig
from med_core.datasets import MedicalMultimodalDataset, split_dataset, get_val_transforms
from med_core.evaluation import calculate_binary_metrics, generate_evaluation_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load config and model
    config = ExperimentConfig.from_yaml("demo_config.yaml")
    checkpoint_path = Path(config.logging.output_dir) / "checkpoints" / "best_model.pth"

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    model.eval()

    # Load test dataset
    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.csv_path,
        image_dir=config.data.image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
    )

    _, _, test_ds = split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    test_ds.transform = get_val_transforms(image_size=config.data.image_size)

    # Evaluate
    logger.info("Evaluating on test set...")
    all_preds = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for i in range(len(test_ds)):
            image, tabular, label = test_ds[i]
            image = image.unsqueeze(0).to(device)
            tabular = tabular.unsqueeze(0).to(device)

            outputs = model(image, tabular)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            pred = torch.argmax(logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)

    # Calculate metrics
    metrics = calculate_binary_metrics(all_labels, all_preds)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Test Set Results:")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC:   {metrics.get('auc_roc', 'N/A')}")
    logger.info("="*50)

    # Generate report
    report_path = generate_evaluation_report(
        metrics=metrics,
        output_dir=config.logging.output_dir,
        experiment_name=config.experiment_name,
        config=config.to_dict(),
    )

    logger.info(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
```

运行评估：

```bash
uv run python evaluate_model.py
```

**预期输出：**

```
2026-03-14 10:05:00 - INFO - Loading checkpoint: demo_output/checkpoints/best_model.pth
2026-03-14 10:05:01 - INFO - Evaluating on test set...

==================================================
Test Set Results:
==================================================
Accuracy:  0.9000
Precision: 0.8824
Recall:    0.9375
F1 Score:  0.9091
AUC-ROC:   0.9467
==================================================

2026-03-14 10:05:15 - INFO - Detailed report saved to: demo_output/evaluation_report.html
```

## 结果解读

### 训练过程分析

1. **渐进式训练**：模型分 3 个阶段训练
   - Stage 1：只训练融合层和分类头（快速收敛）
   - Stage 2：解冻表格网络（学习临床特征）
   - Stage 3：解冻所有层（端到端微调）

2. **性能提升**：验证准确率从 63% 提升到 90%
   - 说明模型成功学习了图像和临床数据的联合表示

### 评估指标说明

- **Accuracy（准确率）**：90% - 整体预测正确的比例
- **Precision（精确率）**：88% - 预测为阳性中真正阳性的比例
- **Recall（召回率）**：94% - 真实阳性中被正确识别的比例
- **F1 Score**：91% - 精确率和召回率的调和平均
- **AUC-ROC**：95% - 模型区分阳性和阴性的能力

### 输出文件

训练完成后，`demo_output/` 目录包含：

```
demo_output/
├── checkpoints/
│   ├── best_model.pth      # 最佳模型（验证集上）
│   └── last_model.pth      # 最后一个 epoch 的模型
├── logs/
│   └── tensorboard/        # TensorBoard 日志
├── evaluation_report.html  # 详细评估报告
└── training_log.txt        # 训练日志
```

## 下一步

恭喜！你已经完成了第一个多模态医学模型的训练。接下来可以：

1. **尝试不同的模型架构**
   - 更换骨干网络：`resnet50`, `efficientnet_b0`, `swin_tiny`
   - 尝试其他融合策略：`attention`, `bilinear`, `cross_attention`

2. **优化训练策略**
   - 调整学习率、batch size
   - 使用混合精度训练：`training.mixed_precision: true`
   - 增加数据增强

3. **使用真实数据**
   - 参考 [数据准备指南](../tutorials/fundamentals/data-prep.md)
   - 学习 [多视图支持概览](../guides/multiview/overview.md)

4. **部署模型**
   - 导出为 ONNX 格式
   - 使用 Web UI 进行推理

## 常见问题

### Q: 训练时显存不足怎么办？

减小 batch size：

```yaml
data:
  batch_size: 8  # 从 16 改为 8
```

### Q: 如何使用 GPU 训练？

配置文件中设置：

```yaml
device: "cuda"  # 或 "cuda:0" 指定 GPU
```

### Q: 如何可视化训练过程？

启动 TensorBoard：

```bash
uv run tensorboard --logdir demo_output/logs/tensorboard
```

然后访问 http://localhost:6006

## 相关资源

- [配置文件详解](../tutorials/fundamentals/configs.md)
- [Core Runtime Architecture](../architecture/CORE_RUNTIME_ARCHITECTURE.md)
- [训练工作流](../tutorials/training/workflow.md)
- [API 文档总览](../api/med_core.md)
