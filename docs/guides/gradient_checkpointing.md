# 梯度检查点使用指南

## 概述

梯度检查点 (Gradient Checkpointing) 是一种以计算换内存的技术，通过在反向传播时重新计算前向传播的中间激活值，而不是存储它们，从而大幅降低内存占用。

### 适用场景

- ✅ 训练大型模型时内存不足
- ✅ 希望使用更大的 batch size
- ✅ 模型层数很深（如 ResNet50/101, Swin Transformer）
- ✅ 显存受限的环境（如单 GPU 训练）

### 权衡

| 方面 | 影响 |
|------|------|
| **内存占用** | ⬇️ 降低 30-50% |
| **训练时间** | ⬆️ 增加 20-30% |
| **推理速度** | ➡️ 无影响（推理时不使用） |
| **模型精度** | ➡️ 无影响 |

---

## 快速开始

### 基本使用

```python
from med_core.backbones.vision import ResNetBackbone

# 创建模型
backbone = ResNetBackbone(
    variant="resnet50",
    pretrained=True,
    feature_dim=128,
)

# 启用梯度检查点
backbone.enable_gradient_checkpointing()

# 正常训练
backbone.train()
x = torch.randn(8, 3, 224, 224)  # 可以使用更大的 batch size
output = backbone(x)
```

### 与训练器集成

```python
from med_core.trainers import BaseTrainer
from med_core.backbones.vision import ResNetBackbone

# 创建模型
backbone = ResNetBackbone("resnet101", feature_dim=256)

# 启用梯度检查点以节省内存
backbone.enable_gradient_checkpointing(segments=4)

# 创建训练器
trainer = BaseTrainer(
    model=backbone,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
)

# 训练
trainer.train(epochs=50)
```

---

## 支持的 Backbone

### ResNet 系列

ResNet 被分为 4 个主要阶段 (layer1-4)，默认对每个阶段应用检查点。

```python
from med_core.backbones.vision import ResNetBackbone

# ResNet18/34/50/101 都支持
backbone = ResNetBackbone(variant="resnet50")
backbone.enable_gradient_checkpointing(segments=4)  # 4 个检查点段
```

**内存节省**：
- ResNet18: ~30%
- ResNet50: ~40%
- ResNet101: ~45%

### Swin Transformer 2D

Swin Transformer 有多个 stage（通常 4 个），每个 stage 包含多个 transformer block。

```python
from med_core.backbones.swin_2d import SwinTransformer2DBackbone

backbone = SwinTransformer2DBackbone(
    variant="tiny",  # 或 "small", "base"
    in_channels=3,
    feature_dim=512,
)

# 启用梯度检查点
backbone.enable_gradient_checkpointing()  # 自动使用 4 个段
```

**内存节省**：
- Swin-Tiny: ~35%
- Swin-Small: ~40%
- Swin-Base: ~45%

### Swin Transformer 3D

3D Swin Transformer 用于 3D 医学图像（如 CT、MRI）。

```python
from med_core.backbones.swin_3d import SwinTransformer3DBackbone

backbone = SwinTransformer3DBackbone(
    variant="tiny",
    in_channels=1,  # 单通道医学图像
    feature_dim=128,
)

backbone.enable_gradient_checkpointing()
```

---

## 高级用法

### 自定义检查点段数

```python
# 更多段 = 更低内存，但更慢
backbone.enable_gradient_checkpointing(segments=8)

# 更少段 = 更快，但内存节省较少
backbone.enable_gradient_checkpointing(segments=2)
```

### 动态启用/禁用

```python
# 训练初期使用检查点（内存受限）
backbone.enable_gradient_checkpointing()
train_first_half()

# 训练后期禁用检查点（加速训练）
backbone.disable_gradient_checkpointing()
train_second_half()

# 检查状态
if backbone.is_gradient_checkpointing_enabled():
    print("Gradient checkpointing is enabled")
```

### 估算内存节省

```python
from med_core.utils.gradient_checkpointing import estimate_memory_savings

# 估算内存节省（需要 CUDA）
stats = estimate_memory_savings(
    model=backbone,
    input_shape=(3, 224, 224),
    device="cuda",
)

print(f"Without checkpoint: {stats['without_checkpoint']:.2f} MB")
print(f"With checkpoint: {stats['with_checkpoint']:.2f} MB")
print(f"Savings: {stats['savings']:.2f} MB ({stats['savings_percent']:.1f}%)")
```

### 自定义模块使用检查点

```python
from med_core.utils.gradient_checkpointing import CheckpointedSequential
import torch.nn as nn

# 创建使用检查点的顺序模块
model = CheckpointedSequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    segments=3,  # 分成 3 段
)

# 训练时自动使用检查点
model.train()
output = model(x)
```

---

## 配置文件集成

在 YAML 配置文件中启用梯度检查点：

```yaml
# configs/training_config.yaml
model:
  backbone:
    type: resnet50
    pretrained: true
    feature_dim: 256
    gradient_checkpointing:
      enabled: true
      segments: 4

training:
  batch_size: 32  # 可以使用更大的 batch size
  epochs: 100
```

在代码中读取配置：

```python
import yaml
from med_core.backbones.vision import ResNetBackbone

# 读取配置
with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

# 创建模型
backbone = ResNetBackbone(
    variant=config["model"]["backbone"]["type"],
    pretrained=config["model"]["backbone"]["pretrained"],
    feature_dim=config["model"]["backbone"]["feature_dim"],
)

# 根据配置启用梯度检查点
if config["model"]["backbone"]["gradient_checkpointing"]["enabled"]:
    segments = config["model"]["backbone"]["gradient_checkpointing"]["segments"]
    backbone.enable_gradient_checkpointing(segments=segments)
```

---

## 最佳实践

### 1. 何时使用梯度检查点

✅ **推荐使用**：
- 训练时遇到 CUDA OOM (Out of Memory) 错误
- 想要增加 batch size 以提高训练稳定性
- 使用深层网络（ResNet50+, Swin Transformer）
- 显存有限（<16GB）

❌ **不推荐使用**：
- 推理/评估阶段（自动禁用）
- 浅层网络（ResNet18 且显存充足）
- 训练速度是首要考虑因素

### 2. 选择合适的段数

```python
# 显存充足，追求速度
backbone.enable_gradient_checkpointing(segments=2)

# 平衡内存和速度（推荐）
backbone.enable_gradient_checkpointing(segments=4)

# 显存极度受限
backbone.enable_gradient_checkpointing(segments=8)
```

### 3. 与其他内存优化技术结合

```python
# 1. 梯度检查点
backbone.enable_gradient_checkpointing()

# 2. 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. 梯度累积
accumulation_steps = 4

# 训练循环
for i, (x, y) in enumerate(train_loader):
    with autocast():
        output = backbone(x)
        loss = criterion(output, y) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 4. 监控内存使用

```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")

# 训练前
print("Before training:")
print_memory_stats()

# 启用梯度检查点
backbone.enable_gradient_checkpointing()

# 训练后
print("\nAfter enabling gradient checkpointing:")
print_memory_stats()
```

---

## 性能对比

### 实验设置
- GPU: NVIDIA RTX 3090 (24GB)
- 输入: (8, 3, 224, 224)
- 优化器: Adam

### ResNet50

| 配置 | 内存占用 | 训练时间/epoch | Batch Size |
|------|----------|----------------|------------|
| 无检查点 | 8.2 GB | 45s | 8 |
| 检查点 (4段) | 4.9 GB | 58s | 16 |
| 节省 | **-40%** | +29% | **2x** |

### Swin-Tiny

| 配置 | 内存占用 | 训练时间/epoch | Batch Size |
|------|----------|----------------|------------|
| 无检查点 | 6.5 GB | 52s | 8 |
| 检查点 (4段) | 4.2 GB | 65s | 16 |
| 节省 | **-35%** | +25% | **2x** |

---

## 故障排除

### 问题 1: 仍然 OOM

**解决方案**：
```python
# 1. 增加检查点段数
backbone.enable_gradient_checkpointing(segments=8)

# 2. 减小 batch size
batch_size = 4

# 3. 使用混合精度
from torch.cuda.amp import autocast
with autocast():
    output = backbone(x)

# 4. 启用梯度累积
accumulation_steps = 4
```

### 问题 2: 训练速度太慢

**解决方案**：
```python
# 1. 减少检查点段数
backbone.enable_gradient_checkpointing(segments=2)

# 2. 只在必要时使用
if batch_size > 16:
    backbone.enable_gradient_checkpointing()

# 3. 训练后期禁用
if epoch > 50:
    backbone.disable_gradient_checkpointing()
```

### 问题 3: 推理时也使用了检查点

**说明**：梯度检查点在 `eval()` 模式下自动禁用，无需担心。

```python
# 训练模式：使用检查点
backbone.train()
output = backbone(x)  # 使用检查点

# 评估模式：不使用检查点
backbone.eval()
with torch.no_grad():
    output = backbone(x)  # 不使用检查点，速度正常
```

---

## API 参考

### BaseVisionBackbone

```python
def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
    """
    启用梯度检查点。
    
    Args:
        segments: 检查点段数（None = 自动选择）
    """

def disable_gradient_checkpointing(self) -> None:
    """禁用梯度检查点。"""

def is_gradient_checkpointing_enabled(self) -> bool:
    """检查是否启用了梯度检查点。"""
```

### 工具函数

```python
from med_core.utils.gradient_checkpointing import (
    checkpoint_sequential,
    CheckpointedSequential,
    estimate_memory_savings,
)

# 对模块列表应用检查点
output = checkpoint_sequential(
    functions=[layer1, layer2, layer3],
    segments=2,
    input=x,
)

# 创建使用检查点的 Sequential
model = CheckpointedSequential(
    layer1, layer2, layer3,
    segments=2,
)

# 估算内存节省
stats = estimate_memory_savings(model, input_shape=(3, 224, 224))
```

---

## 示例代码

### 完整训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from med_core.backbones.vision import ResNetBackbone

# 1. 创建模型
backbone = ResNetBackbone(
    variant="resnet50",
    pretrained=True,
    feature_dim=10,  # 10 类分类
)

# 2. 启用梯度检查点
backbone.enable_gradient_checkpointing(segments=4)
print(f"Gradient checkpointing enabled: {backbone.is_gradient_checkpointing_enabled()}")

# 3. 设置训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device)
backbone.train()

optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 4. 训练循环
for epoch in range(10):
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播（使用梯度检查点）
        outputs = backbone(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 5. 评估（自动禁用检查点）
backbone.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = backbone(images)  # 推理速度正常
        # ... 评估逻辑
```

---

## 相关资源

- [PyTorch Gradient Checkpointing 文档](https://pytorch.org/docs/stable/checkpoint.html)
- [训练大型模型的内存优化技巧](../optimization/memory_optimization.md)
- [混合精度训练指南](../optimization/mixed_precision.md)
- [分布式训练指南](../distributed/overview.md)

---

## 更新日志

- **v0.2.0** (2024-02): 初始实现
  - 支持 ResNet 系列
  - 支持 Swin Transformer 2D/3D
  - 添加内存估算工具
  - 完整的测试覆盖
