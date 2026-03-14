# Trainers API

训练器模块，提供完整的模型训练流程。

## 概述

Trainers 模块提供了灵活的训练系统，支持：

- **基础训练**: 标准训练循环、验证、早停
- **多模态训练**: 专门处理视觉 + 表格数据
- **多视图训练**: 支持多视图多模态场景
- **混合精度**: 自动混合精度训练（AMP）
- **渐进式训练**: 分阶段冻结/解冻网络
- **注意力监督**: 可选的注意力引导训练

## 核心类

### BaseTrainer

所有训练器的抽象基类。

**参数：**
- `config` (ExperimentConfig): 实验配置对象
- `model` (nn.Module): 待训练模型
- `train_loader` (DataLoader): 训练数据加载器
- `val_loader` (DataLoader): 验证数据加载器
- `optimizer` (Optimizer): 优化器，默认 None（自动创建 AdamW）
- `scheduler` (Any): 学习率调度器，默认 None
- `device` (str): 设备，默认自动检测

**核心方法：**
- `train()` - 主训练循环
- `training_step(batch, batch_idx)` - 单步训练（需子类实现）
- `validation_step(batch, batch_idx)` - 单步验证（需子类实现）
- `on_train_start()` - 训练开始钩子
- `on_epoch_start()` - 每轮开始钩子
- `on_epoch_end(train_metrics, val_metrics)` - 每轮结束钩子
- `load_checkpoint(path)` - 加载检查点
- `_save_checkpoint(filename, metrics)` - 保存检查点

**特性：**
- 自动早停（基于验证指标）
- 检查点管理（保存最佳和最新模型）
- TensorBoard 日志记录
- 梯度裁剪
- 学习率调度

**示例：**
```python
from med_core.trainers import BaseTrainer
from med_core.configs import ExperimentConfig

# 加载配置
config = ExperimentConfig.from_yaml("configs/experiment.yaml")

# 创建训练器（需要继承并实现 training_step 和 validation_step）
class MyTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        images, tabular, labels = batch
        outputs = self.model(images, tabular)
        loss = self.criterion(outputs, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, tabular, labels = batch
        outputs = self.model(images, tabular)
        loss = self.criterion(outputs, labels)
        return {"loss": loss}

trainer = MyTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# 开始训练
history = trainer.train()
```

### MultimodalTrainer

多模态模型专用训练器。

**参数：**
- 继承 `BaseTrainer` 的所有参数

**额外特性：**
- 自动处理 (image, tabular, label) 元组输入
- 混合精度训练（AMP）
- 渐进式训练（分阶段冻结/解冻）
- 注意力监督损失
- 类别权重和标签平滑

**配置项：**
```yaml
training:
  # 基础设置
  num_epochs: 100
  batch_size: 32

  # 优化器
  optimizer:
    type: adamw
    learning_rate: 0.001
    weight_decay: 0.01

  # 学习率调度
  scheduler:
    type: cosine
    warmup_epochs: 5

  # 混合精度
  mixed_precision: true

  # 早停
  patience: 10
  monitor: val_auc
  mode: max
  min_delta: 0.001

  # 梯度裁剪
  gradient_clip: 1.0

  # 类别平衡
  class_weights: [1.0, 2.0]  # 可选
  label_smoothing: 0.1

  # 渐进式训练
  use_progressive_training: true
  stage1_epochs: 20  # 阶段1: 冻结表格，训练视觉
  stage2_epochs: 20  # 阶段2: 冻结视觉，训练表格
  # 阶段3: 全部解冻

  # 注意力监督
  use_attention_supervision: false
  attention_loss_weight: 0.1
  attention_supervision_method: mask_guided
```

**示例：**
```python
from med_core.trainers import MultimodalTrainer
from med_core.configs import ExperimentConfig

# 加载配置
config = ExperimentConfig.from_yaml("configs/multimodal.yaml")

# 创建训练器
trainer = MultimodalTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# 训练
history = trainer.train()

# 查看训练历史
print(f"最佳验证 AUC: {max(history['val_auc']):.4f}")
print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
```

### MultiViewMultimodalTrainer

多视图多模态训练器。

**特性：**
- 支持多视图输入（每个样本多张图像）
- 自动处理视图聚合
- 支持视图级别的注意力监督
- 继承 `MultimodalTrainer` 的所有特性

**示例：**
```python
from med_core.trainers import MultiViewMultimodalTrainer

trainer = MultiViewMultimodalTrainer(
    config=config,
    model=multiview_model,
    train_loader=multiview_train_loader,
    val_loader=multiview_val_loader
)

history = trainer.train()
```

## 工厂函数

### create_trainer

创建训练器的工厂函数。

**参数：**
- `config` (ExperimentConfig): 实验配置
- `model` (nn.Module): 模型
- `train_loader` (DataLoader): 训练数据加载器
- `val_loader` (DataLoader): 验证数据加载器
- `trainer_type` (str): 训练器类型，默认 "multimodal"

**支持的训练器类型：**
- `"multimodal"` - 多模态训练器
- `"multiview"` - 多视图多模态训练器

**示例：**
```python
from med_core.trainers import create_trainer

trainer = create_trainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    trainer_type="multimodal"
)

history = trainer.train()
```

## 训练流程

### 标准训练循环

```python
# 1. 准备数据
train_loader, val_loader, test_loader = create_dataloaders(dataset)

# 2. 创建模型
model = build_model_from_config(config)

# 3. 创建训练器
trainer = MultimodalTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# 4. 训练
history = trainer.train()

# 5. 加载最佳模型
trainer.load_checkpoint(config.checkpoint_dir / "best.pth")

# 6. 测试
test_metrics = evaluate_model(model, test_loader)
```

### 渐进式训练

渐进式训练分为三个阶段：

**阶段 1 (0-20 epochs)**: 冻结表格分支，训练视觉分支
- 适应视觉特征提取
- 表格分支保持预训练权重

**阶段 2 (20-40 epochs)**: 冻结视觉分支，训练表格分支
- 适应表格特征提取
- 视觉分支保持阶段1权重

**阶段 3 (40+ epochs)**: 全部解冻，端到端微调
- 联合优化所有组件
- 学习最优的模态融合

```yaml
training:
  use_progressive_training: true
  stage1_epochs: 20
  stage2_epochs: 20
  num_epochs: 60  # 总共60轮
```

### 混合精度训练

自动混合精度（AMP）可以加速训练并减少显存占用：

```yaml
training:
  mixed_precision: true  # 启用 AMP
```

**优势：**
- 训练速度提升 2-3 倍
- 显存占用减少 30-50%
- 精度损失可忽略

### 注意力监督

可选的注意力引导训练，提高模型可解释性：

```yaml
training:
  use_attention_supervision: true
  attention_loss_weight: 0.1
  attention_supervision_method: mask_guided

model:
  vision:
    enable_attention_supervision: true
    attention_type: cbam
```

**监督方法：**
- `mask_guided` - 使用分割掩码引导注意力
- `cam_based` - 基于 CAM 的监督
- `consistency` - 注意力一致性约束

## 检查点管理

### 自动保存

训练器自动保存两种检查点：

- `best.pth` - 验证指标最佳的模型
- `last.pth` - 最新的模型（每轮更新）

### 加载检查点

```python
# 加载最佳模型
trainer.load_checkpoint("outputs/checkpoints/best.pth")

# 或使用工具函数
from med_core.utils import load_checkpoint

checkpoint = load_checkpoint(
    filepath="outputs/checkpoints/best.pth",
    model=model,
    optimizer=optimizer,
    device="cuda"
)

print(f"加载的轮次: {checkpoint['epoch']}")
print(f"验证指标: {checkpoint['metrics']}")
```

## 日志记录

### TensorBoard

训练器自动记录到 TensorBoard：

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/logs

# 访问 http://localhost:6006
```

**记录内容：**
- 训练/验证损失
- 各种评估指标
- 学习率变化
- 梯度统计
- 注意力权重可视化

### Weights & Biases

可选的 W&B 集成：

```yaml
logging:
  use_wandb: true
  wandb_project: medfusion
  wandb_entity: your-team
```

```python
import wandb

wandb.init(
    project="medfusion",
    config=config.to_dict()
)

# 训练器会自动记录到 W&B
trainer.train()
```

## 最佳实践

**学习率设置：**
- 视觉骨干: 1e-4 到 1e-3
- 表格骨干: 1e-3 到 1e-2
- 融合模块: 1e-3 到 1e-2

**批次大小：**
- 单视图: 32-64
- 多视图: 8-16（取决于视图数量）

**早停：**
- patience: 10-20 轮
- min_delta: 0.001

**梯度裁剪：**
- 推荐值: 1.0
- 防止梯度爆炸

**数据增强：**
- 训练集: medium 强度
- 验证/测试集: 仅归一化

## 故障排除

**显存不足：**
- 减小 batch_size
- 启用混合精度训练
- 使用梯度累积
- 减少模型大小

**训练不收敛：**
- 降低学习率
- 增加 warmup epochs
- 检查数据归一化
- 使用渐进式训练

**过拟合：**
- 增加 dropout
- 使用数据增强
- 添加权重衰减
- 减少模型复杂度

**验证指标波动：**
- 增加验证集大小
- 使用更稳定的指标（AUC vs Accuracy）
- 增加 patience

## 参考

完整实现请参考：
- `/home/yixian/Projects/med-ml/medfusion/med_core/trainers/base.py` - 基础训练器
- `/home/yixian/Projects/med-ml/medfusion/med_core/trainers/multimodal.py` - 多模态训练器
- `/home/yixian/Projects/med-ml/medfusion/med_core/trainers/multiview_trainer.py` - 多视图训练器
