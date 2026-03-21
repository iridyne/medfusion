# 超参数调优

**预计时间：25分钟**

本教程介绍如何系统地调优模型超参数以获得最佳性能。

## 关键超参数

### 1. 学习率（Learning Rate）

最重要的超参数，影响训练速度和最终性能。

**推荐范围：**
- Adam: 1e-4 到 1e-3
- SGD: 1e-2 到 1e-1
- AdamW: 1e-5 到 1e-3

**调优策略：**

```python
# 方法 1：学习率范围测试
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
for lr in learning_rates:
    config.training.optimizer.lr = lr
    # 训练几个 epoch 观察损失
```

```yaml
# 方法 2：使用学习率调度器
training:
  optimizer:
    name: "adam"
    lr: 0.001
  scheduler:
    name: "cosine"
    warmup_epochs: 5
    min_lr: 1e-6
```

### 2. 批次大小（Batch Size）

影响训练稳定性和内存使用。

**推荐范围：**
- 小数据集（<5k）: 16-32
- 中等数据集（5k-50k）: 32-64
- 大数据集（>50k）: 64-128

**权衡：**
- 更大批次：更稳定，但可能泛化性差
- 更小批次：更好泛化，但训练不稳定

```yaml
data:
  batch_size: 32

# 如果内存不足，使用梯度累积
training:
  gradient_accumulation_steps: 2  # 等效批次大小 = 32 * 2 = 64
```

### 3. 优化器选择

**Adam（推荐）：**
```yaml
training:
  optimizer:
    name: "adam"
    lr: 0.001
    betas: [0.9, 0.999]
    weight_decay: 0.0001
```

**AdamW（更好的正则化）：**
```yaml
training:
  optimizer:
    name: "adamw"
    lr: 0.0001
    weight_decay: 0.01  # 更强的正则化
```

**SGD（需要更多调优）：**
```yaml
training:
  optimizer:
    name: "sgd"
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0001
```

### 4. 数据增强

**基础增强：**
```yaml
data:
  augmentation:
    random_flip: true
    random_rotation: 15
    random_scale: [0.9, 1.1]
    color_jitter: 0.2
```

**强增强（大数据集）：**
```yaml
data:
  augmentation:
    random_flip: true
    random_rotation: 30
    random_scale: [0.8, 1.2]
    color_jitter: 0.4
    random_erasing: 0.2
    mixup_alpha: 0.2
```

**弱增强（小数据集）：**
```yaml
data:
  augmentation:
    random_flip: true
    random_rotation: 10
    normalize: true
```

### 5. 正则化

**Dropout：**
```yaml
model:
  fusion:
    dropout: 0.3  # 0.1-0.5
  head:
    dropout: 0.5  # 分类头通常更高
```

**Weight Decay：**
```yaml
training:
  optimizer:
    weight_decay: 0.0001  # 1e-5 到 1e-3
```

**Label Smoothing：**
```yaml
training:
  label_smoothing: 0.1  # 0.0-0.2
```

## 系统调优流程

### 阶段 1：粗调（Coarse Tuning）

目标：快速找到合理范围

```python
# 1. 固定其他参数，只调学习率
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
epochs = 10  # 快速测试

# 2. 选择最佳学习率后，调批次大小
batch_sizes = [16, 32, 64]

# 3. 选择优化器
optimizers = ["adam", "adamw", "sgd"]
```

### 阶段 2：细调（Fine Tuning）

目标：在最佳范围内精确搜索

```python
# 在最佳学习率附近搜索
best_lr = 1e-3
learning_rates = [best_lr * 0.5, best_lr, best_lr * 2]

# 调整正则化
weight_decays = [1e-5, 1e-4, 1e-3]
dropouts = [0.1, 0.3, 0.5]
```

### 阶段 3：微调（Micro Tuning）

目标：榨取最后的性能

```python
# 学习率调度器
schedulers = ["cosine", "step", "exponential"]

# 数据增强强度
augmentation_strengths = ["weak", "medium", "strong"]
```

## 使用 Optuna 自动调优

### 安装 Optuna

```bash
uv pip install optuna
```

### 基本示例

```python
import optuna
from med_core.models import MultiModalModelBuilder
from med_core.trainers import MultimodalTrainer

def objective(trial):
    # 定义搜索空间
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # 创建配置
    config = load_config("configs/starter/quickstart.yaml")
    config.training.optimizer.lr = lr
    config.data.batch_size = batch_size
    config.model.fusion.dropout = dropout
    config.training.optimizer.weight_decay = weight_decay

    # 训练模型
    model = build_model_from_config(config)
    trainer = MultimodalTrainer(model, train_loader, val_loader, config)
    history = trainer.train()

    # 返回验证指标
    return history['val_auc'][-1]

# 创建研究
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 查看最佳参数
print("最佳参数:", study.best_params)
print("最佳 AUC:", study.best_value)
```

### 高级搜索空间

```python
def objective(trial):
    # 条件搜索
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

    if optimizer_name == "sgd":
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
    else:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        momentum = None

    # 骨干网络选择
    backbone = trial.suggest_categorical("backbone",
        ["resnet18", "resnet50", "efficientnet_b0"])

    # 融合策略
    fusion_type = trial.suggest_categorical("fusion_type",
        ["concat", "gated", "attention"])

    # 数据增强
    aug_strength = trial.suggest_float("aug_strength", 0.0, 1.0)

    # ... 构建和训练模型 ...

    return val_auc
```

### 可视化结果

```python
# 优化历史
optuna.visualization.plot_optimization_history(study)

# 参数重要性
optuna.visualization.plot_param_importances(study)

# 并行坐标图
optuna.visualization.plot_parallel_coordinate(study)

# 切片图
optuna.visualization.plot_slice(study)
```

## 使用 Ray Tune

### 安装 Ray Tune

```bash
uv pip install ray[tune]
```

### 基本示例

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    # 使用 config 训练模型
    model = build_model_from_config(config)
    trainer = MultimodalTrainer(model, train_loader, val_loader, config)

    for epoch in range(config["epochs"]):
        metrics = trainer.train_epoch()
        # 报告指标给 Ray Tune
        tune.report(
            loss=metrics["val_loss"],
            accuracy=metrics["val_acc"],
            auc=metrics["val_auc"]
        )

# 定义搜索空间
config = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64]),
    "dropout": tune.uniform(0.1, 0.5),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
}

# 使用 ASHA 调度器（早停表现差的试验）
scheduler = ASHAScheduler(
    metric="auc",
    mode="max",
    max_t=50,
    grace_period=10,
    reduction_factor=2
)

# 运行调优
analysis = tune.run(
    train_model,
    config=config,
    num_samples=50,
    scheduler=scheduler,
    resources_per_trial={"cpu": 4, "gpu": 1}
)

# 获取最佳配置
best_config = analysis.get_best_config(metric="auc", mode="max")
print("最佳配置:", best_config)
```

## 网格搜索 vs 随机搜索

### 网格搜索

适合：参数空间小，需要完整探索

```python
from itertools import product

# 定义参数网格
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'dropout': [0.1, 0.3, 0.5]
}

# 生成所有组合
results = []
for lr, bs, dropout in product(*param_grid.values()):
    config.training.optimizer.lr = lr
    config.data.batch_size = bs
    config.model.fusion.dropout = dropout

    # 训练并记录结果
    auc = train_and_evaluate(config)
    results.append({
        'lr': lr,
        'batch_size': bs,
        'dropout': dropout,
        'auc': auc
    })

# 找到最佳组合
best = max(results, key=lambda x: x['auc'])
```

### 随机搜索

适合：参数空间大，资源有限

```python
import numpy as np

# 随机采样
n_trials = 50
results = []

for _ in range(n_trials):
    # 随机采样参数
    lr = 10 ** np.random.uniform(-5, -2)
    batch_size = np.random.choice([16, 32, 64])
    dropout = np.random.uniform(0.1, 0.5)
    weight_decay = 10 ** np.random.uniform(-5, -3)

    # 训练并记录
    config.training.optimizer.lr = lr
    config.data.batch_size = batch_size
    config.model.fusion.dropout = dropout
    config.training.optimizer.weight_decay = weight_decay

    auc = train_and_evaluate(config)
    results.append({
        'lr': lr,
        'batch_size': batch_size,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'auc': auc
    })
```

## 学习率查找器

### 实现学习率范围测试

```python
import torch
import matplotlib.pyplot as plt

def find_lr(model, train_loader, optimizer, criterion,
            start_lr=1e-7, end_lr=10, num_iter=100):
    """
    学习率范围测试
    """
    model.train()
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr

    lrs = []
    losses = []
    best_loss = float('inf')

    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_iter:
            break

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 记录
        lrs.append(lr)
        losses.append(loss.item())

        # 如果损失爆炸，停止
        if loss.item() > 4 * best_loss:
            break
        if loss.item() < best_loss:
            best_loss = loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 增加学习率
        lr *= lr_mult
        optimizer.param_groups[0]['lr'] = lr

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.savefig('lr_finder.png')

    # 推荐学习率（损失下降最快的点）
    gradients = np.gradient(losses)
    recommended_lr = lrs[np.argmin(gradients)]
    print(f"推荐学习率: {recommended_lr:.2e}")

    return lrs, losses, recommended_lr

# 使用
lrs, losses, best_lr = find_lr(model, train_loader, optimizer, criterion)
```

## 实用技巧

### 1. 从已知良好配置开始

```yaml
# 医学影像分类的良好起点
training:
  optimizer:
    name: "adam"
    lr: 0.0001
    weight_decay: 0.0001
  scheduler:
    name: "cosine"
    warmup_epochs: 5

data:
  batch_size: 32
  augmentation:
    random_flip: true
    random_rotation: 15
    normalize: true

model:
  vision:
    backbone: "resnet50"
    pretrained: true
  fusion:
    type: "gated"
    dropout: 0.3
```

### 2. 使用早停节省时间

```yaml
training:
  early_stopping:
    enabled: true
    patience: 5  # 调优时使用更小的 patience
    min_delta: 0.001
```

### 3. 记录所有实验

```python
import json

# 保存实验结果
experiment_log = {
    'config': config.to_dict(),
    'results': {
        'val_auc': val_auc,
        'val_acc': val_acc,
        'train_time': train_time
    }
}

with open(f'experiments/exp_{timestamp}.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

### 4. 并行运行多个实验

```bash
# 使用不同 GPU 并行运行
CUDA_VISIBLE_DEVICES=0 uv run medfusion train --config config1.yaml &
CUDA_VISIBLE_DEVICES=1 uv run medfusion train --config config2.yaml &
CUDA_VISIBLE_DEVICES=2 uv run medfusion train --config config3.yaml &
wait
```

## 常见问题

### Q1: 调优顺序是什么？

**A:** 推荐顺序：
1. 学习率（最重要）
2. 批次大小
3. 优化器
4. 正则化（dropout, weight decay）
5. 数据增强
6. 学习率调度器

### Q2: 需要调多少次？

**A:** 取决于资源：
- 最少：10-20 次试验
- 推荐：50-100 次试验
- 充足资源：200+ 次试验

### Q3: 如何判断是否过拟合？

**A:** 观察训练和验证曲线：
```python
if train_acc > 0.95 and val_acc < 0.80:
    print("过拟合！增加正则化")
    # 增加 dropout, weight_decay
    # 或减少模型复杂度
```

### Q4: 调优后性能反而下降？

**A:** 可能原因：
1. 随机性：多次运行取平均
2. 验证集太小：使用交叉验证
3. 过度调优：在验证集上过拟合

## 下一步

- [注意力监督](11_attention_supervision.md) - 提高模型可解释性
- [多视图支持](12_multiview_support.md) - 处理多角度数据
- [模型导出](13_model_export.md) - 部署优化后的模型

## 参考资源

- [Optuna 文档](https://optuna.readthedocs.io/)
- [Ray Tune 文档](https://docs.ray.io/en/latest/tune/)
- [超参数调优最佳实践](../../guides/hyperparameter_tuning.md)
