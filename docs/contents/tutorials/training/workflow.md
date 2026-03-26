# 训练工作流

**预计时间：30分钟**

本教程详细介绍 MedFusion 的完整训练工作流，从训练前准备到模型保存和问题排查。

## 训练前检查清单

在启动训练前，确保以下项目已完成：

### 1. 环境检查

```bash
# 检查 Python 版本（需要 3.11+）
python --version

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查 GPU 信息
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 检查依赖安装
uv pip list | grep -E "torch|pandas|numpy"
```

### 2. 数据准备

```bash
# 验证数据文件存在
ls -lh data/dataset.csv
ls -lh data/images/ | head

# 检查数据格式
python -c "import pandas as pd; df = pd.read_csv('data/dataset.csv'); print(df.head()); print(df.info())"

# 验证图像可读取
python -c "from PIL import Image; img = Image.open('data/images/sample_0001.png'); print(f'Image size: {img.size}')"
```

### 3. 配置文件验证

```bash
# 验证配置文件语法
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"

# 使用内置验证器
uv run python -c "from med_core.configs import load_config; config = load_config('configs/my_config.yaml'); print('Config valid!')"
```

### 4. 磁盘空间检查

```bash
# 检查可用空间（至少需要 10GB）
df -h outputs/

# 预估检查点大小
python -c "from med_core.models import MultiModalModelBuilder; builder = MultiModalModelBuilder(num_classes=2); builder.add_modality('ct', backbone='resnet50'); model = builder.build(); print(f'Estimated checkpoint size: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB')"
```

## 启动训练

### 方法 1：使用命令行（推荐）

```bash
# 基础训练（当前稳定入口）
uv run medfusion train --config configs/starter/quickstart.yaml

# 指定输出目录
uv run medfusion train \
  --config configs/starter/quickstart.yaml \
  --output-dir outputs/exp_001
```

当前 CLI 主链实际支持的训练参数只有：

- `--config`
- `--output-dir`

像 `--override`、`--resume` 这类参数目前不是这条主链的稳定接口，不建议按旧文档使用。

### 方法 2：使用 Python 脚本

```python
# train.py
from pathlib import Path
from med_core.configs import load_config
from med_core.datasets import MedicalMultimodalDataset, create_dataloaders, split_dataset
from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.fusion import MultiModalFusionModel, create_fusion_module
from med_core.trainers import MultimodalTrainer
import torch.optim as optim

# 加载配置
config = load_config("configs/starter/quickstart.yaml")

# 准备数据
dataset, preprocessor = MedicalMultimodalDataset.from_csv(
    csv_path=config.data.csv_path,
    image_dir=config.data.image_dir,
    image_column=config.data.image_path_column,
    target_column=config.data.target_column,
    numerical_features=config.data.numerical_features,
    categorical_features=config.data.categorical_features,
)

train_ds, val_ds, test_ds = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15)
dataloaders = create_dataloaders(train_ds, val_ds, test_ds, batch_size=config.data.batch_size)

# 构建模型
vision_backbone = create_vision_backbone(
    backbone_name=config.model.vision.backbone,
    pretrained=config.model.vision.pretrained,
    freeze=config.model.vision.freeze_backbone,
    feature_dim=config.model.vision.feature_dim,
    dropout=config.model.vision.dropout,
    attention_type=config.model.vision.attention_type,
)
tabular_backbone = create_tabular_backbone(
    input_dim=train_ds.get_tabular_dim(),
    output_dim=config.model.tabular.output_dim,
    hidden_dims=config.model.tabular.hidden_dims,
    dropout=config.model.tabular.dropout,
)
fusion_module = create_fusion_module(
    fusion_type=config.model.fusion.fusion_type,
    vision_dim=config.model.vision.feature_dim,
    tabular_dim=config.model.tabular.output_dim,
    output_dim=config.model.fusion.hidden_dim,
    dropout=config.model.fusion.dropout,
)
model = MultiModalFusionModel(
    vision_backbone=vision_backbone,
    tabular_backbone=tabular_backbone,
    fusion_module=fusion_module,
    num_classes=config.model.num_classes,
    use_auxiliary_heads=config.model.use_auxiliary_heads,
)

# 创建优化器
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.training.optimizer.learning_rate,
    weight_decay=config.training.optimizer.weight_decay
)

# 创建训练器
trainer = MultimodalTrainer(
    config=config,
    model=model,
    train_loader=dataloaders["train"],
    val_loader=dataloaders["val"],
    optimizer=optimizer
)

# 开始训练
history = trainer.train()
```

### 方法 3：使用 Web UI

```bash
# 启动 Web 服务
./start-webui.sh

# 或手动启动
uv run python -m med_core.web.cli web
```

访问 http://localhost:8000，通过图形界面配置和启动训练。

## 监控训练进度

### TensorBoard（推荐）

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/exp_001/logs --port 6006

# 在浏览器打开
# http://localhost:6006
```

**关键指标：**
- `train/loss` - 训练损失
- `val/loss` - 验证损失
- `val/auc` - 验证集 AUC
- `learning_rate` - 当前学习率

### Weights & Biases

```yaml
# 在配置文件中启用
logging:
  use_wandb: true
  wandb_project: "medfusion-experiments"
  wandb_entity: "your-team"
```

```bash
# 登录 WandB
wandb login

# 训练时自动上传
uv run medfusion train --config configs/starter/quickstart.yaml
```

### 实时日志监控

```bash
# 查看训练日志
tail -f outputs/exp_001/logs/train.log

# 使用 grep 过滤关键信息
tail -f outputs/exp_001/logs/train.log | grep -E "Epoch|loss|AUC"
```

### 命令行进度条

训练时会显示实时进度：

```
Epoch 10 [Train]: 100%|████████| 125/125 [02:15<00:00, 1.08s/it, loss=0.4523]
Epoch 10 [Val]:   100%|████████| 32/32 [00:28<00:00, 1.12it/s]
Epoch 10: Train Loss: 0.4523 | Val Loss: 0.3891 | Val AUC: 0.8456
New best model saved with val_auc: 0.8456
```

## 解读损失曲线和指标

### 正常训练曲线

```
Loss
 │
 │ ╲
 │  ╲___train_loss
 │   ╲
 │    ╲___val_loss
 │      ╲___
 │         ╲___
 └──────────────> Epoch
```

**特征：**
- 训练损失持续下降
- 验证损失先降后趋于平稳
- 两条曲线间隙适中

### 过拟合曲线

```
Loss
 │
 │ ╲___train_loss
 │     ╲___
 │         ╲___
 │    ╱╱╱╱╱ val_loss
 │  ╱╱
 │╱╱
 └──────────────> Epoch
```

**特征：**
- 训练损失持续下降
- 验证损失先降后上升
- 两条曲线间隙越来越大

**解决方案：**
```yaml
# 增加正则化
model:
  vision:
    dropout: 0.5  # 增加 dropout
  fusion:
    dropout: 0.5

training:
  optimizer:
    weight_decay: 0.05  # 增加权重衰减
  label_smoothing: 0.1  # 标签平滑

# 使用数据增强
data:
  augmentation_strength: "heavy"

# 早停
training:
  early_stopping: true
  patience: 10
```

### 欠拟合曲线

```
Loss
 │
 │ ╲___train_loss
 │     ╲___
 │  ╲___val_loss
 │      ╲___
 │          (两条线都很高)
 └──────────────> Epoch
```

**特征：**
- 训练和验证损失都很高
- 下降缓慢或停滞

**解决方案：**
```yaml
# 增加模型容量
model:
  vision:
    backbone: "resnet50"  # 使用更大的模型
  fusion:
    hidden_dim: 256  # 增加融合层维度

# 提高学习率
training:
  optimizer:
    learning_rate: 0.001  # 增大学习率

# 减少正则化
model:
  vision:
    dropout: 0.1  # 降低 dropout
```

### 学习率过大

```
Loss
 │
 │ ╱╲╱╲╱╲╱╲  (震荡)
 │╱  ╲  ╲
 │    ╲  ╲
 │     ╲  ╲
 └──────────────> Epoch
```

**解决方案：**
```yaml
training:
  optimizer:
    learning_rate: 0.00001  # 降低学习率
  scheduler:
    scheduler: "cosine"
    warmup_epochs: 5  # 使用 warmup
```

## 早停策略

### 配置早停

```yaml
training:
  early_stopping: true
  patience: 15  # 15 个 epoch 无改善则停止
  monitor: "val_auc"  # 监控指标
  mode: "max"  # "max" 表示越大越好，"min" 表示越小越好
  min_delta: 0.001  # 最小改善阈值
```

### 早停触发示例

```
Epoch 45: Val AUC: 0.8523 (best: 0.8534)
Epoch 46: Val AUC: 0.8519 (best: 0.8534)
...
Epoch 60: Val AUC: 0.8528 (best: 0.8534)
Early stopping triggered (patience=15)
Training stopped at epoch 60
```

### 自定义早停逻辑

```python
from med_core.trainers import BaseTrainer

class CustomTrainer(BaseTrainer):
    def _handle_checkpointing(self, val_metrics):
        # 自定义早停逻辑
        current_auc = val_metrics.get('auc', 0)
        current_loss = val_metrics.get('loss', float('inf'))

        # 同时考虑 AUC 和 loss
        if current_auc > self.best_auc and current_loss < self.best_loss:
            self.best_auc = current_auc
            self.best_loss = current_loss
            self.patience_counter = 0
            self._save_checkpoint("best.pth", val_metrics)
        else:
            self.patience_counter += 1
```

## 检查点管理

### 检查点保存策略

```yaml
training:
  save_top_k: 3  # 保存最好的 3 个模型
  save_last: true  # 始终保存最后一个 epoch
  monitor: "val_auc"
  mode: "max"
```

**生成的检查点：**
```
outputs/exp_001/checkpoints/
├── best.pth          # 最佳模型
├── last.pth          # 最后一个 epoch
├── epoch_10.pth      # Top-3 之一
├── epoch_25.pth      # Top-3 之一
└── epoch_42.pth      # Top-3 之一
```

### 检查点内容

```python
import torch

checkpoint = torch.load("outputs/exp_001/checkpoints/best.pth")
print(checkpoint.keys())
# dict_keys(['epoch', 'global_step', 'model_state_dict',
#            'optimizer_state_dict', 'metrics', 'config'])

print(f"Epoch: {checkpoint['epoch']}")
print(f"Metrics: {checkpoint['metrics']}")
```

### 加载检查点

```python
from med_core.trainers import MultimodalTrainer

# 方法 1：从检查点恢复训练
trainer = MultimodalTrainer(config, model, train_loader, val_loader)
trainer.load_checkpoint("outputs/exp_001/checkpoints/last.pth")
trainer.train()  # 继续训练

# 方法 2：仅加载模型权重用于推理
model.load_state_dict(torch.load("outputs/exp_001/checkpoints/best.pth")["model_state_dict"])
model.eval()
```

### 检查点清理

```bash
# 删除旧实验的检查点
rm -rf outputs/old_exp/checkpoints/

# 仅保留最佳模型
cd outputs/exp_001/checkpoints/
ls | grep -v "best.pth" | xargs rm

# 压缩检查点节省空间
tar -czf checkpoints_backup.tar.gz outputs/*/checkpoints/best.pth
```

## 常见问题排查

### 问题 1：CUDA Out of Memory

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案：**
```yaml
# 1. 减小批量大小
data:
  batch_size: 8  # 从 32 降到 8

# 2. 启用混合精度训练
training:
  mixed_precision: true

# 3. 使用梯度累积
training:
  gradient_accumulation_steps: 4  # 等效 batch_size = 8 * 4 = 32

# 4. 减小模型尺寸
model:
  vision:
    backbone: "resnet18"  # 使用更小的模型
```

```bash
# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 问题 2：Loss 变成 NaN

**症状：**
```
Epoch 5 [Train]: loss=nan
```

**原因：**
- 学习率过大
- 梯度爆炸
- 数值不稳定

**解决方案：**
```yaml
training:
  optimizer:
    learning_rate: 0.00001  # 降低学习率
  gradient_clip: 1.0  # 启用梯度裁剪
  mixed_precision: false  # 暂时禁用混合精度

# 检查数据归一化
data:
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

```python
# 检查数据中是否有异常值
import pandas as pd
df = pd.read_csv("data/dataset.csv")
print(df.describe())
print(df.isnull().sum())
```

### 问题 3：训练速度慢

**症状：**
- 每个 epoch 耗时过长
- GPU 利用率低

**解决方案：**
```yaml
# 1. 增加数据加载 workers
data:
  num_workers: 8  # 根据 CPU 核心数调整
  pin_memory: true
  persistent_workers: true

# 2. 启用混合精度
training:
  mixed_precision: true

# 3. 优化数据增强
data:
  augmentation_strength: "light"  # 减少增强操作
```

```bash
# 检查 GPU 利用率
nvidia-smi -l 1

# 检查数据加载瓶颈
python -c "
from med_core.datasets import create_dataloaders
import time
loader = create_dataloaders(..., num_workers=8)['train']
start = time.time()
for i, batch in enumerate(loader):
    if i == 10: break
print(f'Time per batch: {(time.time()-start)/10:.3f}s')
"
```

### 问题 4：验证集指标不稳定

**症状：**
```
Epoch 10: Val AUC: 0.85
Epoch 11: Val AUC: 0.72
Epoch 12: Val AUC: 0.88
```

**原因：**
- 验证集太小
- 批量大小太小
- 模型不稳定

**解决方案：**
```yaml
# 1. 增加验证集大小
data:
  train_ratio: 0.6
  val_ratio: 0.3  # 增加验证集比例
  test_ratio: 0.1

# 2. 使用更大的验证批量
data:
  batch_size: 32
  val_batch_size: 64  # 验证时可以用更大的批量

# 3. 设置随机种子
seed: 42
deterministic: true
```

### 问题 5：模型不收敛

**症状：**
- 损失不下降或下降极慢
- 准确率接近随机猜测

**检查清单：**
```python
# 1. 检查数据标签分布
import pandas as pd
df = pd.read_csv("data/dataset.csv")
print(df['diagnosis'].value_counts())
# 确保类别平衡

# 2. 检查模型输出
model.eval()
with torch.no_grad():
    output = model(sample_image, sample_tabular)
    print(output)  # 检查是否有异常值

# 3. 测试简单样本
# 创建一个明显可分的小数据集，验证模型能否学习
```

**解决方案：**
```yaml
# 1. 使用预训练模型
model:
  vision:
    pretrained: true
    freeze_backbone: false  # 允许微调

# 2. 调整学习率
training:
  optimizer:
    learning_rate: 0.001  # 尝试不同学习率
  scheduler:
    scheduler: "step"
    step_size: 10
    gamma: 0.1

# 3. 处理类别不平衡
training:
  class_weights: [1.0, 2.0]  # 为少数类增加权重
```

## 训练完成后

### 1. 保存最终模型

```bash
# 复制最佳检查点
cp outputs/exp_001/checkpoints/best.pth models/final_model.pth

# 导出为 ONNX（可选）
uv run python -c "
from med_core.configs import load_config
import torch

config = load_config('configs/starter/quickstart.yaml')
from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.fusion import MultiModalFusionModel, create_fusion_module

vision_backbone = create_vision_backbone(
    backbone_name=config.model.vision.backbone,
    pretrained=config.model.vision.pretrained,
    freeze=config.model.vision.freeze_backbone,
    feature_dim=config.model.vision.feature_dim,
    dropout=config.model.vision.dropout,
    attention_type=config.model.vision.attention_type,
)
tabular_backbone = create_tabular_backbone(
    input_dim=max(len(config.data.numerical_features) + len(config.data.categorical_features), 1),
    output_dim=config.model.tabular.output_dim,
    hidden_dims=config.model.tabular.hidden_dims,
    dropout=config.model.tabular.dropout,
)
fusion_module = create_fusion_module(
    fusion_type=config.model.fusion.fusion_type,
    vision_dim=config.model.vision.feature_dim,
    tabular_dim=config.model.tabular.output_dim,
    output_dim=config.model.fusion.hidden_dim,
    dropout=config.model.fusion.dropout,
)
model = MultiModalFusionModel(
    vision_backbone=vision_backbone,
    tabular_backbone=tabular_backbone,
    fusion_module=fusion_module,
    num_classes=config.model.num_classes,
    use_auxiliary_heads=config.model.use_auxiliary_heads,
)
model.load_state_dict(torch.load('models/final_model.pth')['model_state_dict'])
model.eval()

dummy_image = torch.randn(1, 3, 224, 224)
dummy_tabular = torch.randn(1, max(len(config.data.numerical_features) + len(config.data.categorical_features), 1))
torch.onnx.export(model, (dummy_image, dummy_tabular), 'models/final_model.onnx')
"
```

### 2. 生成训练报告

```python
from med_core.evaluation import generate_training_report

report = generate_training_report(
    checkpoint_path="outputs/exp_001/checkpoints/best.pth",
    history_path="outputs/exp_001/logs/history.json",
    output_dir="outputs/exp_001/reports"
)
print(f"Report saved to: {report}")
```

### 3. 评估测试集

```bash
uv run medfusion evaluate \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/exp_001/checkpoints/best.pth \
  --split test \
  --output-dir outputs/exp_001/evaluation
```

### 4. 清理临时文件

```bash
# 删除中间检查点
rm outputs/exp_001/checkpoints/epoch_*.pth

# 压缩日志
gzip outputs/exp_001/logs/*.log

# 归档实验
tar -czf archives/exp_001.tar.gz outputs/exp_001/
```

## 最佳实践

1. **使用版本控制**：将配置文件纳入 Git 管理
2. **记录实验**：为每个实验写清楚的描述和标签
3. **定期备份**：重要检查点及时备份到云存储
4. **监控资源**：使用 `nvidia-smi` 和 `htop` 监控资源使用
5. **渐进式训练**：从小模型、小数据开始，逐步扩展
6. **对比实验**：保持其他变量不变，每次只改变一个超参数

## 下一步

- [评估 API](../../api/evaluation.md) - 详细评估训练好的模型
- [超参数调优](tuning.md) - 系统化调优超参数
- [部署指南](../deployment/production.md) - 将模型部署到生产环境
