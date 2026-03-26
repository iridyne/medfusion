# 监控训练进度

**预计时间：15分钟**

本教程介绍如何使用 TensorBoard 和 Weights & Biases 监控训练进度。

## TensorBoard 监控

### 启用 TensorBoard

```yaml
# configs/my_config.yaml
logging:
  use_tensorboard: true
  tensorboard_dir: "outputs/tensorboard"
  log_every_n_steps: 10
```

### 启动 TensorBoard

```bash
# 训练时自动记录日志
uv run medfusion train --config configs/my_config.yaml

# 在另一个终端启动 TensorBoard
tensorboard --logdir outputs/tensorboard --port 6006

# 访问 http://localhost:6006
```

### 查看指标

TensorBoard 自动记录以下指标：

**训练指标：**
- `train/loss` - 训练损失
- `train/accuracy` - 训练准确率
- `train/learning_rate` - 学习率变化

**验证指标：**
- `val/loss` - 验证损失
- `val/accuracy` - 验证准确率
- `val/auc` - AUC 分数
- `val/f1` - F1 分数

**系统指标：**
- `system/gpu_memory` - GPU 内存使用
- `system/epoch_time` - 每个 epoch 时间

### 可视化示例

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('outputs/tensorboard')

# 记录标量
writer.add_scalar('train/loss', loss.item(), step)

# 记录图像
writer.add_images('train/images', images, step)

# 记录直方图
writer.add_histogram('model/weights', model.fc.weight, step)

# 记录模型图
writer.add_graph(model, input_tensor)

writer.close()
```

## Weights & Biases 集成

### 安装和配置

```bash
# 安装 wandb
uv pip install wandb

# 登录（首次使用）
wandb login
```

### 启用 W&B

```yaml
# configs/my_config.yaml
logging:
  use_wandb: true
  wandb_project: "medfusion-experiments"
  wandb_entity: "your-team"  # 可选
  wandb_run_name: "resnet50-baseline"  # 可选
```

### 训练时自动记录

```bash
uv run medfusion train --config configs/my_config.yaml
```

训练开始后，会自动：
1. 创建 W&B run
2. 记录配置参数
3. 记录训练指标
4. 上传模型检查点
5. 生成可视化报告

### 手动记录

```python
import wandb

# 初始化
wandb.init(
    project="medfusion-experiments",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    }
)

# 记录指标
wandb.log({
    "train/loss": loss.item(),
    "train/accuracy": acc,
    "epoch": epoch
})

# 记录图像
wandb.log({"predictions": wandb.Image(image)})

# 记录表格
wandb.log({"confusion_matrix": wandb.Table(dataframe=df)})

# 保存模型
wandb.save("model.pth")

wandb.finish()
```

### W&B 高级功能

**1. 超参数扫描**

```yaml
# sweep.yaml
program: scripts/run_sweep.py  # 需要你自己封装对 medfusion CLI 的调用
method: bayes
metric:
  name: val/auc
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [16, 32, 64]
  backbone:
    values: ["resnet50", "efficientnet_b0"]
```

```bash
# 创建扫描
wandb sweep sweep.yaml

# 运行 agent
wandb agent your-entity/your-project/sweep-id
```

**2. 模型版本管理**

```python
# 保存模型到 W&B Artifacts
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)

# 加载模型
artifact = wandb.use_artifact('model:latest')
artifact_dir = artifact.download()
```

**3. 实验对比**

在 W&B 网页界面：
1. 选择多个 runs
2. 点击 "Compare"
3. 查看并排对比图表

## 实时日志监控

### 使用 tail 命令

```bash
# 查看训练日志
tail -f outputs/logs/train.log

# 只看错误
tail -f outputs/logs/train.log | grep ERROR

# 实时统计
tail -f outputs/logs/train.log | grep "Epoch"
```

### 日志配置

```yaml
# configs/my_config.yaml
logging:
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_file: "outputs/logs/train.log"
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Python 日志记录

```python
import logging

logger = logging.getLogger(__name__)

# 不同级别的日志
logger.debug("详细调试信息")
logger.info("训练进度信息")
logger.warning("警告信息")
logger.error("错误信息")

# 记录异常
try:
    model.train()
except Exception as e:
    logger.exception("训练失败")
```

## 指标可视化

### 使用 matplotlib

```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取训练历史
history = pd.read_csv('outputs/history.csv')

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['train_loss'], label='Train')
plt.plot(history['epoch'], history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['epoch'], history['train_acc'], label='Train')
plt.plot(history['epoch'], history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig('outputs/training_curves.png')
```

### 使用 seaborn

```python
import seaborn as sns

# 混淆矩阵
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('outputs/confusion_matrix.png')
```

## Web UI 监控

### 启动 Web UI

```bash
# 启动 Web 服务
./start-webui.sh

# 或手动启动
uv run python -m med_core.web.cli web

# 访问 http://localhost:8000
```

### Web UI 功能

**实时监控：**
- 训练进度条
- 实时损失曲线
- GPU 使用率
- 预计剩余时间

**实验管理：**
- 查看所有实验
- 对比实验结果
- 下载检查点
- 查看配置

**可视化：**
- 注意力热力图
- 预测结果
- 特征分布
- ROC 曲线

## 性能监控

### GPU 监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 记录 GPU 使用历史
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv -l 1 > gpu_usage.csv
```

### 系统资源监控

```python
import psutil
import GPUtil

# CPU 使用率
cpu_percent = psutil.cpu_percent(interval=1)

# 内存使用
memory = psutil.virtual_memory()
memory_percent = memory.percent

# GPU 使用
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.load*100:.1f}% | {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
```

### 训练速度分析

```python
import time

# 记录每个 epoch 时间
epoch_start = time.time()
# ... 训练代码 ...
epoch_time = time.time() - epoch_start

# 记录每个 batch 时间
batch_times = []
for batch in dataloader:
    batch_start = time.time()
    # ... 训练代码 ...
    batch_times.append(time.time() - batch_start)

avg_batch_time = sum(batch_times) / len(batch_times)
print(f"平均 batch 时间: {avg_batch_time:.3f}s")
```

## 早停和检查点

### 配置早停

```yaml
# configs/my_config.yaml
training:
  early_stopping:
    enabled: true
    patience: 10  # 10 个 epoch 无改善则停止
    min_delta: 0.001  # 最小改善阈值
    monitor: "val_loss"  # 监控指标
    mode: "min"  # min 或 max
```

### 检查点保存

```yaml
training:
  checkpoint:
    save_best: true  # 保存最佳模型
    save_last: true  # 保存最后一个模型
    save_top_k: 3  # 保存前 3 个最佳模型
    monitor: "val_auc"
    mode: "max"
    save_dir: "outputs/checkpoints"
```

## 实用技巧

### 1. 多实验对比

```bash
# 复制出多份配置，再分别运行
uv run medfusion train --config configs/exp_lr_1e3.yaml
uv run medfusion train --config configs/exp_lr_1e4.yaml
uv run medfusion train --config configs/exp_lr_1e5.yaml
```

### 2. 自动通知

```python
# 训练完成后发送通知
import requests

def send_notification(message):
    # Slack webhook
    webhook_url = "your-slack-webhook-url"
    requests.post(webhook_url, json={"text": message})

# 训练结束时调用
send_notification(f"训练完成！最佳验证 AUC: {best_auc:.4f}")
```

### 3. 定期保存快照

```python
# 每 N 个 epoch 保存一次
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'outputs/checkpoints/snapshot_epoch_{epoch}.pth')
```

## 常见问题

### Q1: TensorBoard 显示不出图表？

**A:** 检查以下几点：
1. 确认日志目录正确
2. 刷新浏览器
3. 检查防火墙设置
4. 使用 `--bind_all` 参数

### Q2: W&B 上传太慢？

**A:** 优化方法：
```python
# 减少记录频率
wandb.log(metrics, step=step, commit=(step % 10 == 0))

# 禁用代码保存
wandb.init(project="...", save_code=False)
```

### Q3: 如何恢复中断的训练？

**A:** 当前稳定 CLI 主链还没有把 `--resume` 收敛成正式入口。
更稳的做法是：

1. 保留已有 checkpoint
2. 重新发起一轮训练并明确新的输出目录
3. 如果你要做真正的断点恢复，建议先在脚本层自己接入

## 下一步

- [超参数调优](tuning.md) - 优化模型性能
- [注意力监督](../advanced/attention.md) - 提高可解释性
- [模型导出](../deployment/model-export.md) - 部署模型

## 参考资源

- [TensorBoard 文档](https://www.tensorflow.org/tensorboard)
- [Weights & Biases 文档](https://docs.wandb.ai/)
- [功能速查](../../guides/core/quick-reference.md)
