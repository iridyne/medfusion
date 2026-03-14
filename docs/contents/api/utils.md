# Utils API

工具函数模块，提供常用的辅助功能。

## 概述

Utils 模块包含以下工具：

- **随机种子**: 确保实验可复现
- **设备管理**: GPU/CPU 自动检测和管理
- **检查点**: 模型保存和加载
- **日志**: 统一的日志系统
- **梯度检查点**: 内存优化技术

## 随机种子

### set_seed

设置所有随机种子以确保可复现性。

**参数：**
- `seed` (int): 随机种子值
- `deterministic` (bool): 是否使用确定性算法，默认 True

**影响范围：**
- Python `random` 模块
- NumPy 随机数生成器
- PyTorch CPU 和 CUDA 随机数生成器
- cuDNN 后端行为

**示例：**
```python
from med_core.utils import set_seed

# 设置随机种子
set_seed(42, deterministic=True)

# 现在所有随机操作都是可复现的
```

**注意：**
- `deterministic=True` 会降低性能（10-30%）
- 生产环境可以设置为 `False` 以提高速度
- 分布式训练需要在每个进程中设置

## 设备管理

### get_device

获取可用的计算设备。

**参数：**
- `device` (str): 设备规格，默认 "auto"
  - `"auto"` - 自动检测（优先 CUDA > MPS > CPU）
  - `"cuda"` - 强制使用 CUDA
  - `"cpu"` - 强制使用 CPU
  - `"mps"` - 强制使用 Apple Silicon GPU

**返回：**
- `torch.device`: PyTorch 设备对象

**示例：**
```python
from med_core.utils import get_device

# 自动检测
device = get_device("auto")
print(device)  # cuda:0 或 cpu

# 强制使用 CPU
device = get_device("cpu")

# 将模型移到设备
model = model.to(device)
```

### get_device_info

获取设备信息。

**返回：**
- `dict[str, Any]`: 设备信息字典

**包含信息：**
- `cuda_available` - CUDA 是否可用
- `cuda_device_count` - CUDA 设备数量
- `cuda_device_name` - GPU 名称
- `cuda_memory_total` - GPU 总显存
- `mps_available` - Apple MPS 是否可用

**示例：**
```python
from med_core.utils import get_device_info

info = get_device_info()
print(f"CUDA 可用: {info['cuda_available']}")
print(f"GPU 数量: {info['cuda_device_count']}")
if info['cuda_available']:
    print(f"GPU 名称: {info['cuda_device_name']}")
    print(f"显存: {info['cuda_memory_total'] / 1e9:.2f} GB")
```

### move_to_device

将张量或模型移动到指定设备。

**参数：**
- `obj` (Any): 张量、模型、字典或列表
- `device` (torch.device): 目标设备

**返回：**
- `Any`: 移动后的对象

**支持类型：**
- `torch.Tensor` - 单个张量
- `nn.Module` - 模型
- `dict` - 张量字典（递归处理）
- `list/tuple` - 张量列表（递归处理）

**示例：**
```python
from med_core.utils import move_to_device, get_device

device = get_device("cuda")

# 移动单个张量
tensor = torch.randn(10, 20)
tensor = move_to_device(tensor, device)

# 移动字典
batch = {
    'images': torch.randn(32, 3, 224, 224),
    'tabular': torch.randn(32, 64),
    'labels': torch.randint(0, 2, (32,))
}
batch = move_to_device(batch, device)

# 移动模型
model = move_to_device(model, device)
```

## 检查点管理

### save_checkpoint

保存模型检查点。

**参数：**
- `model` (nn.Module): 模型
- `optimizer` (Optimizer): 优化器
- `epoch` (int): 当前轮次
- `filepath` (str | Path): 保存路径
- `metrics` (dict[str, float]): 可选的指标字典
- `scheduler` (Any): 可选的学习率调度器
- `**kwargs`: 其他要保存的内容

**示例：**
```python
from med_core.utils import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=50,
    filepath="checkpoints/epoch_50.pth",
    metrics={
        'train_loss': 0.234,
        'val_loss': 0.267,
        'val_auc': 0.892
    },
    scheduler=scheduler,
    config=config.to_dict()
)
```

### load_checkpoint

加载模型检查点。

**参数：**
- `filepath` (str | Path): 检查点路径
- `model` (nn.Module): 模型
- `optimizer` (Optimizer): 可选的优化器
- `scheduler` (Any): 可选的调度器
- `device` (str): 设备，默认 "cpu"

**返回：**
- `dict[str, Any]`: 检查点元数据

**示例：**
```python
from med_core.utils import load_checkpoint

checkpoint = load_checkpoint(
    filepath="checkpoints/best.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda"
)

print(f"从轮次 {checkpoint['epoch']} 恢复")
print(f"验证 AUC: {checkpoint['metrics']['val_auc']:.4f}")
```

### find_best_checkpoint

查找最佳检查点。

**参数：**
- `checkpoint_dir` (str | Path): 检查点目录
- `metric` (str): 指标名称，默认 "val_loss"
- `mode` (str): 模式 ("min" 或 "max")，默认 "min"

**返回：**
- `Path`: 最佳检查点路径

**示例：**
```python
from med_core.utils import find_best_checkpoint

best_ckpt = find_best_checkpoint(
    checkpoint_dir="outputs/checkpoints",
    metric="val_auc",
    mode="max"
)

print(f"最佳检查点: {best_ckpt}")
```

### cleanup_checkpoints

清理旧的检查点文件。

**参数：**
- `checkpoint_dir` (str | Path): 检查点目录
- `keep_last_n` (int): 保留最近 N 个检查点，默认 5
- `keep_best` (bool): 是否保留 best.pth，默认 True

**示例：**
```python
from med_core.utils import cleanup_checkpoints

# 只保留最近 3 个检查点和 best.pth
cleanup_checkpoints(
    checkpoint_dir="outputs/checkpoints",
    keep_last_n=3,
    keep_best=True
)
```

## 日志系统

### setup_logging

配置日志系统。

**参数：**
- `log_level` (str): 日志级别，默认 "INFO"
- `log_file` (str | Path): 可选的日志文件路径
- `format_string` (str): 可选的日志格式

**示例：**
```python
from med_core.utils import setup_logging

# 基础配置
setup_logging(log_level="INFO")

# 保存到文件
setup_logging(
    log_level="DEBUG",
    log_file="outputs/training.log"
)
```

### get_logger

获取日志记录器。

**参数：**
- `name` (str): 日志记录器名称

**返回：**
- `logging.Logger`: 日志记录器对象

**示例：**
```python
from med_core.utils import get_logger

logger = get_logger(__name__)

logger.info("开始训练")
logger.debug("批次大小: 32")
logger.warning("学习率较高，可能不稳定")
logger.error("加载数据失败")
```

## 梯度检查点

梯度检查点是一种内存优化技术，通过重新计算中间激活值来减少显存占用。

### apply_gradient_checkpointing

为模型应用梯度检查点。

**参数：**
- `model` (nn.Module): 模型
- `checkpoint_segments` (int): 检查点段数，默认 2

**示例：**
```python
from med_core.utils import apply_gradient_checkpointing

# 应用梯度检查点
model = apply_gradient_checkpointing(model, checkpoint_segments=4)

# 显存占用减少 50-70%，训练速度降低 20-30%
```

### checkpoint_sequential

对顺序模块应用梯度检查点。

**参数：**
- `functions` (list[nn.Module]): 模块列表
- `segments` (int): 分段数
- `input` (torch.Tensor): 输入张量

**返回：**
- `torch.Tensor`: 输出张量

**示例：**
```python
from med_core.utils import checkpoint_sequential

layers = [layer1, layer2, layer3, layer4]
output = checkpoint_sequential(layers, segments=2, input=x)
```

### estimate_memory_savings

估算梯度检查点的显存节省。

**参数：**
- `model` (nn.Module): 模型
- `input_size` (tuple): 输入大小

**返回：**
- `dict[str, float]`: 显存统计

**示例：**
```python
from med_core.utils import estimate_memory_savings

savings = estimate_memory_savings(
    model=model,
    input_size=(32, 3, 224, 224)
)

print(f"原始显存: {savings['original_memory_mb']:.2f} MB")
print(f"优化后显存: {savings['optimized_memory_mb']:.2f} MB")
print(f"节省: {savings['savings_percent']:.1f}%")
```

## 使用示例

### 完整训练脚本

```python
from med_core.utils import (
    set_seed,
    get_device,
    get_device_info,
    setup_logging,
    get_logger,
    save_checkpoint,
    load_checkpoint
)

# 1. 设置日志
setup_logging(log_level="INFO", log_file="training.log")
logger = get_logger(__name__)

# 2. 设置随机种子
set_seed(42, deterministic=True)
logger.info("随机种子已设置")

# 3. 检查设备
device_info = get_device_info()
logger.info(f"设备信息: {device_info}")
device = get_device("auto")
logger.info(f"使用设备: {device}")

# 4. 训练循环
for epoch in range(num_epochs):
    # 训练代码...

    # 保存检查点
    if epoch % 10 == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            filepath=f"checkpoints/epoch_{epoch}.pth",
            metrics=metrics
        )
        logger.info(f"检查点已保存: epoch {epoch}")

# 5. 加载最佳模型
checkpoint = load_checkpoint(
    filepath="checkpoints/best.pth",
    model=model,
    device=device
)
logger.info(f"加载最佳模型: epoch {checkpoint['epoch']}")
```

### 显存优化

```python
from med_core.utils import (
    apply_gradient_checkpointing,
    estimate_memory_savings
)

# 估算显存节省
savings = estimate_memory_savings(
    model=model,
    input_size=(32, 3, 224, 224)
)
print(f"预计节省显存: {savings['savings_percent']:.1f}%")

# 应用梯度检查点
if savings['savings_percent'] > 30:
    model = apply_gradient_checkpointing(model, checkpoint_segments=4)
    print("已启用梯度检查点")
```

## 最佳实践

**随机种子：**
- 研究实验使用 `deterministic=True`
- 生产环境使用 `deterministic=False`
- 记录种子值到配置文件

**设备管理：**
- 使用 `get_device("auto")` 自动检测
- 训练前检查 `get_device_info()`
- 使用 `move_to_device()` 统一处理

**检查点：**
- 定期保存检查点（每 5-10 轮）
- 始终保存 best.pth
- 定期清理旧检查点

**日志：**
- 训练使用 INFO 级别
- 调试使用 DEBUG 级别
- 保存日志到文件

**梯度检查点：**
- 显存不足时使用
- 权衡显存和速度
- 大模型推荐使用

## 参考

完整实现请参考：
- `/home/yixian/Projects/med-ml/medfusion/med_core/utils/seed.py` - 随机种子
- `/home/yixian/Projects/med-ml/medfusion/med_core/utils/device.py` - 设备管理
- `/home/yixian/Projects/med-ml/medfusion/med_core/utils/checkpoint.py` - 检查点管理
- `/home/yixian/Projects/med-ml/medfusion/med_core/utils/logging.py` - 日志系统
- `/home/yixian/Projects/med-ml/medfusion/med_core/utils/gradient_checkpointing.py` - 梯度检查点
