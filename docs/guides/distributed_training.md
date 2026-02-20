# 分布式训练指南

本指南介绍如何使用 DDP 和 FSDP 进行分布式训练。

## 概述

MedFusion 支持两种分布式训练策略：

1. **DDP (DistributedDataParallel)**: 数据并行，适合中等规模模型
2. **FSDP (Fully Sharded Data Parallel)**: 完全分片，适合大规模模型

## 快速开始

### 单机多卡训练

```bash
# DDP 训练（4 个 GPU）
torchrun --nproc_per_node=4 train.py --strategy ddp

# FSDP 训练（4 个 GPU）
torchrun --nproc_per_node=4 train.py --strategy fsdp
```

### 多机多卡训练

```bash
# 节点 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=192.168.1.1 --master_port=29500 \
         train.py --strategy ddp

# 节点 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=192.168.1.1 --master_port=29500 \
         train.py --strategy ddp
```

## DDP 使用

### 基本用法

```python
from med_core.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    DDPWrapper,
)

# 设置分布式环境
rank, local_rank, world_size = setup_distributed()

# 创建模型
model = MyModel()
model = DDPWrapper(model)

# 训练...

# 清理
cleanup_distributed()
```

### 完整示例

```python
import torch
from torch.utils.data import DataLoader, DistributedSampler
from med_core.utils.distributed import *

# 设置
rank, local_rank, world_size = setup_distributed()
device = torch.device(f"cuda:{local_rank}")

# 模型
model = MyModel()
model = DDPWrapper(model)

# 数据
dataset = MyDataset()
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

# 训练
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 保存检查点（仅主进程）
    save_checkpoint(model, optimizer, epoch, "checkpoint.pt")

cleanup_distributed()
```

## FSDP 使用

### 基本用法

```python
from med_core.utils.distributed import FSDPWrapper

# 创建模型
model = MyModel()
model = FSDPWrapper(
    model,
    sharding_strategy="FULL_SHARD",
    min_num_params=1e6,
)
```

### 分片策略

- **FULL_SHARD**: 完全分片（最省内存）
- **SHARD_GRAD_OP**: 仅分片梯度和优化器
- **NO_SHARD**: 不分片（类似 DDP）
- **HYBRID_SHARD**: 混合分片

## 工具函数

### 进程管理

```python
from med_core.utils.distributed import *

# 检查是否为主进程
if is_main_process():
    print("I am rank 0")

# 获取 rank 和 world_size
rank = get_rank()
world_size = get_world_size()

# 同步所有进程
barrier()
```

### 指标归约

```python
# 归约指标
metrics = {"loss": torch.tensor(0.5), "acc": torch.tensor(0.9)}
avg_metrics = reduce_dict(metrics)
```

### 检查点保存

```python
# 保存（仅主进程）
save_checkpoint(model, optimizer, epoch, "checkpoint.pt")

# 加载
checkpoint = load_checkpoint(model, optimizer, "checkpoint.pt")
```

## 最佳实践

1. 使用 DistributedSampler
2. 每个 epoch 调用 sampler.set_epoch()
3. 仅在主进程保存检查点
4. 使用 reduce_dict 归约指标
5. 训练结束后调用 cleanup_distributed()

## 性能对比

| 策略 | 内存使用 | 通信开销 | 适用场景 |
|------|---------|---------|---------|
| DDP | 高 | 低 | 中等模型 |
| FSDP | 低 | 中 | 大模型 |

## 参考资源

- `med_core/utils/distributed.py`
- `examples/distributed_training_demo.py`
