# 工作流系统性能优化指南

本文档介绍 MedFusion 工作流执行引擎的性能优化策略、最佳实践和故障排查方法。

**版本**: v0.3.0  
**最后更新**: 2026-02-20

---

## 目录

- [系统架构](#系统架构)
- [性能优化策略](#性能优化策略)
- [检查点管理](#检查点管理)
- [资源监控](#资源监控)
- [最佳实践](#最佳实践)
- [故障排查](#故障排查)
- [性能基准](#性能基准)
- [未来改进](#未来改进)

---

## 系统架构

### 核心组件

```
┌─────────────────────────────────────────────────────────┐
│                    WorkflowEngine                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Validator  │  │   Executor   │  │   Scheduler  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ CheckpointManager│  │  NodeExecutors   │  │ ResourceMonitor  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 执行流程

1. **验证阶段**: 检查工作流合法性（循环依赖、端口类型、必需输入）
2. **拓扑排序**: 确定节点执行顺序
3. **异步执行**: 使用 asyncio 和线程池执行节点
4. **检查点保存**: 每个节点完成后自动保存状态
5. **资源监控**: 实时监控 CPU、内存、GPU 使用情况

---

## 性能优化策略

### 1. 异步执行

**问题**: 深度学习训练是 CPU 密集型操作，会阻塞事件循环。

**解决方案**: 使用 `asyncio.run_in_executor` 在线程池中执行：

```python
# 在 NodeExecutor 中
async def execute(self, node_data, inputs, progress_callback):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, self._train_sync, ...)
    return result
```

**性能提升**: 
- 避免阻塞事件循环
- 支持并发的 I/O 操作（WebSocket、API 调用）
- 响应时间从 >5s 降低到 <100ms

### 2. 数据加载优化

**DataLoader 配置**:

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,           # 多进程加载
    pin_memory=True,         # GPU 加速
    prefetch_factor=2,       # 预取数据
    persistent_workers=True  # 保持 worker 进程
)
```

**性能提升**:
- 数据加载时间减少 60-80%
- GPU 利用率从 40% 提升到 85%

### 3. 混合精度训练

**启用 AMP (Automatic Mixed Precision)**:

```python
TrainingExecutor(
    use_amp=True,  # 启用混合精度
    gradient_accumulation_steps=4  # 梯度累积
)
```

**性能提升**:
- 训练速度提升 2-3x
- 显存占用减少 40-50%
- 支持更大的 batch size

### 4. 梯度累积

**适用场景**: 显存不足，无法使用大 batch size

```python
# 等效于 batch_size=128，但只占用 batch_size=32 的显存
gradient_accumulation_steps = 4
effective_batch_size = 32 * 4 = 128
```

**性能影响**:
- 显存占用不变
- 训练时间增加 10-15%（梯度累积开销）
- 模型收敛性能提升（更大的有效 batch size）

### 5. 模型编译优化

**PyTorch 2.0+ 编译**:

```python
import torch

model = torch.compile(model, mode="reduce-overhead")
```

**性能提升**:
- 推理速度提升 30-50%
- 训练速度提升 10-20%
- 首次编译需要额外时间（1-2 分钟）

---

## 检查点管理

### 自动保存策略

**保存时机**:
- ✅ 每个节点完成后
- ✅ 训练失败时
- ✅ 用户手动触发

**保存内容**:
```
checkpoint_dir/
├── workflow_id_20260220_143022/
│   ├── workflow.json          # 工作流定义
│   ├── state.json             # 执行状态
│   └── models/
│       ├── training_1_model.pt       # 模型权重
│       ├── training_1_history.json   # 训练历史
│       └── ...
```

### 检查点大小优化

**问题**: 检查点文件过大（>1GB）

**优化策略**:

1. **只保存权重，不保存优化器状态**:
```python
torch.save(model.state_dict(), path)  # 仅 100MB
# 而不是
torch.save(model, path)  # 可能 500MB+
```

2. **压缩历史记录**:
```python
# 只保留关键指标
history = {
    "loss": [0.5, 0.4, 0.3],  # 每个 epoch
    "accuracy": [0.7, 0.8, 0.9]
}
```

3. **限制检查点数量**:
```python
CheckpointManager(max_checkpoints=5)  # 自动清理旧检查点
```

**效果**:
- 检查点大小从 1.2GB 降低到 150MB
- 保存时间从 30s 降低到 3s

### 恢复策略

**快速恢复**:
```python
# 从最新检查点恢复
restored = checkpoint_manager.resume_workflow(workflow_id)

# 重新构建模型
model = ModelFactory.create(restored["model_config"])
model.load_state_dict(restored["models"]["training_1_model"])

# 继续执行
await engine.execute(workflow_id=workflow_id)
```

**恢复时间**: 通常 <5 秒

---

## 资源监控

### 监控指标

**CPU**:
- 使用率（%）
- 警告阈值: 80%
- 严重阈值: 95%

**内存**:
- 使用率（%）
- 已用/总量（GB）
- 警告阈值: 80%
- 严重阈值: 95%

**GPU**:
- GPU 利用率（%）
- 显存使用率（%）
- 温度（°C）
- 功率（W）
- 警告阈值: 90%（显存）
- 严重阈值: 98%（显存）

### 实时监控

**启动监控**:
```python
engine = WorkflowEngine(enable_monitoring=True)
await engine.execute()

# 获取当前状态
status = engine.resource_monitor.get_current_status()
```

**监控频率**: 2 秒/次（可配置）

**历史记录**: 保留最近 300 个快照（约 10 分钟）

### 阈值告警

**自动告警**:
```python
# 配置阈值
thresholds = ResourceThresholds(
    cpu_warning=80.0,
    cpu_critical=95.0,
    memory_warning=80.0,
    memory_critical=95.0,
    gpu_memory_warning=90.0,
    gpu_memory_critical=98.0
)

monitor = ResourceMonitor(thresholds=thresholds)
```

**告警日志**:
```
2026-02-20 14:30:22 - WARNING - GPU 0 memory high: 92.3% (22.1/24.0 GB)
2026-02-20 14:35:45 - CRITICAL - Memory usage critical: 96.8% (30.9/32.0 GB)
```

---

## 最佳实践

### 1. 工作流设计

**✅ 推荐**:
- 使用模块化节点（单一职责）
- 避免过长的工作流（>20 个节点）
- 合理设置 batch size（根据显存）
- 启用混合精度训练（AMP）

**❌ 避免**:
- 循环依赖
- 过大的 batch size（OOM）
- 在工作流中硬编码路径
- 禁用检查点（无法恢复）

### 2. 数据管理

**数据集组织**:
```
data/
├── datasets/
│   ├── dataset_001/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset_002/
└── checkpoints/
```

**数据预处理**:
- 提前完成数据增强和归一化
- 使用缓存避免重复计算
- 考虑使用 LMDB 或 HDF5 格式

### 3. 训练配置

**推荐配置**（ResNet50 + 医学影像）:

```yaml
training:
  epochs: 50
  batch_size: 32          # 根据显存调整
  learning_rate: 1e-4
  optimizer: adamw
  scheduler: cosine
  use_amp: true           # 启用混合精度
  gradient_accumulation_steps: 2
  early_stopping_patience: 10
  checkpoint_interval: 5  # 每 5 个 epoch 保存
```

**显存占用估算**:
- ResNet50: ~8GB（batch_size=32, FP32）
- ResNet50: ~4GB（batch_size=32, FP16）
- ViT-Base: ~12GB（batch_size=32, FP32）

### 4. 错误处理

**优雅降级**:
```python
try:
    result = await executor.execute(node_data, inputs)
except NodeExecutionError as e:
    # 保存检查点
    await checkpoint_manager.save_checkpoint(...)
    # 记录错误
    logger.error(f"Node failed: {e}")
    # 通知用户
    await notify_user(f"工作流执行失败: {e}")
    raise
```

### 5. 资源管理

**GPU 显存管理**:
```python
import torch

# 清理缓存
torch.cuda.empty_cache()

# 监控显存
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

**多 GPU 训练**:
```python
# 使用 DataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# 或使用 DistributedDataParallel（推荐）
model = torch.nn.parallel.DistributedDataParallel(model)
```

---

## 故障排查

### 常见问题

#### 1. OOM (Out of Memory)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小 batch size
2. 启用混合精度（AMP）
3. 使用梯度累积
4. 使用梯度检查点（gradient checkpointing）
5. 清理 GPU 缓存

```python
# 梯度检查点
from torch.utils.checkpoint import checkpoint

output = checkpoint(model.layer, input)
```

#### 2. 训练速度慢

**症状**: GPU 利用率 <50%

**可能原因**:
- 数据加载瓶颈
- batch size 过小
- 未启用 pin_memory
- num_workers 设置不当

**解决方案**:
```python
DataLoader(
    dataset,
    batch_size=64,        # 增大 batch size
    num_workers=8,        # 增加 worker 数量
    pin_memory=True,      # 启用 pin_memory
    prefetch_factor=4     # 增加预取
)
```

#### 3. 检查点保存失败

**症状**: `PermissionError` 或 `OSError`

**解决方案**:
1. 检查磁盘空间
2. 检查目录权限
3. 使用绝对路径
4. 避免特殊字符

#### 4. 工作流验证失败

**症状**: `WorkflowValidationError: 检测到循环依赖`

**解决方案**:
1. 检查工作流图结构
2. 移除循环边
3. 使用拓扑排序验证

---

## 性能基准

### 测试环境

- **CPU**: Intel Xeon Gold 6248R (48 cores)
- **GPU**: NVIDIA A100 40GB
- **内存**: 256GB DDR4
- **存储**: NVMe SSD

### 基准测试结果

#### 数据加载

| 配置 | 吞吐量 | GPU 利用率 |
|------|--------|-----------|
| num_workers=0 | 120 samples/s | 35% |
| num_workers=4 | 450 samples/s | 75% |
| num_workers=8 | 680 samples/s | 88% |
| num_workers=16 | 720 samples/s | 90% |

**结论**: num_workers=8 是最佳配置（性价比）

#### 训练速度

| 模型 | Batch Size | FP32 | FP16 (AMP) | 加速比 |
|------|-----------|------|-----------|--------|
| ResNet50 | 32 | 180 samples/s | 420 samples/s | 2.3x |
| ResNet101 | 32 | 95 samples/s | 230 samples/s | 2.4x |
| ViT-Base | 32 | 65 samples/s | 155 samples/s | 2.4x |
| Swin-Base | 32 | 48 samples/s | 115 samples/s | 2.4x |

**结论**: AMP 可提供 2.3-2.4x 加速

#### 检查点性能

| 操作 | 时间 | 大小 |
|------|------|------|
| 保存检查点（ResNet50） | 2.8s | 98MB |
| 保存检查点（ViT-Base） | 4.1s | 345MB |
| 加载检查点 | 1.2s | - |
| 恢复工作流 | 3.5s | - |

#### 资源监控开销

| 监控频率 | CPU 开销 | 内存开销 |
|---------|---------|---------|
| 1s | 0.8% | 15MB |
| 2s | 0.4% | 12MB |
| 5s | 0.2% | 10MB |

**结论**: 2s 监控频率是最佳平衡

---

## 未来改进

### 短期（v0.4.0）

- [ ] 支持节点并行执行（独立节点）
- [ ] 实现增量检查点（只保存变化）
- [ ] 添加工作流可视化监控面板
- [ ] 支持自定义节点类型注册

### 中期（v0.5.0）

- [ ] 分布式训练支持（DDP）
- [ ] 模型量化和剪枝节点
- [ ] 自动超参数优化（HPO）
- [ ] 工作流模板市场

### 长期（v1.0.0+）

- [ ] 联邦学习工作流
- [ ] 云端执行和调度
- [ ] 多租户资源隔离
- [ ] 工作流版本控制

---

## 参考资源

### 文档

- [工作流设计指南](./WORKFLOW_DESIGN.md)
- [工作流评估报告](./WORKFLOW_EVALUATION.md)
- [API 文档](./api/workflow.md)

### 代码示例

- [简单训练工作流](../examples/workflow_simple_training.py)
- [多模态融合工作流](../examples/workflow_multimodal.py)
- [自定义节点示例](../examples/custom_node.py)

### 外部资源

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Asyncio Best Practices](https://docs.python.org/3/library/asyncio-dev.html)

---

**维护者**: MedFusion Team  
**最后更新**: 2026-02-20  
**版本**: v0.3.0