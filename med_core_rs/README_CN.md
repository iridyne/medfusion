# 🦀 MedCore Rust 加速模块

> **快速开始**: 阅读 [PRACTICAL_GUIDE.md](PRACTICAL_GUIDE.md) 或 [INDEX.md](INDEX.md)

## 📊 核心数据

- **批量处理加速**: 3.5x
- **训练速度提升**: 10-12%
- **数据预处理吞吐量**: +270%

## 🚀 快速使用

```python
from med_core_rs import normalize_intensity_batch

# 批量处理 - 3.5x 加速
images = np.random.rand(32, 512, 512).astype(np.float32) * 255
normalized = normalize_intensity_batch(images, method="percentile")
```

## 📚 文档导航

| 需求 | 文档 |
|------|------|
| 快速上手 | [PRACTICAL_GUIDE.md](PRACTICAL_GUIDE.md) |
| 文档索引 | [INDEX.md](INDEX.md) |
| 性能分析 | [OPTIMIZATION_DEEP_DIVE.md](OPTIMIZATION_DEEP_DIVE.md) |
| API 文档 | [README.md](README.md) |

## ✅ 推荐使用场景

- 批量处理（≥10 张图像）
- 训练数据加载（batch_size ≥ 16）
- 大规模数据预处理

## ❌ 不推荐场景

- 单图像处理（比 NumPy 慢）
- 小批量（<10 张）

## 🎯 最佳实践

```python
def collate_fn(batch):
    images, labels = zip(*batch)
    images = np.stack(images)
    images = normalize_intensity_batch(images, method="percentile")
    return torch.from_numpy(images), torch.tensor(labels)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

**详见**: [PRACTICAL_GUIDE.md](PRACTICAL_GUIDE.md)
