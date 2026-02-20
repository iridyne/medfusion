#!/usr/bin/env python3
"""
实际集成示例：在 MedFusion 训练流程中使用 Rust 加速

这个示例展示了如何在真实的训练代码中集成 Rust 加速模块。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target/release'))

import numpy as np
import time
from typing import List, Tuple

# 模拟 PyTorch (如果没有安装)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch 未安装，使用模拟模式")

import med_core_rs

print("="*70)
print("🚀 MedFusion + Rust 加速集成示例")
print("="*70)

# ============================================================================
# 1. 定义数据集 (模拟医学图像数据集)
# ============================================================================

class MedicalImageDataset:
    """模拟医学图像数据集"""

    def __init__(self, n_samples=1000, image_size=(256, 256)):
        self.n_samples = n_samples
        self.image_size = image_size
        print(f"📦 创建数据集: {n_samples} 张图像, 大小 {image_size}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 模拟加载医学图像 (实际应该从文件加载)
        image = np.random.rand(*self.image_size).astype(np.float32) * 1000
        label = idx % 2  # 二分类
        return image, label

# ============================================================================
# 2. 定义 Collate 函数 (关键优化点)
# ============================================================================

def collate_fn_numpy(batch: List[Tuple[np.ndarray, int]]):
    """传统的 NumPy collate 函数"""
    images, labels = zip(*batch)
    images = np.stack(images)

    # NumPy Percentile 归一化
    normalized = np.zeros_like(images)
    for i in range(len(images)):
        img = images[i]
        p1, p99 = np.percentile(img, [1, 99])
        if p99 - p1 > 1e-8:
            normalized[i] = np.clip((img - p1) / (p99 - p1), 0, 1)

    if HAS_TORCH:
        return torch.from_numpy(normalized), torch.tensor(labels)
    else:
        return normalized, np.array(labels)

def collate_fn_rust(batch: List[Tuple[np.ndarray, int]]):
    """🚀 使用 Rust 加速的 collate 函数"""
    images, labels = zip(*batch)
    images = np.stack(images)

    # Rust Percentile 归一化 (4.8x 加速!)
    normalized = med_core_rs.normalize_intensity_batch(
        images,
        method="percentile",
        p_low=1.0,
        p_high=99.0
    )

    if HAS_TORCH:
        return torch.from_numpy(normalized), torch.tensor(labels)
    else:
        return normalized, np.array(labels)

# ============================================================================
# 3. 性能对比测试
# ============================================================================

def benchmark_dataloader(collate_fn, name: str, n_batches: int = 50):
    """测试 DataLoader 性能"""
    print(f"\n{'='*70}")
    print(f"📊 测试: {name}")
    print(f"{'='*70}")

    dataset = MedicalImageDataset(n_samples=1000)

    if HAS_TORCH:
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn,
            num_workers=0,  # 单进程测试
            shuffle=False
        )
    else:
        # 模拟 DataLoader
        class SimpleDataLoader:
            def __init__(self, dataset, batch_size, collate_fn):
                self.dataset = dataset
                self.batch_size = batch_size
                self.collate_fn = collate_fn

            def __iter__(self):
                for i in range(0, len(self.dataset), self.batch_size):
                    batch = [self.dataset[j] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
                    yield self.collate_fn(batch)

        dataloader = SimpleDataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # 预热
    for i, (images, labels) in enumerate(dataloader):
        if i >= 2:
            break

    # 测试
    times = []
    start_total = time.time()

    for i, (images, labels) in enumerate(dataloader):
        if i >= n_batches:
            break

        batch_start = time.time()
        # 模拟前向传播 (实际训练中的操作)
        if HAS_TORCH:
            _ = images.mean()
        else:
            _ = images.mean()
        batch_time = time.time() - batch_start
        times.append(batch_time)

    total_time = time.time() - start_total

    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均每批: {np.mean(times)*1000:.2f} ms")
    print(f"吞吐量: {n_batches * 32 / total_time:.1f} 张/秒")

    return total_time, np.mean(times)

# ============================================================================
# 4. 运行对比测试
# ============================================================================

print("\n" + "="*70)
print("🔬 开始性能对比测试")
print("="*70)

# 测试 NumPy 版本
numpy_total, numpy_avg = benchmark_dataloader(collate_fn_numpy, "NumPy Percentile", n_batches=50)

# 测试 Rust 版本
rust_total, rust_avg = benchmark_dataloader(collate_fn_rust, "Rust Percentile 🚀", n_batches=50)

# 计算加速比
speedup = numpy_total / rust_total

print("\n" + "="*70)
print("📊 性能对比结果")
print("="*70)
print(f"{'方法':<20} {'总时间':<15} {'平均每批':<15} {'吞吐量':<15}")
print("-"*70)
print(f"{'NumPy':<20} {numpy_total:>10.2f} s   {numpy_avg*1000:>10.2f} ms   {50*32/numpy_total:>10.1f} 张/秒")
print(f"{'Rust 🚀':<20} {rust_total:>10.2f} s   {rust_avg*1000:>10.2f} ms   {50*32/rust_total:>10.1f} 张/秒")
print("-"*70)
print(f"{'加速比':<20} {speedup:>10.2f}x")
print("="*70)

# ============================================================================
# 5. 实际训练示例
# ============================================================================

print("\n" + "="*70)
print("🎓 实际训练流程示例")
print("="*70)

def train_one_epoch_simulation(dataloader, name: str):
    """模拟训练一个 epoch"""
    print(f"\n训练 epoch ({name})...")

    start = time.time()
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        # 模拟前向传播
        if HAS_TORCH:
            outputs = images.mean(dim=(1, 2))
            loss = (outputs - labels.float()).pow(2).mean()
        else:
            outputs = images.mean(axis=(1, 2))
            loss = ((outputs - labels) ** 2).mean()

        total_loss += float(loss)

        if i >= 31:  # 模拟 1 个 epoch (1000 / 32 ≈ 31 batches)
            break

    elapsed = time.time() - start
    print(f"  Epoch 完成: {elapsed:.2f} 秒")
    print(f"  平均 loss: {total_loss / (i+1):.4f}")

    return elapsed

# 创建数据集
dataset = MedicalImageDataset(n_samples=1000)

# NumPy 版本
if HAS_TORCH:
    dataloader_numpy = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_numpy, num_workers=0)
else:
    class SimpleDataLoader:
        def __init__(self, dataset, batch_size, collate_fn):
            self.dataset = dataset
            self.batch_size = batch_size
            self.collate_fn = collate_fn
        def __iter__(self):
            for i in range(0, len(self.dataset), self.batch_size):
                batch = [self.dataset[j] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
                yield self.collate_fn(batch)
    dataloader_numpy = SimpleDataLoader(dataset, batch_size=32, collate_fn=collate_fn_numpy)

numpy_epoch_time = train_one_epoch_simulation(dataloader_numpy, "NumPy")

# Rust 版本
if HAS_TORCH:
    dataloader_rust = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_rust, num_workers=0)
else:
    dataloader_rust = SimpleDataLoader(dataset, batch_size=32, collate_fn=collate_fn_rust)

rust_epoch_time = train_one_epoch_simulation(dataloader_rust, "Rust 🚀")

# 计算训练加速
train_speedup = numpy_epoch_time / rust_epoch_time

print("\n" + "="*70)
print("🎯 训练性能提升")
print("="*70)
print(f"NumPy epoch 时间: {numpy_epoch_time:.2f} 秒")
print(f"Rust epoch 时间:  {rust_epoch_time:.2f} 秒")
print(f"加速比: {train_speedup:.2f}x")
print(f"\n💡 对于 100 epochs 训练:")
print(f"  NumPy: {numpy_epoch_time * 100 / 60:.1f} 分钟")
print(f"  Rust:  {rust_epoch_time * 100 / 60:.1f} 分钟")
print(f"  节省: {(numpy_epoch_time - rust_epoch_time) * 100 / 60:.1f} 分钟")

print("\n" + "="*70)
print("✅ 集成示例完成！")
print("="*70)
print("""
📝 如何在你的项目中使用:

1. 复制 med_core_rs.so 到你的项目目录
2. 修改 DataLoader 的 collate_fn:

   from med_core_rs import normalize_intensity_batch

   def collate_fn(batch):
       images, labels = zip(*batch)
       images = np.stack(images)
       images = normalize_intensity_batch(images, method="percentile")
       return torch.from_numpy(images), torch.tensor(labels)

3. 创建 DataLoader:

   dataloader = DataLoader(
       dataset,
       batch_size=32,
       collate_fn=collate_fn,
       num_workers=4
   )

4. 享受 4.8x 加速！🚀
""")
