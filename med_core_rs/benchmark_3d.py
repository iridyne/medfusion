#!/usr/bin/env python3
"""
Detailed performance analysis for 3D volume processing
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target/release'))

import time

import numpy as np

import med_core_rs

print("="*70)
print("📊 3D 体积处理性能深度分析")
print("="*70)

def benchmark(func, *args, n_runs=10, warmup=2):
    """运行基准测试"""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times), result

# Test different batch sizes
print("\n[测试 1] 不同批量大小的性能")
print("-" * 70)
print(f"{'批量大小':<10} {'Rust (ms)':<15} {'NumPy (ms)':<15} {'加速比':<10}")
print("-" * 70)

for batch_size in [1, 2, 4, 8, 16, 32]:
    volumes = np.random.rand(batch_size, 32, 64, 64).astype(np.float32) * 100

    # Rust
    rust_mean, rust_std, _ = benchmark(
        lambda v: med_core_rs.normalize_3d_batch(v, method="minmax"),
        volumes
    )

    # NumPy
    def numpy_batch(vols):
        result = np.zeros_like(vols)
        for i in range(len(vols)):
            vol = vols[i]
            vmin, vmax = vol.min(), vol.max()
            if vmax - vmin > 1e-8:
                result[i] = (vol - vmin) / (vmax - vmin)
        return result

    numpy_mean, numpy_std, _ = benchmark(numpy_batch, volumes)

    speedup = numpy_mean / rust_mean
    print(f"{batch_size:<10} {rust_mean*1000:>10.2f}±{rust_std*1000:.2f}  "
          f"{numpy_mean*1000:>10.2f}±{numpy_std*1000:.2f}  {speedup:>8.2f}x")

# Test different volume sizes
print("\n[测试 2] 不同体积大小的性能 (batch_size=8)")
print("-" * 70)
print(f"{'体积大小':<15} {'Rust (ms)':<15} {'NumPy (ms)':<15} {'加速比':<10}")
print("-" * 70)

for size in [(16, 32, 32), (32, 64, 64), (64, 128, 128), (128, 256, 256)]:
    batch_size = 8
    volumes = np.random.rand(batch_size, *size).astype(np.float32) * 100

    # Rust
    rust_mean, rust_std, _ = benchmark(
        lambda v: med_core_rs.normalize_3d_batch(v, method="minmax"),
        volumes,
        n_runs=5
    )

    # NumPy
    def numpy_batch(vols):
        result = np.zeros_like(vols)
        for i in range(len(vols)):
            vol = vols[i]
            vmin, vmax = vol.min(), vol.max()
            if vmax - vmin > 1e-8:
                result[i] = (vol - vmin) / (vmax - vmin)
        return result

    numpy_mean, numpy_std, _ = benchmark(numpy_batch, volumes, n_runs=5)

    speedup = numpy_mean / rust_mean
    size_str = f"{size[0]}x{size[1]}x{size[2]}"
    print(f"{size_str:<15} {rust_mean*1000:>10.2f}±{rust_std*1000:.2f}  "
          f"{numpy_mean*1000:>10.2f}±{numpy_std*1000:.2f}  {speedup:>8.2f}x")

# Test single volume vs batch
print("\n[测试 3] 单体积 vs 批量处理")
print("-" * 70)

volume = np.random.rand(32, 64, 64).astype(np.float32) * 100
volumes_batch = np.stack([volume] * 16)

# Single volume (Rust)
single_rust_mean, _, _ = benchmark(
    lambda v: med_core_rs.normalize_3d_minmax(v),
    volume
)

# Batch (Rust)
batch_rust_mean, _, _ = benchmark(
    lambda v: med_core_rs.normalize_3d_batch(v, method="minmax"),
    volumes_batch
)

# Single volume (NumPy)
def numpy_single(vol):
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 1e-8:
        return (vol - vmin) / (vmax - vmin)
    return np.zeros_like(vol)

single_numpy_mean, _, _ = benchmark(numpy_single, volume)

print(f"单体积 Rust:   {single_rust_mean*1000:.2f} ms")
print(f"单体积 NumPy:  {single_numpy_mean*1000:.2f} ms")
print(f"批量 Rust (16): {batch_rust_mean*1000:.2f} ms ({batch_rust_mean*1000/16:.2f} ms/体积)")
print(f"批量效率: {single_rust_mean*16/batch_rust_mean:.2f}x")

print("\n" + "="*70)
print("💡 分析结论")
print("="*70)
print("""
1. NumPy 在小批量时非常快，因为它的 min/max 操作高度优化
2. Rust 的并行开销在小数据量时不值得
3. 需要更大的批量或更复杂的操作才能体现 Rust 优势
4. 建议：只在批量 ≥32 或体积 ≥128³ 时使用 Rust
""")
