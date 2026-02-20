#!/usr/bin/env python3
"""
Test 3D volume preprocessing functions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target/release'))

import numpy as np
import time

try:
    import med_core_rs
    print("✅ med_core_rs 模块加载成功")
except ImportError as e:
    print(f"❌ 无法加载 med_core_rs: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("🧪 3D 体积处理功能测试")
print("="*60)

# Test 1: Single volume normalization
print("\n[测试 1] 单个 3D 体积归一化")
volume = np.random.rand(32, 64, 64).astype(np.float32) * 100
print(f"输入形状: {volume.shape}, 范围: [{volume.min():.2f}, {volume.max():.2f}]")

# MinMax normalization
result_minmax = med_core_rs.normalize_3d_minmax(volume)
print(f"MinMax 归一化: 范围 [{result_minmax.min():.4f}, {result_minmax.max():.4f}]")
assert result_minmax.min() >= 0.0 and result_minmax.max() <= 1.0, "MinMax 归一化失败"
print("✅ MinMax 归一化正确")

# Percentile normalization
result_percentile = med_core_rs.normalize_3d_percentile(volume, p_low=1.0, p_high=99.0)
print(f"Percentile 归一化: 范围 [{result_percentile.min():.4f}, {result_percentile.max():.4f}]")
assert result_percentile.min() >= 0.0 and result_percentile.max() <= 1.0, "Percentile 归一化失败"
print("✅ Percentile 归一化正确")

# Test 2: Batch processing
print("\n[测试 2] 批量 3D 体积处理")
batch_size = 8
volumes = np.random.rand(batch_size, 32, 64, 64).astype(np.float32) * 100
print(f"批量输入形状: {volumes.shape}")

start = time.time()
result_batch = med_core_rs.normalize_3d_batch(volumes, method="percentile")
elapsed = time.time() - start

print(f"批量处理完成: {elapsed*1000:.2f} ms")
print(f"吞吐量: {batch_size/elapsed:.1f} 体积/秒")
print(f"输出形状: {result_batch.shape}")
assert result_batch.shape == volumes.shape, "批量处理形状不匹配"
print("✅ 批量处理正确")

# Test 3: 3D resampling
print("\n[测试 3] 3D 体积重采样")
volume = np.random.rand(64, 128, 128).astype(np.float32)
print(f"原始形状: {volume.shape}")

target_shape = (32, 64, 64)
start = time.time()
resampled = med_core_rs.resample_3d(volume, target_shape[0], target_shape[1], target_shape[2])
elapsed = time.time() - start

print(f"重采样后形状: {resampled.shape}")
print(f"重采样耗时: {elapsed*1000:.2f} ms")
assert resampled.shape == target_shape, "重采样形状不匹配"
print("✅ 重采样正确")

# Test 4: Performance comparison
print("\n[测试 4] 性能对比 (Rust vs NumPy)")
batch_size = 16
volumes = np.random.rand(batch_size, 32, 64, 64).astype(np.float32) * 100

# Rust batch processing
start = time.time()
rust_result = med_core_rs.normalize_3d_batch(volumes, method="minmax")
rust_time = time.time() - start

# NumPy processing
def numpy_normalize_batch(volumes):
    result = np.zeros_like(volumes)
    for i in range(len(volumes)):
        vol = volumes[i]
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin > 1e-8:
            result[i] = (vol - vmin) / (vmax - vmin)
    return result

start = time.time()
numpy_result = numpy_normalize_batch(volumes)
numpy_time = time.time() - start

speedup = numpy_time / rust_time
print(f"Rust 批量处理: {rust_time*1000:.2f} ms ({batch_size/rust_time:.1f} 体积/秒)")
print(f"NumPy 批量处理: {numpy_time*1000:.2f} ms ({batch_size/numpy_time:.1f} 体积/秒)")
print(f"🚀 加速比: {speedup:.2f}x")

# Verify correctness
max_diff = np.abs(rust_result - numpy_result).max()
print(f"最大差异: {max_diff:.6f}")
assert max_diff < 1e-5, "结果不匹配"
print("✅ 结果正确性验证通过")

print("\n" + "="*60)
print("🎉 所有测试通过！")
print("="*60)

print("\n📊 性能总结:")
print(f"  - 单体积归一化: ~{elapsed*1000:.1f} ms")
print(f"  - 批量处理 ({batch_size} 体积): {rust_time*1000:.1f} ms")
print(f"  - 吞吐量: {batch_size/rust_time:.1f} 体积/秒")
print(f"  - 相比 NumPy 加速: {speedup:.2f}x")
print(f"  - 3D 重采样: ~{elapsed*1000:.1f} ms")
