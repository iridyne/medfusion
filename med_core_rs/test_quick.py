#!/usr/bin/env python3
"""快速测试 Rust 模块是否正常工作"""

import time

import numpy as np

print("🧪 测试 Rust 加速模块")
print("=" * 60)

# 测试导入
try:
    from med_core_rs import (
        center_crop_rust,
        normalize_intensity_batch,
        normalize_intensity_minmax,
        normalize_intensity_percentile,
    )
    print("✅ 模块导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# 测试 1: 单图像归一化
print("\n📊 测试 1: 单图像归一化")
image = np.random.rand(512, 512).astype(np.float32) * 255
print(f"  输入: {image.shape}, 范围: [{image.min():.2f}, {image.max():.2f}]")

start = time.perf_counter()
normalized = normalize_intensity_minmax(image)
elapsed = time.perf_counter() - start

print(f"  输出: {normalized.shape}, 范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
print(f"  ⏱️  耗时: {elapsed*1000:.2f} ms")

# 测试 2: Percentile 归一化
print("\n📊 测试 2: Percentile 归一化")
start = time.perf_counter()
normalized = normalize_intensity_percentile(image, 1.0, 99.0)
elapsed = time.perf_counter() - start
print(f"  输出: {normalized.shape}, 范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
print(f"  ⏱️  耗时: {elapsed*1000:.2f} ms")

# 测试 3: 批量处理
print("\n📊 测试 3: 批量处理 (100 张图像)")
images = np.random.rand(100, 512, 512).astype(np.float32) * 255
print(f"  输入: {images.shape}")

start = time.perf_counter()
normalized_batch = normalize_intensity_batch(images, method="percentile")
elapsed = time.perf_counter() - start

print(f"  输出: {normalized_batch.shape}")
print(f"  ⏱️  总耗时: {elapsed*1000:.2f} ms")
print(f"  ⏱️  单张耗时: {elapsed/100*1000:.2f} ms")
print(f"  📈 吞吐量: {100/elapsed:.1f} 张/秒")

# 测试 4: 中心裁剪
print("\n📊 测试 4: 中心裁剪")
image = np.random.rand(1024, 1024).astype(np.float32)
start = time.perf_counter()
cropped = center_crop_rust(image, 224, 224)
elapsed = time.perf_counter() - start
print(f"  输入: {image.shape} -> 输出: {cropped.shape}")
print(f"  ⏱️  耗时: {elapsed*1000:.2f} ms")

# 测试 5: 正确性验证
print("\n📊 测试 5: 正确性验证")
test_img = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
result = normalize_intensity_minmax(test_img)
expected_min = 0.0
expected_max = 1.0
assert abs(result.min() - expected_min) < 1e-5, "最小值不正确"
assert abs(result.max() - expected_max) < 1e-5, "最大值不正确"
print("  ✅ MinMax 归一化正确")

print("\n" + "=" * 60)
print("🎉 所有测试通过！Rust 模块工作正常！")
print("\n下一步:")
print("  1. 运行完整基准测试: python benchmark_comparison.py")
print("  2. 查看集成示例: python example_integration.py")
