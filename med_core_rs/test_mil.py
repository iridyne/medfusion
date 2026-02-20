#!/usr/bin/env python3
"""
Test MIL aggregation functions
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
print("🧪 MIL 聚合器功能测试")
print("="*60)

# Test 1: Max pooling
print("\n[测试 1] Max Pooling MIL")
instances = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 1.0, 2.0, 3.0],
    [2.0, 4.0, 1.0, 5.0],
], dtype=np.float32)
print(f"输入实例: {instances.shape}")

result = med_core_rs.max_pooling_mil(instances)
print(f"Max pooling 结果: {result}")
expected = np.array([5.0, 4.0, 3.0, 5.0])
assert np.allclose(result, expected), f"期望 {expected}, 得到 {result}"
print("✅ Max pooling 正确")

# Test 2: Mean pooling
print("\n[测试 2] Mean Pooling MIL")
instances = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
], dtype=np.float32)

result = med_core_rs.mean_pooling_mil(instances)
print(f"Mean pooling 结果: {result}")
expected = np.array([3.0, 4.0])
assert np.allclose(result, expected), f"期望 {expected}, 得到 {result}"
print("✅ Mean pooling 正确")

# Test 3: Attention MIL
print("\n[测试 3] Attention MIL")
instances = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
], dtype=np.float32)
attention_weights = np.array([[0.5], [0.3], [0.2]], dtype=np.float32)

result = med_core_rs.attention_mil(instances, attention_weights)
print(f"Attention MIL 结果: {result}")
# Expected: 0.5*[1,2] + 0.3*[3,4] + 0.2*[5,6] = [2.4, 3.4]
expected = np.array([2.4, 3.4])
assert np.allclose(result, expected), f"期望 {expected}, 得到 {result}"
print("✅ Attention MIL 正确")

# Test 4: Batch processing
print("\n[测试 4] 批量 MIL 聚合")
n_bags = 100
bags = [np.random.rand(np.random.randint(10, 50), 512).astype(np.float32)
        for _ in range(n_bags)]

print(f"批量大小: {n_bags} bags")
print(f"特征维度: 512")

# Max pooling
start = time.time()
result_max = med_core_rs.batch_mil_aggregation(bags, method="max")
time_max = time.time() - start
print(f"Max pooling: {time_max*1000:.2f} ms ({n_bags/time_max:.1f} bags/秒)")
assert result_max.shape == (n_bags, 512), f"形状不匹配: {result_max.shape}"

# Mean pooling
start = time.time()
result_mean = med_core_rs.batch_mil_aggregation(bags, method="mean")
time_mean = time.time() - start
print(f"Mean pooling: {time_mean*1000:.2f} ms ({n_bags/time_mean:.1f} bags/秒)")
assert result_mean.shape == (n_bags, 512), f"形状不匹配: {result_mean.shape}"

print("✅ 批量处理正确")

# Test 5: Performance comparison
print("\n[测试 5] 性能对比 (Rust vs NumPy)")
n_bags = 200
bags = [np.random.rand(np.random.randint(20, 100), 512).astype(np.float32)
        for _ in range(n_bags)]

# Rust batch processing
start = time.time()
rust_result = med_core_rs.batch_mil_aggregation(bags, method="max")
rust_time = time.time() - start

# NumPy processing
def numpy_batch_max(bags):
    return np.array([bag.max(axis=0) for bag in bags])

start = time.time()
numpy_result = numpy_batch_max(bags)
numpy_time = time.time() - start

speedup = numpy_time / rust_time
print(f"Rust 批量处理: {rust_time*1000:.2f} ms ({n_bags/rust_time:.1f} bags/秒)")
print(f"NumPy 批量处理: {numpy_time*1000:.2f} ms ({n_bags/numpy_time:.1f} bags/秒)")
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
print(f"  - Max pooling (100 bags): {time_max*1000:.1f} ms")
print(f"  - Mean pooling (100 bags): {time_mean*1000:.1f} ms")
print(f"  - 吞吐量: {n_bags/rust_time:.1f} bags/秒")
print(f"  - 相比 NumPy 加速: {speedup:.2f}x")
