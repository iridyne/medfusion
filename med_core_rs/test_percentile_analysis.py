#!/usr/bin/env python3
"""
测试 Quickselect vs 排序算法的性能差异
"""

import numpy as np
import time

def test_percentile_algorithms():
    """对比不同大小数据的 percentile 计算"""

    print("="*70)
    print("Percentile 算法性能对比")
    print("="*70)

    sizes = [256*256, 512*512, 1024*1024]

    for size in sizes:
        print(f"\n数据大小: {size:,} 个元素 ({int(np.sqrt(size))}×{int(np.sqrt(size))})")

        data = np.random.rand(size).astype(np.float32) * 255

        # NumPy percentile (使用优化的 C 实现)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            low = np.percentile(data, 1.0)
            high = np.percentile(data, 99.0)
            times.append(time.perf_counter() - start)

        numpy_time = np.mean(times)
        print(f"  NumPy percentile:  {numpy_time*1000:6.2f} ms")

        # 纯 Python 排序
        times = []
        for _ in range(50):
            start = time.perf_counter()
            sorted_data = np.sort(data)
            idx_low = int(0.01 * len(sorted_data))
            idx_high = int(0.99 * len(sorted_data))
            low = sorted_data[idx_low]
            high = sorted_data[idx_high]
            times.append(time.perf_counter() - start)

        sort_time = np.mean(times)
        print(f"  NumPy sort:        {sort_time*1000:6.2f} ms")
        print(f"  排序 vs percentile: {sort_time/numpy_time:.2f}x")

if __name__ == "__main__":
    test_percentile_algorithms()

    print("\n" + "="*70)
    print("结论:")
    print("="*70)
    print("NumPy 的 percentile 函数已经高度优化（使用 quickselect）")
    print("我们的 Rust 实现需要：")
    print("  1. 数据从 Python 传到 Rust（开销）")
    print("  2. 在 Rust 中计算 percentile")
    print("  3. 结果传回 Python（开销）")
    print("\n对于单图像，边界开销 > 计算加速")
    print("对于批量，并行处理 > 边界开销")
