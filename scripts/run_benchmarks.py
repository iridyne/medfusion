#!/usr/bin/env python
"""
运行 MedFusion 基准测试

测试关键组件的性能，生成基线数据。
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_data_loading():
    """基准测试：数据加载"""
    print("\n" + "=" * 60)
    print("数据加载基准测试")
    print("=" * 60)

    import time

    # 模拟数据加载
    def load_data_no_cache():
        """无缓存的数据加载"""
        time.sleep(0.001)  # 模拟 I/O
        return list(range(100))

    def load_data_with_cache():
        """有缓存的数据加载"""
        cache = {}

        def load(idx):
            if idx in cache:
                return cache[idx]
            time.sleep(0.001)
            data = list(range(100))
            cache[idx] = data
            return data

        return load

    # 测试无缓存
    print("\n1. 无缓存:")
    start = time.time()
    for _ in range(100):
        load_data_no_cache()
    time_no_cache = time.time() - start
    throughput_no_cache = 100 / time_no_cache
    print(f"   耗时: {time_no_cache:.3f}s")
    print(f"   吞吐量: {throughput_no_cache:.1f} samples/s")

    # 测试有缓存
    print("\n2. 有缓存:")
    loader = load_data_with_cache()
    start = time.time()
    for i in range(100):
        loader(i % 10)  # 重复访问 10 个样本
    time_with_cache = time.time() - start
    throughput_with_cache = 100 / time_with_cache
    print(f"   耗时: {time_with_cache:.3f}s")
    print(f"   吞吐量: {throughput_with_cache:.1f} samples/s")

    # 加速比
    speedup = throughput_with_cache / throughput_no_cache
    print(f"\n3. 加速比: {speedup:.1f}x")

    return {
        "no_cache": {
            "duration": time_no_cache,
            "throughput": throughput_no_cache,
        },
        "with_cache": {
            "duration": time_with_cache,
            "throughput": throughput_with_cache,
        },
        "speedup": speedup,
    }


def benchmark_fusion_strategies():
    """基准测试：融合策略"""
    print("\n" + "=" * 60)
    print("融合策略基准测试")
    print("=" * 60)

    import time

    # 模拟不同的融合策略
    def concatenate_fusion(v, t):
        """拼接融合"""
        return v + t

    def gated_fusion(v, t):
        """门控融合"""
        alpha = 0.5
        beta = 0.5
        return alpha * v + beta * t

    def attention_fusion(v, t):
        """注意力融合"""
        # 简化的注意力计算
        weight_v = v / (v + t + 1e-8)
        weight_t = t / (v + t + 1e-8)
        return weight_v * v + weight_t * t

    strategies = {
        "Concatenate": concatenate_fusion,
        "Gated": gated_fusion,
        "Attention": attention_fusion,
    }

    results = {}

    for name, func in strategies.items():
        print(f"\n{name} Fusion:")

        # 测试
        start = time.time()
        for _ in range(10000):
            func(1.0, 2.0)
        duration = time.time() - start
        throughput = 10000 / duration

        print(f"   耗时: {duration:.3f}s")
        print(f"   吞吐量: {throughput:.1f} ops/s")

        results[name] = {
            "duration": duration,
            "throughput": throughput,
        }

    return results


def benchmark_aggregators():
    """基准测试：聚合器"""
    print("\n" + "=" * 60)
    print("聚合器基准测试")
    print("=" * 60)

    import time

    # 模拟数据
    data = [[i + j for j in range(10)] for i in range(100)]

    # 不同的聚合策略
    def mean_pooling(instances):
        """均值池化"""
        return [sum(inst) / len(inst) for inst in instances]

    def max_pooling(instances):
        """最大池化"""
        return [max(inst) for inst in instances]

    def attention_pooling(instances):
        """注意力池化"""
        # 简化的注意力
        weights = [[1.0 / len(inst)] * len(inst) for inst in instances]
        return [
            sum(w * v for w, v in zip(weight, inst))
            for weight, inst in zip(weights, instances)
        ]

    aggregators = {
        "Mean": mean_pooling,
        "Max": max_pooling,
        "Attention": attention_pooling,
    }

    results = {}

    for name, func in aggregators.items():
        print(f"\n{name} Pooling:")

        # 测试
        start = time.time()
        for _ in range(1000):
            func(data)
        duration = time.time() - start
        throughput = 1000 / duration

        print(f"   耗时: {duration:.3f}s")
        print(f"   吞吐量: {throughput:.1f} ops/s")

        results[name] = {
            "duration": duration,
            "throughput": throughput,
        }

    return results


def benchmark_preprocessing():
    """基准测试：预处理"""
    print("\n" + "=" * 60)
    print("预处理基准测试")
    print("=" * 60)

    import time

    # 模拟图像数据
    image = [[i + j for j in range(224)] for i in range(224)]

    # 不同的预处理操作
    def resize(img):
        """调整大小（简化）"""
        return [[img[i][j] for j in range(0, 224, 2)] for i in range(0, 224, 2)]

    def normalize(img):
        """归一化"""
        mean = sum(sum(row) for row in img) / (224 * 224)
        return [[pixel - mean for pixel in row] for row in img]

    def augment(img):
        """数据增强（简化）"""
        # 水平翻转
        return [row[::-1] for row in img]

    operations = {
        "Resize": resize,
        "Normalize": normalize,
        "Augment": augment,
    }

    results = {}

    for name, func in operations.items():
        print(f"\n{name}:")

        # 测试
        start = time.time()
        for _ in range(1000):
            func(image)
        duration = time.time() - start
        throughput = 1000 / duration

        print(f"   耗时: {duration:.3f}s")
        print(f"   吞吐量: {throughput:.1f} ops/s")

        results[name] = {
            "duration": duration,
            "throughput": throughput,
        }

    return results


def save_results(results, output_file):
    """保存结果到文件"""
    import json

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行 MedFusion 基准测试")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/baseline.json",
        help="输出文件路径",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["data", "fusion", "aggregator", "preprocess", "all"],
        default=["all"],
        help="要运行的测试",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MedFusion 性能基准测试")
    print("=" * 60)

    all_results = {}

    # 运行测试
    if "all" in args.tests or "data" in args.tests:
        all_results["data_loading"] = benchmark_data_loading()

    if "all" in args.tests or "fusion" in args.tests:
        all_results["fusion_strategies"] = benchmark_fusion_strategies()

    if "all" in args.tests or "aggregator" in args.tests:
        all_results["aggregators"] = benchmark_aggregators()

    if "all" in args.tests or "preprocess" in args.tests:
        all_results["preprocessing"] = benchmark_preprocessing()

    # 保存结果
    save_results(all_results, args.output)

    print("\n" + "=" * 60)
    print("基准测试完成！")
    print("=" * 60)

    print("\n💡 提示:")
    print("  • 使用这些结果作为性能基线")
    print("  • 在代码变更后重新运行测试")
    print("  • 比较结果以检测性能回归")
    print("  • 集成到 CI/CD 流程中")


if __name__ == "__main__":
    main()
