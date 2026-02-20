"""
性能基准测试演示

展示如何使用基准测试工具来测量和比较性能。
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_simple_benchmark():
    """演示简单的基准测试"""
    print("=" * 60)
    print("简单基准测试演示")
    print("=" * 60)

    # 模拟两个不同的实现
    def slow_function():
        """慢速实现"""
        time.sleep(0.001)
        return sum(range(1000))

    def fast_function():
        """快速实现"""
        return sum(range(1000))

    # 测试慢速实现
    print("\n1. 测试慢速实现:")
    start = time.time()
    for _ in range(100):
        slow_function()
    slow_time = time.time() - start
    print(f"   耗时: {slow_time:.3f}s")
    print(f"   吞吐量: {100/slow_time:.1f} ops/s")

    # 测试快速实现
    print("\n2. 测试快速实现:")
    start = time.time()
    for _ in range(100):
        fast_function()
    fast_time = time.time() - start
    print(f"   耗时: {fast_time:.3f}s")
    print(f"   吞吐量: {100/fast_time:.1f} ops/s")

    # 比较
    speedup = slow_time / fast_time
    print(f"\n3. 加速比: {speedup:.1f}x")


def demo_benchmark_suite():
    """演示基准测试套件"""
    print("\n" + "=" * 60)
    print("基准测试套件演示")
    print("=" * 60)

    print("\n基准测试套件的功能:")
    print("  • 管理多个基准测试")
    print("  • 自动运行所有测试")
    print("  • 保存结果到 JSON")
    print("  • 与基线比较")
    print("  • 检测性能回归")

    print("\n使用示例:")
    code = '''
from med_core.utils.benchmark import BenchmarkSuite, PerformanceBenchmark

# 1. 创建测试套件
suite = BenchmarkSuite(name="v0.2.0")

# 2. 添加基准测试
def test_model_inference():
    benchmark = PerformanceBenchmark("model_inference")
    return benchmark.run(lambda: model(input))

suite.add_benchmark("model_inference", test_model_inference)

# 3. 运行所有测试
results = suite.run_all()

# 4. 保存结果
suite.save_results("baseline.json")

# 5. 与基线比较
suite.compare_with("baseline.json")
'''
    print(code)


def demo_performance_metrics():
    """演示性能指标"""
    print("\n" + "=" * 60)
    print("性能指标说明")
    print("=" * 60)

    metrics = {
        "Duration": "总执行时间（秒）",
        "Throughput": "吞吐量（样本/秒或操作/秒）",
        "Memory Allocated": "分配的内存（MB）",
        "Memory Reserved": "保留的内存（MB）",
        "Latency": "延迟（毫秒/样本）",
    }

    print("\n关键性能指标:")
    for metric, desc in metrics.items():
        print(f"  • {metric:20s}: {desc}")

    print("\n性能目标:")
    print("  • 数据加载: > 1000 samples/s")
    print("  • 模型推理: > 100 samples/s (CPU), > 1000 samples/s (GPU)")
    print("  • 内存使用: < 8GB (训练), < 2GB (推理)")


def demo_regression_testing():
    """演示回归测试"""
    print("\n" + "=" * 60)
    print("性能回归测试")
    print("=" * 60)

    print("\n什么是性能回归?")
    print("  代码变更导致性能下降超过可接受的阈值")

    print("\n如何检测回归?")
    print("  1. 建立性能基线（baseline）")
    print("  2. 每次代码变更后运行基准测试")
    print("  3. 比较当前结果与基线")
    print("  4. 如果性能下降 > 5%，标记为回归")

    print("\n示例:")
    print("  基线吞吐量: 1000 samples/s")
    print("  当前吞吐量: 900 samples/s")
    print("  变化: -10% ❌ 回归!")

    print("\n  基线吞吐量: 1000 samples/s")
    print("  当前吞吐量: 980 samples/s")
    print("  变化: -2% ✓ 正常")


def demo_optimization_workflow():
    """演示优化工作流"""
    print("\n" + "=" * 60)
    print("性能优化工作流")
    print("=" * 60)

    steps = [
        ("1. 建立基线", "运行基准测试，保存结果"),
        ("2. 识别瓶颈", "分析哪个部分最慢"),
        ("3. 实施优化", "修改代码提升性能"),
        ("4. 验证改进", "重新运行基准测试"),
        ("5. 比较结果", "确认性能提升"),
        ("6. 更新基线", "如果满意，更新基线"),
    ]

    print("\n优化步骤:")
    for step, desc in steps:
        print(f"  {step:20s} → {desc}")

    print("\n示例场景:")
    print("  问题: 数据加载太慢（100 samples/s）")
    print("  优化: 添加 LRU 缓存")
    print("  结果: 提升到 300 samples/s (3x)")
    print("  决策: ✓ 接受优化，更新基线")


def demo_best_practices():
    """演示最佳实践"""
    print("\n" + "=" * 60)
    print("基准测试最佳实践")
    print("=" * 60)

    print("\n✅ 推荐做法:")
    practices = [
        "预热（warmup）- 避免冷启动影响",
        "多次迭代 - 减少测量误差",
        "固定随机种子 - 确保可重复性",
        "隔离测试 - 避免相互干扰",
        "记录环境 - CPU/GPU 型号、驱动版本",
        "自动化 - 集成到 CI/CD",
    ]

    for practice in practices:
        print(f"  • {practice}")

    print("\n❌ 避免的做法:")
    antipatterns = [
        "在生产环境测试",
        "忽略预热阶段",
        "单次测量",
        "不记录环境信息",
        "手动运行测试",
    ]

    for antipattern in antipatterns:
        print(f"  • {antipattern}")


def demo_ci_integration():
    """演示 CI 集成"""
    print("\n" + "=" * 60)
    print("CI/CD 集成")
    print("=" * 60)

    print("\nGitHub Actions 示例:")
    yaml = '''
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e .

      - name: Run benchmarks
        run: python scripts/run_benchmarks.py

      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py \\
            --baseline benchmarks/baseline.json \\
            --current benchmarks/current.json \\
            --tolerance 0.05

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/
'''
    print(yaml)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MedFusion 性能基准测试演示")
    print("=" * 60)

    try:
        # 演示 1: 简单基准测试
        demo_simple_benchmark()

        # 演示 2: 基准测试套件
        demo_benchmark_suite()

        # 演示 3: 性能指标
        demo_performance_metrics()

        # 演示 4: 回归测试
        demo_regression_testing()

        # 演示 5: 优化工作流
        demo_optimization_workflow()

        # 演示 6: 最佳实践
        demo_best_practices()

        # 演示 7: CI 集成
        demo_ci_integration()

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("  1. 建立性能基线并定期更新")
        print("  2. 自动化基准测试，集成到 CI/CD")
        print("  3. 监控性能回归，及时发现问题")
        print("  4. 记录优化历史，追踪性能改进")

        print("\n📖 相关资源:")
        print("  • med_core/utils/benchmark.py - 基准测试工具")
        print("  • scripts/run_benchmarks.py - 运行脚本")
        print("  • benchmarks/ - 基准测试结果")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
