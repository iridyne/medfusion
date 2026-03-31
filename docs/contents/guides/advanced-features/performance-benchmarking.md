# 性能基准测试

## 概述

性能基准测试是确保代码质量的重要环节。MedFusion 提供了完整的基准测试工具，用于：

- 测量关键组件的性能
- 建立性能基线
- 检测性能回归
- 追踪性能改进

## 快速开始

### 运行基准测试

```bash
# 运行所有基准测试
python scripts/run_benchmarks.py

# 运行特定测试
python scripts/run_benchmarks.py --tests data fusion

# 指定输出文件
python scripts/run_benchmarks.py --output benchmarks/v0.2.0.json
```

### 比较结果

```bash
# 与基线比较
python scripts/compare_benchmarks.py \
    --baseline benchmarks/baseline.json \
    --current benchmarks/current.json

# 在 CI 中使用（有回归时失败）
python scripts/compare_benchmarks.py \
    --baseline benchmarks/baseline.json \
    --current benchmarks/current.json \
    --fail-on-regression
```

## 基准测试工具

### 1. PerformanceBenchmark

通用的性能基准测试器。

**特点**:
- 自动预热
- 多次迭代
- 内存跟踪
- GPU 支持

**示例**:
```python
from med_core.utils.benchmark import PerformanceBenchmark

# 创建基准测试
benchmark = PerformanceBenchmark(
    name="my_function",
    warmup_iterations=10,
    test_iterations=100,
    device="cuda",
)

# 运行测试
result = benchmark.run(my_function, arg1, arg2)

# 查看结果
print(result)
# Output:
# my_function:
#   Duration: 0.123s
#   Throughput: 813.0 samples/s
#   Memory: 256.0MB allocated, 512.0MB reserved
```

### 2. DataLoaderBenchmark

数据加载性能测试。

**示例**:
```python
from med_core.utils.benchmark import DataLoaderBenchmark

benchmark = DataLoaderBenchmark(
    dataset=my_dataset,
    batch_size=32,
    num_workers=4,
)

result = benchmark.run(num_batches=100)
print(f"吞吐量: {result.throughput:.1f} samples/s")
```

### 3. ModelBenchmark

模型推理性能测试。

**示例**:
```python
from med_core.utils.benchmark import ModelBenchmark

benchmark = ModelBenchmark(
    model=my_model,
    input_shape=(3, 224, 224),
    device="cuda",
)

result = benchmark.run(batch_size=32, num_iterations=100)
print(f"吞吐量: {result.throughput:.1f} samples/s")
```

### 4. BenchmarkSuite

管理多个基准测试。

**示例**:
```python
from med_core.utils.benchmark import BenchmarkSuite

# 创建套件
suite = BenchmarkSuite(name="v0.2.0")

# 添加测试
suite.add_benchmark("test1", lambda: test_function1())
suite.add_benchmark("test2", lambda: test_function2())

# 运行所有测试
results = suite.run_all()

# 保存结果
suite.save_results("baseline.json")

# 与基线比较
suite.compare_with("previous_baseline.json")
```

## 测试类型

### 1. 数据加载测试

测试数据加载的性能，包括：
- 无缓存加载
- 有缓存加载
- 不同 num_workers 的影响

**当前基线** (v0.2.0):
- 无缓存: ~900 samples/s
- 有缓存: ~9,000 samples/s
- 加速比: 10x

### 2. 融合策略测试

测试不同融合策略的性能：
- Concatenate: ~22M ops/s
- Gated: ~10M ops/s
- Attention: ~8M ops/s

### 3. 聚合器测试

测试不同聚合方法的性能：
- Mean Pooling: ~100K ops/s
- Max Pooling: ~127K ops/s
- Attention Pooling: ~18K ops/s

### 4. 预处理测试

测试图像预处理操作：
- Resize: ~6,700 ops/s
- Normalize: ~950 ops/s
- Augment: ~9,100 ops/s

## 性能回归检测

### 什么是性能回归？

性能回归是指代码变更导致性能下降超过可接受的阈值（默认 5%）。

### 检测流程

1. **建立基线**
   ```bash
   python scripts/run_benchmarks.py --output benchmarks/baseline.json
   ```

2. **修改代码**
   ```bash
   # 进行代码修改
   git commit -m "Optimize data loading"
   ```

3. **运行新测试**
   ```bash
   python scripts/run_benchmarks.py --output benchmarks/current.json
   ```

4. **比较结果**
   ```bash
   python scripts/compare_benchmarks.py \
       --baseline benchmarks/baseline.json \
       --current benchmarks/current.json
   ```

### 解读结果

```
❌ 性能回归 (下降 > 5%):
  data_loading.throughput:
    基线: 1000.0
    当前: 900.0
    变化: -10.0%

✅ 性能改进 (提升 > 5%):
  fusion.throughput:
    基线: 1000.0
    当前: 1200.0
    变化: +20.0%

总结:
  回归: 1
  改进: 1
  稳定: 20
```

## CI/CD 集成

### GitHub Actions

在 `.github/workflows/benchmark.yml` 中添加：

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .
      
      - name: Run benchmarks
        run: |
          python scripts/run_benchmarks.py \
            --output benchmarks/current.json
      
      - name: Download baseline
        uses: actions/download-artifact@v3
        with:
          name: benchmark-baseline
          path: benchmarks/
        continue-on-error: true
      
      - name: Compare with baseline
        if: success()
        run: |
          if [ -f benchmarks/baseline.json ]; then
            python scripts/compare_benchmarks.py \
              --baseline benchmarks/baseline.json \
              --current benchmarks/current.json \
              --fail-on-regression
          else
            echo "No baseline found, skipping comparison"
          fi
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/current.json
      
      - name: Update baseline (on main)
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-baseline
          path: benchmarks/current.json
```

### GitLab CI

在 `.gitlab-ci.yml` 中添加：

```yaml
benchmark:
  stage: test
  script:
    - pip install -e .
    - python scripts/run_benchmarks.py --output benchmarks/current.json
    - |
      if [ -f benchmarks/baseline.json ]; then
        python scripts/compare_benchmarks.py \
          --baseline benchmarks/baseline.json \
          --current benchmarks/current.json \
          --fail-on-regression
      fi
  artifacts:
    paths:
      - benchmarks/
    expire_in: 30 days
```

## 性能优化工作流

### 1. 识别瓶颈

```python
# 使用 profiler 找到慢的部分
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 运行代码
train_one_epoch()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### 2. 实施优化

```python
# 优化前
def slow_function():
    result = []
    for i in range(1000):
        result.append(expensive_operation(i))
    return result

# 优化后
def fast_function():
    # 使用缓存
    cache = {}
    result = []
    for i in range(1000):
        if i not in cache:
            cache[i] = expensive_operation(i)
        result.append(cache[i])
    return result
```

### 3. 验证改进

```python
from med_core.utils.benchmark import PerformanceBenchmark

# 测试优化前
benchmark_slow = PerformanceBenchmark("slow")
result_slow = benchmark_slow.run(slow_function)

# 测试优化后
benchmark_fast = PerformanceBenchmark("fast")
result_fast = benchmark_fast.run(fast_function)

# 计算加速比
speedup = result_fast.throughput / result_slow.throughput
print(f"加速比: {speedup:.1f}x")
```

### 4. 更新基线

```bash
# 如果优化效果满意，更新基线
cp benchmarks/current.json benchmarks/baseline.json
git add benchmarks/baseline.json
git commit -m "Update performance baseline after optimization"
```

## 最佳实践

### ✅ 推荐做法

1. **定期运行基准测试**
   - 每次重要代码变更后
   - 每周/每月定期测试
   - 发布前必须测试

2. **使用固定的测试环境**
   - 相同的硬件
   - 相同的软件版本
   - 隔离的测试环境

3. **记录环境信息**
   ```python
   import platform
   import torch
   
   env_info = {
       "python": platform.python_version(),
       "torch": torch.__version__,
       "cuda": torch.version.cuda,
       "cpu": platform.processor(),
   }
   ```

4. **设置合理的阈值**
   - 5% 适合大多数情况
   - 关键路径可以设置更严格（2-3%）
   - 非关键路径可以放宽（10%）

5. **自动化测试**
   - 集成到 CI/CD
   - 自动生成报告
   - 自动通知回归

### ❌ 避免的做法

1. **不要在不稳定的环境测试**
   - 避免在笔记本电脑上测试
   - 避免在共享服务器上测试
   - 避免在负载高的时候测试

2. **不要忽略预热**
   - 第一次运行通常较慢
   - 需要预热 JIT、缓存等

3. **不要单次测量**
   - 单次测量误差大
   - 至少运行 10-100 次

4. **不要比较不同环境的结果**
   - CPU vs GPU
   - 不同的 Python 版本
   - 不同的库版本

## 故障排除

### 问题 1: 结果不稳定

**症状**: 每次运行结果差异很大

**原因**:
- 环境不稳定
- 迭代次数太少
- 没有预热

**解决方案**:
```python
# 增加预热和迭代次数
benchmark = PerformanceBenchmark(
    name="test",
    warmup_iterations=50,  # 增加预热
    test_iterations=500,   # 增加迭代
)

# 固定随机种子
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### 问题 2: GPU 内存不足

**症状**: CUDA out of memory

**原因**:
- 批次太大
- 模型太大
- 内存泄漏

**解决方案**:
```python
# 减小批次
benchmark.run(batch_size=16)  # 从 32 减到 16

# 清理缓存
torch.cuda.empty_cache()

# 使用梯度检查点
model.gradient_checkpointing_enable()
```

### 问题 3: 比较失败

**症状**: 无法找到基线文件

**原因**:
- 基线文件不存在
- 路径错误

**解决方案**:
```bash
# 检查文件是否存在
ls -la benchmarks/

# 创建基线
python scripts/run_benchmarks.py --output benchmarks/baseline.json

# 使用正确的路径
python scripts/compare_benchmarks.py \
    --baseline benchmarks/baseline.json \
    --current benchmarks/current.json
```

## 性能目标

### 数据加载

| 组件 | 目标 | 当前 | 状态 |
|------|------|------|------|
| 无缓存加载 | > 500 samples/s | 927 samples/s | ✅ |
| 有缓存加载 | > 5000 samples/s | 9451 samples/s | ✅ |
| 多进程加载 | > 2000 samples/s | TBD | 🔄 |

### 模型推理

| 模型 | 设备 | 目标 | 当前 | 状态 |
|------|------|------|------|------|
| ResNet50 | CPU | > 50 samples/s | TBD | 🔄 |
| ResNet50 | GPU | > 500 samples/s | TBD | 🔄 |
| ViT-B/16 | GPU | > 200 samples/s | TBD | 🔄 |

### 融合策略

| 策略 | 目标 | 当前 | 状态 |
|------|------|------|------|
| Concatenate | > 10M ops/s | 22M ops/s | ✅ |
| Gated | > 5M ops/s | 10M ops/s | ✅ |
| Attention | > 5M ops/s | 8M ops/s | ✅ |

## API 参考

### PerformanceBenchmark

```python
PerformanceBenchmark(
    name: str,
    warmup_iterations: int = 10,
    test_iterations: int = 100,
    device: str = "cpu",
)
```

**方法**:
- `run(func, *args, **kwargs)`: 运行基准测试

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    name: str
    duration: float
    throughput: float
    memory_allocated: float
    memory_reserved: float
    metadata: dict
```

**方法**:
- `to_dict()`: 转换为字典
- `__str__()`: 字符串表示

### BenchmarkSuite

```python
BenchmarkSuite(
    name: str,
    output_dir: str | Path = "./benchmarks",
)
```

**方法**:
- `add_benchmark(name, func)`: 添加测试
- `run_all()`: 运行所有测试
- `save_results(filename)`: 保存结果
- `compare_with(baseline_file)`: 与基线比较

## 相关资源

- **实现代码**: `med_core/utils/benchmark.py`
- **运行脚本**: `scripts/run_benchmarks.py`
- **比较脚本**: `scripts/compare_benchmarks.py`
- **演示脚本**: `scripts/dev/benchmark_demo.py`
- **基线数据**: `benchmarks/baseline.json`

## 更新日志

### v0.2.0 (2026-02-20)
- ✨ 新增性能基准测试工具
- ✨ 新增回归检测脚本
- ✨ 建立性能基线
- 📝 完整的文档和示例
- 🔧 CI/CD 集成指南
