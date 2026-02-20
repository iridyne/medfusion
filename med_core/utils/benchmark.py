"""
性能基准测试模块

提供自动化的性能回归测试，确保代码变更不会降低性能。
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    
    name: str
    """测试名称"""
    
    duration: float
    """执行时间（秒）"""
    
    throughput: float
    """吞吐量（样本/秒 或 批次/秒）"""
    
    memory_allocated: float
    """分配的内存（MB）"""
    
    memory_reserved: float
    """保留的内存（MB）"""
    
    metadata: dict[str, Any]
    """额外的元数据"""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"{self.name}:\n"
            f"  Duration: {self.duration:.3f}s\n"
            f"  Throughput: {self.throughput:.1f} samples/s\n"
            f"  Memory: {self.memory_allocated:.1f}MB allocated, "
            f"{self.memory_reserved:.1f}MB reserved"
        )


class PerformanceBenchmark:
    """
    性能基准测试器
    
    用于测量和比较不同实现的性能。
    
    Args:
        name: 基准测试名称
        warmup_iterations: 预热迭代次数
        test_iterations: 测试迭代次数
        device: 测试设备
        
    Example:
        >>> benchmark = PerformanceBenchmark("model_inference")
        >>> result = benchmark.run(lambda: model(input))
        >>> print(result)
    """
    
    def __init__(
        self,
        name: str,
        warmup_iterations: int = 10,
        test_iterations: int = 100,
        device: str = "cpu",
    ):
        self.name = name
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.device = torch.device(device)
    
    def run(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            基准测试结果
        """
        # 预热
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # 清理缓存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 记录初始内存
        if self.device.type == "cuda":
            mem_before_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_before_reserved = torch.cuda.memory_reserved() / 1024**2
        else:
            mem_before_allocated = 0
            mem_before_reserved = 0
        
        # 测试
        start_time = time.time()
        for _ in range(self.test_iterations):
            func(*args, **kwargs)
        
        # 同步（GPU）
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 计算指标
        duration = end_time - start_time
        throughput = self.test_iterations / duration
        
        # 记录内存使用
        if self.device.type == "cuda":
            mem_after_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_after_reserved = torch.cuda.memory_reserved() / 1024**2
            memory_allocated = mem_after_allocated - mem_before_allocated
            memory_reserved = mem_after_reserved - mem_before_reserved
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return BenchmarkResult(
            name=self.name,
            duration=duration,
            throughput=throughput,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            metadata={
                "warmup_iterations": self.warmup_iterations,
                "test_iterations": self.test_iterations,
                "device": str(self.device),
            },
        )


class DataLoaderBenchmark:
    """
    数据加载器基准测试
    
    测量数据加载的性能。
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        device: 设备
        
    Example:
        >>> benchmark = DataLoaderBenchmark(dataset, batch_size=32)
        >>> result = benchmark.run(num_batches=100)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = "cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
    
    def run(self, num_batches: int = 100) -> BenchmarkResult:
        """
        运行数据加载基准测试
        
        Args:
            num_batches: 测试的批次数
            
        Returns:
            基准测试结果
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        
        # 预热
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
        
        # 测试
        start_time = time.time()
        samples_loaded = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # 移动到设备（模拟实际使用）
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            
            samples_loaded += len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = samples_loaded / duration
        
        return BenchmarkResult(
            name=f"DataLoader (bs={self.batch_size}, workers={self.num_workers})",
            duration=duration,
            throughput=throughput,
            memory_allocated=0,
            memory_reserved=0,
            metadata={
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "num_batches": num_batches,
                "samples_loaded": samples_loaded,
            },
        )


class ModelBenchmark:
    """
    模型推理基准测试
    
    测量模型前向传播的性能。
    
    Args:
        model: 模型
        input_shape: 输入形状
        device: 设备
        
    Example:
        >>> benchmark = ModelBenchmark(model, input_shape=(3, 224, 224))
        >>> result = benchmark.run(batch_size=32, num_iterations=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.input_shape = input_shape
        self.device = torch.device(device)
    
    def run(
        self,
        batch_size: int = 32,
        num_iterations: int = 100,
    ) -> BenchmarkResult:
        """
        运行模型推理基准测试
        
        Args:
            batch_size: 批次大小
            num_iterations: 迭代次数
            
        Returns:
            基准测试结果
        """
        # 创建虚拟输入
        dummy_input = torch.randn(batch_size, *self.input_shape).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy_input)
        
        # 清理缓存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 测试
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                self.model(dummy_input)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = (num_iterations * batch_size) / duration
        
        # 内存使用
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return BenchmarkResult(
            name=f"Model Inference (bs={batch_size})",
            duration=duration,
            throughput=throughput,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            metadata={
                "batch_size": batch_size,
                "num_iterations": num_iterations,
                "input_shape": self.input_shape,
            },
        )


class BenchmarkSuite:
    """
    基准测试套件
    
    管理多个基准测试，支持结果保存和比较。
    
    Args:
        name: 套件名称
        output_dir: 输出目录
        
    Example:
        >>> suite = BenchmarkSuite("v0.2.0")
        >>> suite.add_benchmark("test1", lambda: func1())
        >>> suite.add_benchmark("test2", lambda: func2())
        >>> suite.run_all()
        >>> suite.save_results()
    """
    
    def __init__(self, name: str, output_dir: str | Path = "./benchmarks"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarks: list[tuple[str, Callable]] = []
        self.results: list[BenchmarkResult] = []
    
    def add_benchmark(self, name: str, func: Callable) -> None:
        """
        添加基准测试
        
        Args:
            name: 测试名称
            func: 测试函数
        """
        self.benchmarks.append((name, func))
    
    def run_all(self) -> list[BenchmarkResult]:
        """
        运行所有基准测试

        Returns:
            所有测试结果
        """
        self.results = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Running Benchmark Suite: {self.name}")
        logger.info(f"{'='*60}\n")

        for name, func in self.benchmarks:
            logger.info(f"Running: {name}...")
            try:
                result = func()
                self.results.append(result)
                logger.info(f"  ✓ {result.throughput:.1f} samples/s\n")
            except Exception as e:
                logger.error(f"  ✗ Error: {e}\n")

        return self.results
    
    def save_results(self, filename: str | None = None) -> Path:
        """
        保存结果到 JSON 文件
        
        Args:
            filename: 文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"benchmark_{self.name}_{int(time.time())}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "suite_name": self.name,
            "timestamp": time.time(),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def compare_with(self, baseline_file: str | Path) -> dict[str, Any]:
        """
        与基线结果比较
        
        Args:
            baseline_file: 基线结果文件
            
        Returns:
            比较结果
        """
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        
        baseline_results = {r["name"]: r for r in baseline_data["results"]}

        comparisons = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing with baseline: {baseline_file}")
        logger.info(f"{'='*60}\n")

        for result in self.results:
            if result.name not in baseline_results:
                logger.warning(f"{result.name}: No baseline found")
                continue
            
            baseline = baseline_results[result.name]
            
            # 计算变化
            throughput_change = (
                (result.throughput - baseline["throughput"]) / baseline["throughput"]
            ) * 100
            
            memory_change = (
                (result.memory_allocated - baseline["memory_allocated"]) 
                / max(baseline["memory_allocated"], 1)
            ) * 100
            
            # 判断是否回归
            is_regression = throughput_change < -5  # 性能下降超过 5%
            
            comparison = {
                "name": result.name,
                "throughput_change": throughput_change,
                "memory_change": memory_change,
                "is_regression": is_regression,
            }
            
            comparisons.append(comparison)

            # 打印结果
            status = "❌ REGRESSION" if is_regression else "✓ OK"
            logger.info(f"{result.name}:")
            logger.info(f"  Throughput: {throughput_change:+.1f}% {status}")
            logger.info(f"  Memory: {memory_change:+.1f}%\n")

        return {
            "baseline": str(baseline_file),
            "comparisons": comparisons,
        }

    def print_summary(self) -> None:
        """打印结果摘要"""
        if not self.results:
            logger.info("No results to display")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark Summary: {self.name}")
        logger.info(f"{'='*60}\n")

        for result in self.results:
            logger.info(result)
            logger.info("")


def create_regression_test(
    baseline_file: str | Path,
    tolerance: float = 0.05,
) -> Callable:
    """
    创建性能回归测试
    
    Args:
        baseline_file: 基线结果文件
        tolerance: 容忍度（默认 5%）
        
    Returns:
        测试函数
        
    Example:
        >>> test_func = create_regression_test("baseline.json")
        >>> test_func(current_results)
    """
    def test(results: list[BenchmarkResult]) -> bool:
        """检查是否有性能回归"""
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        
        baseline_results = {r["name"]: r for r in baseline_data["results"]}
        
        has_regression = False
        
        for result in results:
            if result.name not in baseline_results:
                continue
            
            baseline = baseline_results[result.name]
            change = (result.throughput - baseline["throughput"]) / baseline["throughput"]

            if change < -tolerance:
                logger.error(f"❌ Regression detected in {result.name}: {change*100:.1f}%")
                has_regression = True
        
        return not has_regression
    
    return test
