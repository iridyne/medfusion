"""
资源监控器

监控系统资源使用情况（GPU、内存、CPU），用于工作流执行期间的资源管理。
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# 尝试导入 GPU 监控库
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU monitoring disabled")


@dataclass
class ResourceSnapshot:
    """资源快照"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceThresholds:
    """资源阈值配置"""

    cpu_warning: float = 80.0  # CPU 使用率警告阈值（%）
    cpu_critical: float = 95.0  # CPU 使用率严重阈值（%）
    memory_warning: float = 80.0  # 内存使用率警告阈值（%）
    memory_critical: float = 95.0  # 内存使用率严重阈值（%）
    gpu_memory_warning: float = 90.0  # GPU 内存使用率警告阈值（%）
    gpu_memory_critical: float = 98.0  # GPU 内存使用率严重阈值（%）


class ResourceMonitor:
    """资源监控器"""

    def __init__(
        self,
        interval: float = 1.0,
        history_size: int = 300,
        thresholds: Optional[ResourceThresholds] = None,
    ):
        """
        初始化资源监控器

        Args:
            interval: 监控间隔（秒）
            history_size: 历史记录大小
            thresholds: 资源阈值配置
        """
        self.interval = interval
        self.history_size = history_size
        self.thresholds = thresholds or ResourceThresholds()

        # 历史记录（使用 deque 实现固定大小的环形缓冲区）
        self.history: Deque[ResourceSnapshot] = deque(maxlen=history_size)

        # 监控状态
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # GPU 初始化
        self.gpu_available = False
        self.gpu_count = 0
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
                logger.info(f"GPU monitoring enabled: {self.gpu_count} GPU(s) detected")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")

    def __del__(self):
        """清理资源"""
        if self.gpu_available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    async def start(self) -> None:
        """启动监控"""
        if self.monitoring:
            logger.warning("Resource monitor already running")
            return

        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitor started")

    async def stop(self) -> None:
        """停止监控"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitor stopped")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                snapshot = self._capture_snapshot()
                self.history.append(snapshot)

                # 检查阈值
                self._check_thresholds(snapshot)

                await asyncio.sleep(self.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.interval)

    def _capture_snapshot(self) -> ResourceSnapshot:
        """捕获资源快照"""
        # CPU 和内存
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # GPU
        gpu_info = []
        if self.gpu_available:
            gpu_info = self._get_gpu_info()

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_info=gpu_info,
        )

    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """获取 GPU 信息"""
        gpu_info = []

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU 名称
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # 内存信息
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_gb = memory.used / (1024**3)
                memory_total_gb = memory.total / (1024**3)
                memory_percent = (memory.used / memory.total) * 100

                # GPU 利用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = utilization.gpu

                # 温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temperature = None

                # 功率
                try:
                    power_usage = (
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )  # mW to W
                    power_limit = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    )
                except Exception:
                    power_usage = None
                    power_limit = None

                gpu_info.append(
                    {
                        "index": i,
                        "name": name,
                        "gpu_percent": gpu_percent,
                        "memory_used_gb": memory_used_gb,
                        "memory_total_gb": memory_total_gb,
                        "memory_percent": memory_percent,
                        "temperature": temperature,
                        "power_usage": power_usage,
                        "power_limit": power_limit,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")

        return gpu_info

    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """检查资源阈值并记录警告"""
        # CPU 检查
        if snapshot.cpu_percent >= self.thresholds.cpu_critical:
            logger.critical(f"CPU usage critical: {snapshot.cpu_percent:.1f}%")
        elif snapshot.cpu_percent >= self.thresholds.cpu_warning:
            logger.warning(f"CPU usage high: {snapshot.cpu_percent:.1f}%")

        # 内存检查
        if snapshot.memory_percent >= self.thresholds.memory_critical:
            logger.critical(
                f"Memory usage critical: {snapshot.memory_percent:.1f}% "
                f"({snapshot.memory_used_gb:.1f}/{snapshot.memory_total_gb:.1f} GB)"
            )
        elif snapshot.memory_percent >= self.thresholds.memory_warning:
            logger.warning(
                f"Memory usage high: {snapshot.memory_percent:.1f}% "
                f"({snapshot.memory_used_gb:.1f}/{snapshot.memory_total_gb:.1f} GB)"
            )

        # GPU 检查
        for gpu in snapshot.gpu_info:
            gpu_idx = gpu["index"]
            memory_percent = gpu["memory_percent"]

            if memory_percent >= self.thresholds.gpu_memory_critical:
                logger.critical(
                    f"GPU {gpu_idx} memory critical: {memory_percent:.1f}% "
                    f"({gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB)"
                )
            elif memory_percent >= self.thresholds.gpu_memory_warning:
                logger.warning(
                    f"GPU {gpu_idx} memory high: {memory_percent:.1f}% "
                    f"({gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB)"
                )

    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前资源状态

        Returns:
            包含 CPU、内存、GPU 信息的字典
        """
        if not self.history:
            snapshot = self._capture_snapshot()
        else:
            snapshot = self.history[-1]

        return {
            "timestamp": datetime.fromtimestamp(snapshot.timestamp).isoformat(),
            "cpu": {
                "percent": snapshot.cpu_percent,
                "warning": snapshot.cpu_percent >= self.thresholds.cpu_warning,
                "critical": snapshot.cpu_percent >= self.thresholds.cpu_critical,
            },
            "memory": {
                "percent": snapshot.memory_percent,
                "used_gb": snapshot.memory_used_gb,
                "total_gb": snapshot.memory_total_gb,
                "warning": snapshot.memory_percent >= self.thresholds.memory_warning,
                "critical": snapshot.memory_percent >= self.thresholds.memory_critical,
            },
            "gpu": [
                {
                    "index": gpu["index"],
                    "name": gpu["name"],
                    "gpu_percent": gpu["gpu_percent"],
                    "memory_percent": gpu["memory_percent"],
                    "memory_used_gb": gpu["memory_used_gb"],
                    "memory_total_gb": gpu["memory_total_gb"],
                    "temperature": gpu["temperature"],
                    "power_usage": gpu["power_usage"],
                    "power_limit": gpu["power_limit"],
                    "warning": gpu["memory_percent"]
                    >= self.thresholds.gpu_memory_warning,
                    "critical": gpu["memory_percent"]
                    >= self.thresholds.gpu_memory_critical,
                }
                for gpu in snapshot.gpu_info
            ],
            "monitoring": self.monitoring,
        }

    def get_history(self, duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        获取历史记录

        Args:
            duration: 可选的时间范围（秒），如果不提供则返回所有历史

        Returns:
            历史快照列表
        """
        if not self.history:
            return []

        if duration is None:
            snapshots = list(self.history)
        else:
            cutoff_time = time.time() - duration
            snapshots = [s for s in self.history if s.timestamp >= cutoff_time]

        return [
            {
                "timestamp": datetime.fromtimestamp(s.timestamp).isoformat(),
                "cpu_percent": s.cpu_percent,
                "memory_percent": s.memory_percent,
                "memory_used_gb": s.memory_used_gb,
                "gpu_info": s.gpu_info,
            }
            for s in snapshots
        ]

    def get_statistics(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        获取统计信息

        Args:
            duration: 可选的时间范围（秒）

        Returns:
            统计信息字典
        """
        history = self.get_history(duration)

        if not history:
            return {}

        cpu_values = [h["cpu_percent"] for h in history]
        memory_values = [h["memory_percent"] for h in history]

        stats = {
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
        }

        # GPU 统计
        if history[0]["gpu_info"]:
            gpu_count = len(history[0]["gpu_info"])
            stats["gpu"] = []

            for gpu_idx in range(gpu_count):
                gpu_memory_values = [
                    h["gpu_info"][gpu_idx]["memory_percent"]
                    for h in history
                    if len(h["gpu_info"]) > gpu_idx
                ]

                if gpu_memory_values:
                    stats["gpu"].append(
                        {
                            "index": gpu_idx,
                            "memory_avg": sum(gpu_memory_values)
                            / len(gpu_memory_values),
                            "memory_max": max(gpu_memory_values),
                            "memory_min": min(gpu_memory_values),
                        }
                    )

        return stats

    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()
        logger.info("Resource monitor history cleared")
