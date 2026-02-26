"""
数据缓存和预取优化模块

提供多种缓存策略来加速数据加载：
- LRU 缓存：最近最少使用缓存
- 内存映射缓存：使用 mmap 减少内存占用
- 预取缓存：后台预加载数据
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class LRUCache:
    """
    LRU (Least Recently Used) 缓存实现

    使用有序字典实现 O(1) 的查找和更新。

    Args:
        capacity: 缓存容量（最大条目数）

    Example:
        >>> cache = LRUCache(capacity=100)
        >>> cache.put("key1", data)
        >>> value = cache.get("key1")
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[Any, Any] = OrderedDict()
        self.lock = threading.Lock()

        # 统计信息
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在返回 None
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # 移动到末尾（最近使用）
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """
        添加缓存值

        Args:
            key: 缓存键
            value: 缓存值
        """
        with self.lock:
            if key in self.cache:
                # 更新现有值
                self.cache.move_to_end(key)
            # 添加新值
            elif len(self.cache) >= self.capacity:
                # 移除最旧的条目
                self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含命中率、大小等信息的字典
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.capacity,
        }


class CachedDataset(Dataset):
    """
    带缓存的数据集包装器

    在原始数据集外包装一层 LRU 缓存，加速重复访问。

    Args:
        dataset: 原始数据集
        cache_size: 缓存大小（条目数）
        cache_images: 是否缓存图像
        cache_tabular: 是否缓存表格数据

    Example:
        >>> base_dataset = MedicalDataset(...)
        >>> cached_dataset = CachedDataset(base_dataset, cache_size=1000)
        >>> dataloader = DataLoader(cached_dataset, ...)
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_size: int = 1000,
        cache_images: bool = True,
        cache_tabular: bool = True,
    ) -> None:
        self.dataset = dataset
        self.cache_images = cache_images
        self.cache_tabular = cache_tabular

        # 创建缓存
        self.cache = LRUCache(capacity=cache_size) if cache_size > 0 else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项（带缓存）

        Args:
            idx: 数据索引

        Returns:
            (image, tabular, label) 元组
        """
        # 尝试从缓存获取
        if self.cache is not None:
            cached_item = self.cache.get(idx)
            if cached_item is not None:
                return cached_item

        # 从原始数据集加载
        item = self.dataset[idx]

        # 添加到缓存
        if self.cache is not None:
            self.cache.put(idx, item)

        return item

    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache is None:
            return {"enabled": False}

        stats = self.cache.get_stats()
        stats["enabled"] = True
        return stats

    def clear_cache(self) -> None:
        """清空缓存"""
        if self.cache is not None:
            self.cache.clear()


class PrefetchDataset(Dataset):
    """
    带预取的数据集包装器

    在后台线程中预加载即将访问的数据，减少等待时间。

    Args:
        dataset: 原始数据集
        prefetch_size: 预取队列大小
        num_workers: 预取线程数

    Example:
        >>> base_dataset = MedicalDataset(...)
        >>> prefetch_dataset = PrefetchDataset(base_dataset, prefetch_size=10)
        >>> dataloader = DataLoader(prefetch_dataset, ...)
    """

    def __init__(
        self,
        dataset: Dataset,
        prefetch_size: int = 10,
        num_workers: int = 2,
    ) -> None:
        self.dataset = dataset
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers

        # 预取缓存
        self.prefetch_cache: dict[int, Any] = {}
        self.prefetch_lock = threading.Lock()

        # 预取队列
        self.prefetch_queue: list[int] = []
        self.queue_lock = threading.Lock()

        # 工作线程
        self.workers: list[threading.Thread] = []
        self.stop_flag = threading.Event()

        # 启动工作线程
        self._start_workers()

    def _start_workers(self) -> None:
        """启动预取工作线程"""
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._prefetch_worker, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _prefetch_worker(self) -> None:
        """预取工作线程函数"""
        while not self.stop_flag.is_set():
            # 获取待预取的索引
            with self.queue_lock:
                if not self.prefetch_queue:
                    time.sleep(0.01)  # 短暂休眠
                    continue

                idx = self.prefetch_queue.pop(0)

            # 检查是否已缓存
            with self.prefetch_lock:
                if idx in self.prefetch_cache:
                    continue

            # 加载数据
            try:
                item = self.dataset[idx]

                # 添加到缓存
                with self.prefetch_lock:
                    # 限制缓存大小
                    if len(self.prefetch_cache) >= self.prefetch_size * 2:
                        # 移除最旧的条目
                        oldest_key = min(self.prefetch_cache.keys())
                        del self.prefetch_cache[oldest_key]

                    self.prefetch_cache[idx] = item
            except Exception:
                # 忽略预取错误
                pass

    def _schedule_prefetch(self, idx: int) -> None:
        """
        调度预取任务

        Args:
            idx: 当前索引
        """
        # 预取接下来的 N 个样本
        with self.queue_lock:
            for offset in range(1, self.prefetch_size + 1):
                next_idx = idx + offset
                if next_idx < len(self.dataset):
                    if next_idx not in self.prefetch_queue:
                        self.prefetch_queue.append(next_idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项（带预取）

        Args:
            idx: 数据索引

        Returns:
            (image, tabular, label) 元组
        """
        # 调度预取
        self._schedule_prefetch(idx)

        # 尝试从预取缓存获取
        with self.prefetch_lock:
            if idx in self.prefetch_cache:
                item = self.prefetch_cache.pop(idx)
                return item

        # 直接从数据集加载
        return self.dataset[idx]

    def __del__(self) -> None:
        """清理资源"""
        self.stop_flag.set()
        for worker in self.workers:
            worker.join(timeout=1.0)


class MemoryMappedCache:
    """
    内存映射缓存

    使用 numpy 的 memmap 将数据映射到磁盘，减少内存占用。
    适合大规模数据集。

    Args:
        cache_dir: 缓存目录
        max_size_gb: 最大缓存大小（GB）

    Example:
        >>> cache = MemoryMappedCache(cache_dir="./cache", max_size_gb=10)
        >>> cache.put("key1", data)
        >>> value = cache.get("key1")
    """

    def __init__(self, cache_dir: str | Path, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.current_size = 0

        # 元数据
        self.metadata: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用哈希避免文件名冲突
        key_hash = hash(key) % 10**8
        return self.cache_dir / f"cache_{key_hash}.npy"

    def get(self, key: str) -> np.ndarray | None:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存的 numpy 数组，如果不存在返回 None
        """
        with self.lock:
            if key not in self.metadata:
                return None

            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                # 缓存文件丢失，清理元数据
                del self.metadata[key]
                return None

            # 加载内存映射数组
            self.metadata[key]
            data = np.load(str(cache_path), mmap_mode="r")

            return data

    def put(self, key: str, value: np.ndarray) -> bool:
        """
        添加缓存值

        Args:
            key: 缓存键
            value: numpy 数组

        Returns:
            是否成功添加
        """
        with self.lock:
            # 检查大小限制
            value_size = value.nbytes
            if value_size > self.max_size_bytes:
                return False

            # 如果超过限制，清理旧缓存
            while self.current_size + value_size > self.max_size_bytes:
                if not self.metadata:
                    return False

                # 移除最旧的条目
                oldest_key = next(iter(self.metadata))
                self._remove(oldest_key)

            # 保存到磁盘
            cache_path = self._get_cache_path(key)
            np.save(str(cache_path), value)

            # 更新元数据
            self.metadata[key] = {
                "size": value_size,
                "shape": value.shape,
                "dtype": str(value.dtype),
            }
            self.current_size += value_size

            return True

    def _remove(self, key: str) -> None:
        """移除缓存条目（内部方法，需要持有锁）"""
        if key not in self.metadata:
            return

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

        self.current_size -= self.metadata[key]["size"]
        del self.metadata[key]

    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            for cache_file in self.cache_dir.glob("cache_*.npy"):
                cache_file.unlink()

            self.metadata.clear()
            self.current_size = 0

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            return {
                "num_entries": len(self.metadata),
                "total_size_mb": self.current_size / (1024**2),
                "max_size_mb": self.max_size_bytes / (1024**2),
                "usage_percent": (self.current_size / self.max_size_bytes) * 100,
            }


def create_cached_dataset(
    dataset: Dataset,
    cache_type: str = "lru",
    cache_size: int = 1000,
    prefetch_size: int = 10,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """
    创建带缓存的数据集

    工厂函数，根据指定类型创建缓存数据集。

    Args:
        dataset: 原始数据集
        cache_type: 缓存类型 ("lru", "prefetch", "both", "none")
        cache_size: LRU 缓存大小
        prefetch_size: 预取队列大小
        cache_dir: 内存映射缓存目录（可选）

    Returns:
        包装后的数据集

    Example:
        >>> dataset = MedicalDataset(...)
        >>> cached = create_cached_dataset(dataset, cache_type="both", cache_size=500)
    """
    if cache_type == "none":
        return dataset

    if cache_type == "lru":
        return CachedDataset(dataset, cache_size=cache_size)

    if cache_type == "prefetch":
        return PrefetchDataset(dataset, prefetch_size=prefetch_size)

    if cache_type == "both":
        # 先缓存，再预取
        cached = CachedDataset(dataset, cache_size=cache_size)
        return PrefetchDataset(cached, prefetch_size=prefetch_size)

    raise ValueError(f"Unknown cache type: {cache_type}")
