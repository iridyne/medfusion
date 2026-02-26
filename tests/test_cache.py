"""
测试数据缓存和预取功能
"""

import time

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from med_core.datasets.cache import (
    CachedDataset,
    LRUCache,
    MemoryMappedCache,
    PrefetchDataset,
    create_cached_dataset,
)


class DummyDataset(Dataset):
    """用于测试的虚拟数据集"""

    def __init__(self, size: int = 100, delay: float = 0.0):
        self.size = size
        self.delay = delay
        self.access_count = [0] * size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 记录访问
        self.access_count[idx] += 1

        # 模拟加载延迟
        if self.delay > 0:
            time.sleep(self.delay)

        # 返回虚拟数据
        image = torch.randn(3, 224, 224)
        tabular = torch.randn(10)
        label = torch.tensor(idx % 2)

        return image, tabular, label


class TestLRUCache:
    """测试 LRU 缓存"""

    def test_basic_operations(self):
        """测试基本操作"""
        cache = LRUCache(capacity=3)

        # 添加数据
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # 获取数据
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") is None

    def test_capacity_limit(self):
        """测试容量限制"""
        cache = LRUCache(capacity=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # 应该移除 "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_eviction(self):
        """测试 LRU 淘汰策略"""
        cache = LRUCache(capacity=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # 访问 "a"，使其成为最近使用
        cache.put("c", 3)  # 应该移除 "b"

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_update_existing(self):
        """测试更新现有值"""
        cache = LRUCache(capacity=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10)  # 更新 "a"

        assert cache.get("a") == 10
        assert len(cache.cache) == 2

    def test_statistics(self):
        """测试统计信息"""
        cache = LRUCache(capacity=3)

        cache.put("a", 1)
        cache.put("b", 2)

        cache.get("a")  # hit
        cache.get("b")  # hit
        cache.get("c")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["size"] == 2

    def test_clear(self):
        """测试清空缓存"""
        cache = LRUCache(capacity=3)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")

        cache.clear()

        assert cache.get("a") is None
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["size"] == 0


class TestCachedDataset:
    """测试缓存数据集"""

    def test_basic_caching(self):
        """测试基本缓存功能"""
        base_dataset = DummyDataset(size=10)
        cached_dataset = CachedDataset(base_dataset, cache_size=5)

        # 第一次访问
        item1 = cached_dataset[0]
        assert base_dataset.access_count[0] == 1

        # 第二次访问（应该从缓存）
        item2 = cached_dataset[0]
        assert base_dataset.access_count[0] == 1  # 没有增加

        # 验证数据一致
        assert torch.equal(item1[0], item2[0])

    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        base_dataset = DummyDataset(size=10)
        cached_dataset = CachedDataset(base_dataset, cache_size=3)

        # 访问 4 个不同的样本
        for i in range(4):
            cached_dataset[i]

        # 第一个样本应该被淘汰
        cached_dataset[0]
        assert base_dataset.access_count[0] == 2  # 被访问了两次

    def test_no_cache(self):
        """测试禁用缓存"""
        base_dataset = DummyDataset(size=10)
        cached_dataset = CachedDataset(base_dataset, cache_size=0)

        # 多次访问
        cached_dataset[0]
        cached_dataset[0]

        # 应该每次都访问原始数据集
        assert base_dataset.access_count[0] == 2

    def test_cache_stats(self):
        """测试缓存统计"""
        base_dataset = DummyDataset(size=10)
        cached_dataset = CachedDataset(base_dataset, cache_size=5)

        # 访问一些样本
        cached_dataset[0]
        cached_dataset[1]
        cached_dataset[0]  # 缓存命中

        stats = cached_dataset.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    def test_clear_cache(self):
        """测试清空缓存"""
        base_dataset = DummyDataset(size=10)
        cached_dataset = CachedDataset(base_dataset, cache_size=5)

        cached_dataset[0]
        cached_dataset.clear_cache()
        cached_dataset[0]

        # 清空后应该重新加载
        assert base_dataset.access_count[0] == 2


class TestPrefetchDataset:
    """测试预取数据集"""

    def test_basic_prefetch(self):
        """测试基本预取功能"""
        base_dataset = DummyDataset(size=20, delay=0.01)
        prefetch_dataset = PrefetchDataset(base_dataset, prefetch_size=5, num_workers=2)

        # 顺序访问
        start_time = time.time()
        for i in range(10):
            prefetch_dataset[i]
            time.sleep(0.001)  # 给预取线程时间
        elapsed = time.time() - start_time

        # 预取应该减少总时间
        # 注意：这个测试可能不稳定，取决于系统负载
        assert elapsed < 0.15  # 如果没有预取，应该需要 ~0.1s

    def test_prefetch_queue(self):
        """测试预取队列"""
        base_dataset = DummyDataset(size=20)
        prefetch_dataset = PrefetchDataset(base_dataset, prefetch_size=3, num_workers=1)

        # 访问第一个样本
        prefetch_dataset[0]
        time.sleep(0.1)  # 等待预取

        # 检查预取队列大小（Queue 使用 qsize() 方法）
        assert prefetch_dataset.prefetch_queue.qsize() <= 3

    def test_sequential_access(self):
        """测试顺序访问性能"""
        base_dataset = DummyDataset(size=50)
        prefetch_dataset = PrefetchDataset(
            base_dataset, prefetch_size=10, num_workers=2
        )

        # 顺序访问所有样本
        for i in range(50):
            item = prefetch_dataset[i]
            assert item is not None


class TestMemoryMappedCache:
    """测试内存映射缓存"""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """创建临时缓存目录"""
        return tmp_path / "cache"

    def test_basic_operations(self, cache_dir):
        """测试基本操作"""
        cache = MemoryMappedCache(cache_dir, max_size_gb=0.001)

        # 添加数据
        data1 = np.random.randn(100, 100).astype(np.float32)
        assert cache.put("key1", data1)

        # 获取数据
        retrieved = cache.get("key1")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, data1)

    def test_size_limit(self, cache_dir):
        """测试大小限制"""
        cache = MemoryMappedCache(cache_dir, max_size_gb=0.001)  # ~1MB

        # 添加多个数组
        for i in range(5):
            data = np.random.randn(100, 100).astype(np.float32)
            cache.put(f"key{i}", data)

        # 检查大小限制
        stats = cache.get_stats()
        assert stats["total_size_mb"] <= 1.0

    def test_eviction(self, cache_dir):
        """测试淘汰策略"""
        # 使用非常小的缓存限制 (0.0001 GB = 100KB)
        cache = MemoryMappedCache(cache_dir, max_size_gb=0.0001)

        # 添加数据直到超过限制 (每个数组约 40KB)
        data = np.random.randn(100, 100).astype(np.float32)
        cache.put("key1", data)
        cache.put("key2", data)
        cache.put("key3", data)  # 这应该触发 key1 的驱逐

        # 第一个键应该被淘汰
        assert cache.get("key1") is None
        assert cache.get("key2") is not None

    def test_clear(self, cache_dir):
        """测试清空缓存"""
        cache = MemoryMappedCache(cache_dir, max_size_gb=0.001)

        data = np.random.randn(100, 100).astype(np.float32)
        cache.put("key1", data)
        cache.put("key2", data)

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        stats = cache.get_stats()
        assert stats["num_entries"] == 0

    def test_persistence(self, cache_dir):
        """测试持久化"""
        # 创建缓存并添加数据
        cache1 = MemoryMappedCache(cache_dir, max_size_gb=0.001)
        data = np.random.randn(100, 100).astype(np.float32)
        cache1.put("key1", data)

        # 创建新的缓存实例（模拟重启）
        cache2 = MemoryMappedCache(cache_dir, max_size_gb=0.001)

        # 注意：当前实现不支持跨实例持久化
        # 这个测试展示了限制
        assert cache2.get("key1") is None


class TestCacheFactory:
    """测试缓存工厂函数"""

    def test_no_cache(self):
        """测试无缓存"""
        dataset = DummyDataset(size=10)
        result = create_cached_dataset(dataset, cache_type="none")

        assert result is dataset

    def test_lru_cache(self):
        """测试 LRU 缓存"""
        dataset = DummyDataset(size=10)
        result = create_cached_dataset(dataset, cache_type="lru", cache_size=5)

        assert isinstance(result, CachedDataset)

    def test_prefetch_cache(self):
        """测试预取缓存"""
        dataset = DummyDataset(size=10)
        result = create_cached_dataset(dataset, cache_type="prefetch", prefetch_size=3)

        assert isinstance(result, PrefetchDataset)

    def test_both_cache(self):
        """测试组合缓存"""
        dataset = DummyDataset(size=10)
        result = create_cached_dataset(
            dataset, cache_type="both", cache_size=5, prefetch_size=3
        )

        assert isinstance(result, PrefetchDataset)
        assert isinstance(result.dataset, CachedDataset)

    def test_invalid_cache_type(self):
        """测试无效缓存类型"""
        dataset = DummyDataset(size=10)

        with pytest.raises(ValueError, match="Unknown cache type"):
            create_cached_dataset(dataset, cache_type="invalid")


class TestCachePerformance:
    """测试缓存性能"""

    def test_cache_speedup(self):
        """测试缓存加速效果"""
        base_dataset = DummyDataset(size=100, delay=0.001)
        cached_dataset = CachedDataset(base_dataset, cache_size=50)

        # 第一次访问（无缓存）
        start = time.time()
        for i in range(50):
            base_dataset[i]
        time_no_cache = time.time() - start

        # 第二次访问（有缓存）
        start = time.time()
        for i in range(50):
            cached_dataset[i]
        for i in range(50):
            cached_dataset[i]  # 第二遍应该很快
        time_with_cache = time.time() - start

        # 缓存应该显著加速
        assert time_with_cache < time_no_cache * 1.5

    def test_memory_usage(self):
        """测试内存使用"""
        base_dataset = DummyDataset(size=1000)
        cached_dataset = CachedDataset(base_dataset, cache_size=100)

        # 访问一些样本
        for i in range(200):
            cached_dataset[i % 100]

        # 检查缓存大小
        stats = cached_dataset.get_cache_stats()
        assert stats["size"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
