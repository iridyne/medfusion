"""
数据缓存功能演示

展示如何使用缓存来加速数据加载。
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from med_core.datasets.cache import (
    LRUCache,
    CachedDataset,
    PrefetchDataset,
    create_cached_dataset,
)


def demo_lru_cache():
    """演示 LRU 缓存"""
    print("=" * 60)
    print("LRU 缓存演示")
    print("=" * 60)
    
    cache = LRUCache(capacity=3)
    
    # 添加数据
    print("\n1. 添加数据到缓存:")
    cache.put("image_001", "data_001")
    cache.put("image_002", "data_002")
    cache.put("image_003", "data_003")
    print(f"   缓存大小: {len(cache.cache)}")
    
    # 获取数据
    print("\n2. 从缓存获取数据:")
    print(f"   image_001: {cache.get('image_001')}")
    print(f"   image_002: {cache.get('image_002')}")
    print(f"   image_999: {cache.get('image_999')}")
    
    # 添加新数据（触发淘汰）
    print("\n3. 添加新数据（超过容量）:")
    cache.put("image_004", "data_004")
    print(f"   缓存大小: {len(cache.cache)}")
    print(f"   image_003 是否还在: {cache.get('image_003') is not None}")
    
    # 统计信息
    print("\n4. 缓存统计:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


def demo_cache_benefits():
    """演示缓存的性能优势"""
    print("\n" + "=" * 60)
    print("缓存性能优势演示")
    print("=" * 60)
    
    # 模拟数据加载函数
    def slow_load_data(idx):
        """模拟慢速数据加载"""
        time.sleep(0.01)  # 模拟 I/O 延迟
        return f"data_{idx}"
    
    # 无缓存
    print("\n1. 无缓存 - 重复加载 100 次:")
    start = time.time()
    for _ in range(100):
        data = slow_load_data(0)
    time_no_cache = time.time() - start
    print(f"   耗时: {time_no_cache:.3f} 秒")
    
    # 有缓存
    print("\n2. 有缓存 - 重复加载 100 次:")
    cache = LRUCache(capacity=10)
    start = time.time()
    for _ in range(100):
        data = cache.get(0)
        if data is None:
            data = slow_load_data(0)
            cache.put(0, data)
    time_with_cache = time.time() - start
    print(f"   耗时: {time_with_cache:.3f} 秒")
    
    # 加速比
    speedup = time_no_cache / time_with_cache
    print(f"\n3. 加速比: {speedup:.1f}x")


def demo_cache_strategies():
    """演示不同的缓存策略"""
    print("\n" + "=" * 60)
    print("缓存策略对比")
    print("=" * 60)
    
    strategies = {
        "LRU": "最近最少使用 - 适合随机访问",
        "Prefetch": "预取 - 适合顺序访问",
        "Both": "LRU + 预取 - 适合混合访问",
    }
    
    print("\n支持的缓存策略:")
    for name, desc in strategies.items():
        print(f"  • {name:10s}: {desc}")
    
    print("\n使用建议:")
    print("  • 小数据集 + 随机访问 → LRU 缓存")
    print("  • 大数据集 + 顺序访问 → 预取缓存")
    print("  • 混合访问模式 → LRU + 预取")
    print("  • 内存受限 → 内存映射缓存")


def demo_usage_example():
    """演示实际使用示例"""
    print("\n" + "=" * 60)
    print("实际使用示例")
    print("=" * 60)
    
    code = '''
# 1. 创建原始数据集
from med_core.datasets import MedicalDataset

dataset = MedicalDataset(
    csv_path="data.csv",
    image_dir="images/",
    transform=transforms
)

# 2. 添加缓存（方法 1：使用工厂函数）
from med_core.datasets.cache import create_cached_dataset

cached_dataset = create_cached_dataset(
    dataset,
    cache_type="both",      # LRU + 预取
    cache_size=1000,        # 缓存 1000 个样本
    prefetch_size=10        # 预取 10 个样本
)

# 3. 创建 DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    cached_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 4. 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练代码
        pass
    
    # 查看缓存统计
    if hasattr(cached_dataset, 'get_cache_stats'):
        stats = cached_dataset.get_cache_stats()
        print(f"缓存命中率: {stats['hit_rate']:.2%}")
'''
    
    print("\n代码示例:")
    print(code)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MedFusion 数据缓存功能演示")
    print("=" * 60)
    
    try:
        # 演示 1: LRU 缓存基础
        demo_lru_cache()
        
        # 演示 2: 缓存性能优势
        demo_cache_benefits()
        
        # 演示 3: 缓存策略对比
        demo_cache_strategies()
        
        # 演示 4: 实际使用示例
        demo_usage_example()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n关键要点:")
        print("  ✓ LRU 缓存可以显著减少重复数据加载")
        print("  ✓ 预取可以隐藏 I/O 延迟")
        print("  ✓ 根据访问模式选择合适的缓存策略")
        print("  ✓ 监控缓存命中率来调优参数")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
