"""
数据缓存功能演示（独立版本）

展示 LRU 缓存的核心功能，不依赖外部库。
"""

import time
from collections import OrderedDict


class SimpleLRUCache:
    """简化的 LRU 缓存实现"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key not in self.cache:
            self.misses += 1
            return None
        
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.capacity,
        }


def demo_lru_cache():
    """演示 LRU 缓存"""
    print("=" * 60)
    print("LRU 缓存演示")
    print("=" * 60)
    
    cache = SimpleLRUCache(capacity=3)
    
    # 添加数据
    print("\n1. 添加数据到缓存:")
    cache.put("image_001", "data_001")
    cache.put("image_002", "data_002")
    cache.put("image_003", "data_003")
    print(f"   缓存大小: {len(cache.cache)}")
    print(f"   缓存内容: {list(cache.cache.keys())}")
    
    # 获取数据
    print("\n2. 从缓存获取数据:")
    result1 = cache.get('image_001')
    print(f"   image_001: {result1} (命中)")
    result2 = cache.get('image_002')
    print(f"   image_002: {result2} (命中)")
    result3 = cache.get('image_999')
    print(f"   image_999: {result3} (未命中)")
    
    # 添加新数据（触发淘汰）
    print("\n3. 添加新数据（超过容量）:")
    print(f"   添加前: {list(cache.cache.keys())}")
    cache.put("image_004", "data_004")
    print(f"   添加后: {list(cache.cache.keys())}")
    print(f"   image_003 被淘汰了吗? {cache.get('image_003') is None}")
    
    # 统计信息
    print("\n4. 缓存统计:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key == 'hit_rate':
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")


def demo_cache_benefits():
    """演示缓存的性能优势"""
    print("\n" + "=" * 60)
    print("缓存性能优势演示")
    print("=" * 60)
    
    # 模拟数据加载函数
    def slow_load_data(idx):
        """模拟慢速数据加载"""
        time.sleep(0.001)  # 模拟 I/O 延迟
        return f"data_{idx}"
    
    # 无缓存
    print("\n1. 无缓存 - 重复加载同一数据 100 次:")
    start = time.time()
    for _ in range(100):
        data = slow_load_data(0)
    time_no_cache = time.time() - start
    print(f"   耗时: {time_no_cache:.3f} 秒")
    
    # 有缓存
    print("\n2. 有缓存 - 重复加载同一数据 100 次:")
    cache = SimpleLRUCache(capacity=10)
    start = time.time()
    for _ in range(100):
        data = cache.get(0)
        if data is None:
            data = slow_load_data(0)
            cache.put(0, data)
    time_with_cache = time.time() - start
    print(f"   耗时: {time_with_cache:.3f} 秒")
    
    # 加速比
    if time_with_cache > 0:
        speedup = time_no_cache / time_with_cache
        print(f"\n3. 加速比: {speedup:.1f}x")
        print(f"   性能提升: {(1 - time_with_cache/time_no_cache) * 100:.1f}%")


def demo_access_patterns():
    """演示不同访问模式下的缓存效果"""
    print("\n" + "=" * 60)
    print("不同访问模式下的缓存效果")
    print("=" * 60)
    
    # 顺序访问
    print("\n1. 顺序访问模式 (0, 1, 2, 3, 4, ...):")
    cache = SimpleLRUCache(capacity=5)
    for i in range(10):
        cache.get(i)
    stats = cache.get_stats()
    print(f"   命中率: {stats['hit_rate']:.2%}")
    print(f"   说明: 顺序访问无法利用缓存")
    
    # 重复访问
    print("\n2. 重复访问模式 (0, 1, 2, 0, 1, 2, ...):")
    cache = SimpleLRUCache(capacity=5)
    for _ in range(3):
        for i in range(3):
            cache.get(i)
    stats = cache.get_stats()
    print(f"   命中率: {stats['hit_rate']:.2%}")
    print(f"   说明: 重复访问可以充分利用缓存")
    
    # 局部性访问
    print("\n3. 局部性访问模式 (0, 0, 1, 1, 2, 2, ...):")
    cache = SimpleLRUCache(capacity=5)
    for i in range(5):
        cache.get(i)
        cache.get(i)  # 立即重复访问
    stats = cache.get_stats()
    print(f"   命中率: {stats['hit_rate']:.2%}")
    print(f"   说明: 局部性访问有较好的缓存效果")


def demo_cache_size_impact():
    """演示缓存大小的影响"""
    print("\n" + "=" * 60)
    print("缓存大小对性能的影响")
    print("=" * 60)
    
    # 测试不同缓存大小
    cache_sizes = [1, 5, 10, 20]
    access_pattern = [i % 15 for i in range(100)]  # 访问 0-14 的数据
    
    print("\n访问模式: 循环访问 15 个不同的数据项，共 100 次")
    print("\n缓存大小 | 命中率 | 说明")
    print("-" * 50)
    
    for size in cache_sizes:
        cache = SimpleLRUCache(capacity=size)
        for idx in access_pattern:
            cache.get(idx)
        
        stats = cache.get_stats()
        hit_rate = stats['hit_rate']
        
        if hit_rate < 0.3:
            desc = "太小，效果差"
        elif hit_rate < 0.7:
            desc = "中等，有改进空间"
        else:
            desc = "良好，接近最优"
        
        print(f"{size:8d} | {hit_rate:6.2%} | {desc}")


def demo_usage_guide():
    """使用指南"""
    print("\n" + "=" * 60)
    print("使用指南")
    print("=" * 60)
    
    print("\n📚 何时使用缓存:")
    print("  ✓ 数据加载耗时（I/O 密集）")
    print("  ✓ 存在重复访问")
    print("  ✓ 内存充足")
    print("  ✓ 数据集不是太大")
    
    print("\n⚙️ 缓存大小选择:")
    print("  • 小数据集: cache_size = dataset_size")
    print("  • 中等数据集: cache_size = batch_size * 10-50")
    print("  • 大数据集: cache_size = 1000-5000")
    
    print("\n🎯 不同场景的建议:")
    print("  • 训练阶段: 使用 LRU 缓存 + 预取")
    print("  • 验证阶段: 使用 LRU 缓存（顺序访问）")
    print("  • 推理阶段: 根据批量大小调整缓存")
    
    print("\n📊 监控指标:")
    print("  • 命中率 > 70%: 缓存效果好")
    print("  • 命中率 30-70%: 可以优化")
    print("  • 命中率 < 30%: 考虑调整策略")


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
        
        # 演示 3: 不同访问模式
        demo_access_patterns()
        
        # 演示 4: 缓存大小影响
        demo_cache_size_impact()
        
        # 演示 5: 使用指南
        demo_usage_guide()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n💡 关键要点:")
        print("  1. LRU 缓存可以显著减少重复数据加载")
        print("  2. 缓存大小需要根据数据集和访问模式调整")
        print("  3. 监控命中率来评估缓存效果")
        print("  4. 结合预取可以进一步提升性能")
        
        print("\n📖 完整文档:")
        print("  med_core/datasets/cache.py - 缓存实现")
        print("  tests/test_cache.py - 测试用例")
        print("  examples/cache_demo.py - 使用示例")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
