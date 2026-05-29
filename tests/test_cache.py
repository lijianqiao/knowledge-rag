"""进程内语义缓存：余弦、阈值命中/未命中、LRU 淘汰（纯逻辑，离线）。"""

from app.cache import SemanticCache, _cosine


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0   # 同向 → 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0   # 正交 → 0.0
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0   # 零向量 → 0.0


def test_cache_hit_above_threshold():
    cache = SemanticCache(threshold=0.97, max_size=128)
    cache.put([1.0, 0.0], "ans")
    # 余弦 ≈ 0.999 ≥ 0.97 → 命中
    assert cache.get([0.999, 0.044]) == "ans"


def test_cache_miss_below_threshold():
    cache = SemanticCache(threshold=0.97, max_size=128)
    cache.put([1.0, 0.0], "ans")
    assert cache.get([0.0, 1.0]) is None  # 余弦 0 < 0.97


def test_lru_eviction():
    cache = SemanticCache(threshold=0.97, max_size=2)
    cache.put([1.0, 0.0], "a")
    cache.put([0.0, 1.0], "b")
    cache.put([0.0, 0.0, 1.0], "c")  # 第三条 → 淘汰最久未用的第一条
    assert cache.get([1.0, 0.0]) is None        # 第一条已被淘汰
    assert cache.get([0.0, 1.0]) == "b"
    assert cache.get([0.0, 0.0, 1.0]) == "c"
