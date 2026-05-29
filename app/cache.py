"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: cache.py
@DateTime: 2026-05-29
@Docs: 进程内语义缓存（自写，零依赖）。embedding-agnostic：调用方传 query embedding，本模块只做余弦匹配 + LRU 淘汰。
"""

import math
from collections import OrderedDict
from itertools import count
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量余弦相似度；任一向量零范数返回 0.0。"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """进程内语义 LRU 缓存：query embedding → value。

    命中条件：与某缓存项余弦 >= threshold（取最相似的一条）。
    淘汰策略：超出 max_size 时弹出最久未使用项（OrderedDict 头部）。
    """

    def __init__(self, threshold: float, max_size: int) -> None:
        self._threshold = threshold
        self._max_size = max_size
        self._store: OrderedDict[int, tuple[list[float], Any]] = OrderedDict()
        self._counter = count()

    def get(self, query_embedding: list[float]) -> Any | None:
        """返回与 query 最相似且余弦 >= threshold 的缓存值；无命中返回 None。命中项标记为最近使用。"""
        best_key: int | None = None
        best_sim = self._threshold
        for key, (emb, _value) in self._store.items():
            sim = _cosine(query_embedding, emb)
            if sim >= best_sim:
                best_sim = sim
                best_key = key
        if best_key is None:
            return None
        self._store.move_to_end(best_key)
        return self._store[best_key][1]

    def put(self, query_embedding: list[float], value: Any) -> None:
        """写入一条缓存；超出容量则淘汰最久未使用项。"""
        key = next(self._counter)
        self._store[key] = (query_embedding, value)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """清空缓存。"""
        self._store.clear()
