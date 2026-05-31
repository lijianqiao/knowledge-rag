"""语义缓存接入 run_ask 热路径的测试（全离线，注入假 embed/graph）。"""

import app.graph as g


def _patch_graph(monkeypatch, calls):
    """把 run_ask 的图执行替换为计数器，返回固定 result。"""

    class _Graph:
        def invoke(self, state):
            calls.append(state["question"])
            return {"result": "答案A"}

    monkeypatch.setattr(g, "get_rag_graph", lambda: _Graph())


def test_cache_hit_skips_graph(monkeypatch):
    calls: list[str] = []
    _patch_graph(monkeypatch, calls)

    # 启用缓存：注入一个真实 SemanticCache（阈值 0.9）+ 固定 embedding
    from app.cache import SemanticCache

    cache = SemanticCache(threshold=0.9, max_size=8)
    monkeypatch.setattr(g, "get_semantic_cache", lambda: cache)

    class _Embed:
        def get_query_embedding(self, q):
            return [1.0, 0.0]  # 同问题 → 同向量 → 命中

    monkeypatch.setattr(g, "get_embed_model", lambda: _Embed())

    a1 = g.run_ask("同一个问题")   # miss → 跑图 → 入缓存
    a2 = g.run_ask("同一个问题")   # hit → 不跑图
    assert a1 == "答案A" and a2 == "答案A"
    assert calls == ["同一个问题"]  # 图只被调用一次（第二次命中缓存）


def test_cache_bypassed_when_rbac_or_doctype(monkeypatch):
    """有 allowed_sources 或非 all 类型时绕过缓存（防跨范围串答）。"""
    calls: list[str] = []
    _patch_graph(monkeypatch, calls)
    # 即便 get_semantic_cache 返回缓存，cacheable=False 时不应被使用
    from app.cache import SemanticCache

    monkeypatch.setattr(g, "get_semantic_cache", lambda: SemanticCache(0.9, 8))

    def _no_embed():
        raise AssertionError("RBAC/非默认类型不应触达 embedding/cache")

    monkeypatch.setattr(g, "get_embed_model", lambda: _no_embed())

    g.run_ask("q", allowed_sources=["运维文档"])   # 有 RBAC → 绕过缓存
    g.run_ask("q", doc_type="doc")                  # 非 all → 绕过缓存
    assert calls == ["q", "q"]  # 两次都真正跑图，未碰缓存/embedding
