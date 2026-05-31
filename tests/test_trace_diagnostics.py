"""run_ask 语义缓存命中/未命中的 trace 埋点（纯逻辑，注入假 cache/embed/graph）。"""

import app.graph as g


class _Embed:
    def get_query_embedding(self, _q):
        return [0.1, 0.2, 0.3]


def test_run_ask_logs_cache_hit(monkeypatch):
    events = []
    monkeypatch.setattr(g, "log_event", lambda tid, ev, payload: events.append((ev, payload)))
    monkeypatch.setattr(g, "get_embed_model", lambda: _Embed())

    class _Cache:
        def get(self, _emb):
            return "cached answer"

        def put(self, _emb, _val):  # pragma: no cover - 命中路径不会调用
            raise AssertionError("命中后不应再写缓存")

    monkeypatch.setattr(g, "get_semantic_cache", lambda: _Cache())

    assert g.run_ask("问题") == "cached answer"
    assert ("cache", {"hit": True}) in events


def test_run_ask_logs_cache_miss(monkeypatch):
    events = []
    monkeypatch.setattr(g, "log_event", lambda tid, ev, payload: events.append((ev, payload)))
    monkeypatch.setattr(g, "get_embed_model", lambda: _Embed())

    class _Cache:
        def __init__(self):
            self.put_called = False

        def get(self, _emb):
            return None

        def put(self, _emb, _val):
            self.put_called = True

    cache = _Cache()
    monkeypatch.setattr(g, "get_semantic_cache", lambda: cache)

    class _Graph:
        def invoke(self, _state):
            return {"result": "fresh answer"}

    monkeypatch.setattr(g, "get_rag_graph", lambda: _Graph())

    assert g.run_ask("问题") == "fresh answer"
    assert ("cache", {"hit": False}) in events
    assert cache.put_called
