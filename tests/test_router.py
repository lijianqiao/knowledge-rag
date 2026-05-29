"""路由分类纯逻辑测试（注入假 LLM）。"""

import app.router as r


class _LLM:
    def __init__(self, text):
        self._t = text

    def complete(self, prompt):
        assert "问题" in prompt
        return self._t


def test_classify_graph(monkeypatch):
    monkeypatch.setattr(r, "ENABLE_GRAPH", True)
    monkeypatch.setattr(r, "graph_store_exists", lambda: True)
    monkeypatch.setattr(r, "get_llm", lambda: _LLM("graph"))
    assert r.classify_route("订单故障会影响哪些服务") == "graph"


def test_classify_defaults_vector_on_garbage(monkeypatch):
    monkeypatch.setattr(r, "ENABLE_GRAPH", True)
    monkeypatch.setattr(r, "graph_store_exists", lambda: True)
    monkeypatch.setattr(r, "get_llm", lambda: _LLM("胡言乱语"))
    assert r.classify_route("x") == "vector"  # 非 graph 一律 vector（R4）


def test_classify_vector_when_graph_disabled(monkeypatch):
    monkeypatch.setattr(r, "ENABLE_GRAPH", False)
    # 关闭时不应调用 LLM
    monkeypatch.setattr(r, "get_llm", lambda: (_ for _ in ()).throw(AssertionError("不应调用")))
    assert r.classify_route("订单故障会影响哪些服务") == "vector"


def test_classify_vector_when_graph_not_built(monkeypatch):
    monkeypatch.setattr(r, "ENABLE_GRAPH", True)
    monkeypatch.setattr(r, "graph_store_exists", lambda: False)  # R5
    monkeypatch.setattr(r, "get_llm", lambda: (_ for _ in ()).throw(AssertionError("不应调用")))
    assert r.classify_route("订单故障会影响哪些服务") == "vector"


def test_classify_falls_back_vector_on_llm_error(monkeypatch):
    monkeypatch.setattr(r, "ENABLE_GRAPH", True)
    monkeypatch.setattr(r, "graph_store_exists", lambda: True)

    class _Boom:
        def complete(self, prompt):
            raise RuntimeError("down")

    monkeypatch.setattr(r, "get_llm", lambda: _Boom())
    assert r.classify_route("任意") == "vector"
