"""图索引构建/检索装配测试（注入假模型，不联网）。"""

import app.graph_index as gi


def test_get_graph_index_none_when_no_store(monkeypatch):
    # 图未构建 → None（R5：路由据此回退）
    monkeypatch.setattr(gi, "graph_store_exists", lambda: False)
    assert gi.get_graph_index() is None


def test_build_graph_index_invokes_extractor_and_persists(monkeypatch):
    from llama_index.core.schema import TextNode

    captured = {}

    class _FakePGI:
        pass

    def _fake_build(nodes, kg_extractors, embed_model, show_progress):
        captured["n_nodes"] = len(nodes)
        captured["has_extractor"] = bool(kg_extractors)
        return _FakePGI()

    monkeypatch.setattr(gi, "_build_property_graph_index", _fake_build)
    monkeypatch.setattr(gi, "_persist_index", lambda idx: captured.setdefault("persisted", True))
    # 避免构造真实 SchemaLLMPathExtractor（其 pydantic 校验需真实 LLM）
    monkeypatch.setattr(gi, "_make_extractor", lambda: object())
    monkeypatch.setattr(gi, "get_embed_model", lambda: object())

    gi.build_graph_index([TextNode(text="订单服务依赖Redis", id_="1")])
    assert captured["n_nodes"] == 1
    assert captured["has_extractor"] is True
    assert captured["persisted"] is True


def test_graph_retrieve_empty_when_no_index(monkeypatch):
    monkeypatch.setattr(gi, "get_graph_index", lambda: None)
    assert gi.graph_retrieve("任意问题", top_k=5) == []
