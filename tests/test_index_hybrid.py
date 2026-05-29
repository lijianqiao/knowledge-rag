"""Hybrid 检索的纯逻辑测试（不连真实服务）。"""

import app.index as index_mod
from llama_index.core.schema import NodeWithScore, TextNode

from app.index import nodes_from_chroma_payload


def test_nodes_from_chroma_payload_maps_fields():
    payload = {
        "ids": ["doc_a__0", "doc_b__1"],
        "documents": ["内容A", "内容B"],
        "metadatas": [{"title": "A"}, {"title": "B"}],
    }
    nodes = nodes_from_chroma_payload(payload)
    assert len(nodes) == 2
    assert nodes[0].id_ == "doc_a__0"
    assert nodes[0].get_content() == "内容A"
    assert nodes[1].metadata["title"] == "B"


def test_nodes_from_chroma_payload_handles_missing_metadata():
    payload = {"ids": ["x__0"], "documents": ["t"], "metadatas": [None]}
    nodes = nodes_from_chroma_payload(payload)
    assert nodes[0].metadata == {}


def test_nodes_from_chroma_payload_empty():
    assert nodes_from_chroma_payload({"ids": [], "documents": [], "metadatas": []}) == []


def test_retrieve_nodes_wires_fusion_and_rerank(monkeypatch):
    nodes = [NodeWithScore(node=TextNode(text=f"n{i}"), score=1.0 / (i + 1)) for i in range(4)]

    class _Coll:
        def count(self):
            return 4

    class _Retriever:
        def retrieve(self, query):
            return nodes

    class _Index:
        def as_retriever(self, **kwargs):
            return _Retriever()

    monkeypatch.setattr(index_mod, "ENABLE_HYBRID", True)
    monkeypatch.setattr(index_mod, "get_chroma_collection", lambda: _Coll())
    monkeypatch.setattr(index_mod, "get_index", lambda: _Index())
    monkeypatch.setattr(index_mod, "get_bm25_retriever", lambda dt="all": _Retriever())
    monkeypatch.setattr(index_mod, "get_llm", lambda: object())
    monkeypatch.setattr(index_mod, "get_reranker", lambda: None)
    monkeypatch.setattr(
        index_mod, "QueryFusionRetriever", lambda retrievers, **kwargs: _Retriever()
    )

    out = index_mod.retrieve_nodes("q", top_k=2, doc_type="all")
    assert len(out) == 2  # 截断到 top_k


def test_retrieve_nodes_raises_on_empty_collection(monkeypatch):
    class _Empty:
        def count(self):
            return 0

    monkeypatch.setattr(index_mod, "get_chroma_collection", lambda: _Empty())
    try:
        index_mod.retrieve_nodes("q")
        assert False, "应抛 ValueError"
    except ValueError:
        pass


# --- F3: 融合后按 doc_type 兜底过滤 ---


def test_filter_by_doc_type_keeps_only_matching():
    nodes = [
        NodeWithScore(node=TextNode(text="p", metadata={"doc_type": "prompt"})),
        NodeWithScore(node=TextNode(text="d", metadata={"doc_type": "doc"})),
    ]
    out = index_mod._filter_by_doc_type(nodes, "prompt")
    assert [n.get_content() for n in out] == ["p"]


def test_filter_by_doc_type_all_is_passthrough():
    nodes = [NodeWithScore(node=TextNode(text="x", metadata={"doc_type": "doc"}))]
    assert index_mod._filter_by_doc_type(nodes, "all") == nodes


# --- Phase 2 Task 2.1: per-doc_type BM25 严格过滤 ---


def test_bm25_built_per_doc_type(monkeypatch):
    import app.index as m
    from llama_index.core.schema import TextNode

    all_nodes = [
        TextNode(text="p", id_="1", metadata={"doc_type": "prompt"}),
        TextNode(text="d", id_="2", metadata={"doc_type": "doc"}),
    ]
    monkeypatch.setattr(m, "load_all_nodes", lambda: all_nodes)

    captured = {}
    # 拦截真实建库（不跑 bm25s/jieba），只验证按 doc_type 过滤后的节点与 clamp 的 k
    monkeypatch.setattr(
        m, "_build_jieba_bm25", lambda nodes, top_k: captured.update(nodes=nodes, top_k=top_k)
    )
    m.reset_index_cache()

    m.get_bm25_retriever("prompt")
    assert [n.id_ for n in captured["nodes"]] == ["1"]  # 仅 prompt 类
    assert captured["top_k"] == 1  # k clamp 到节点数（min(BM25_TOP_K, 1)）


def test_bm25_none_when_doc_type_empty(monkeypatch):
    import app.index as m
    from llama_index.core.schema import TextNode

    monkeypatch.setattr(m, "load_all_nodes", lambda: [TextNode(text="d", id_="2", metadata={"doc_type": "doc"})])
    monkeypatch.setattr(m, "_build_jieba_bm25", lambda nodes, top_k: object())
    m.reset_index_cache()
    assert m.get_bm25_retriever("prompt") is None  # 该类型无节点 → None，调用方退化纯向量


def test_jieba_bm25_real_word_match_and_original_text():
    """真实 bm25s+jieba（进程内，无模型/网络）：词级命中，且召回返回原文。"""
    import app.index as m
    from llama_index.core.schema import TextNode

    nodes = [
        TextNode(text="订单服务内存溢出导致502", id_="1", metadata={"doc_type": "doc"}),
        TextNode(text="Redis 缓存清理操作手册", id_="2", metadata={"doc_type": "doc"}),
    ]
    retriever = m._build_jieba_bm25(nodes, top_k=2)
    res = retriever.retrieve("内存溢出怎么排查")
    assert res[0].node.node_id == "1"  # jieba 词级命中「内存/溢出」
    assert res[0].get_content() == "订单服务内存溢出导致502"  # 返回原文，非切分文本


# --- F1: weak 判断用向量余弦分，与融合/重排分解耦 ---


def test_is_retrieval_weak_uses_threshold(monkeypatch):
    monkeypatch.setattr(index_mod, "RETRIEVE_SCORE_THRESHOLD", 0.35)
    assert index_mod.is_retrieval_weak(0.34) is True
    assert index_mod.is_retrieval_weak(0.36) is False


def test_retrieve_with_diagnostics_returns_vector_cosine_not_rrf(monkeypatch):
    vec_nodes = [
        NodeWithScore(node=TextNode(text="a"), score=0.71),
        NodeWithScore(node=TextNode(text="b"), score=0.42),
    ]
    fused = [NodeWithScore(node=TextNode(text=f"f{i}"), score=0.016) for i in range(3)]

    class _Coll:
        def count(self):
            return 2

    class _VecRetriever:
        def retrieve(self, query):
            return vec_nodes

    class _Index:
        def as_retriever(self, **kwargs):
            return _VecRetriever()

    class _Fusion:
        def retrieve(self, query):
            return fused

    monkeypatch.setattr(index_mod, "ENABLE_HYBRID", True)
    monkeypatch.setattr(index_mod, "get_chroma_collection", lambda: _Coll())
    monkeypatch.setattr(index_mod, "get_index", lambda: _Index())
    monkeypatch.setattr(index_mod, "get_bm25_retriever", lambda dt="all": object())
    monkeypatch.setattr(index_mod, "get_llm", lambda: object())
    monkeypatch.setattr(index_mod, "get_reranker", lambda: None)
    monkeypatch.setattr(index_mod, "QueryFusionRetriever", lambda retrievers, **kwargs: _Fusion())

    nodes, best = index_mod.retrieve_with_diagnostics("q", top_k=2, doc_type="all")
    assert best == 0.71  # 取向量余弦最高分，而非融合 RRF 分 0.016
    assert len(nodes) == 2  # 最终节点来自融合结果，截断到 top_k


# --- Phase 2 Task 2.3: 多查询扩展开关控制 num_queries ---


def test_multi_query_sets_num_queries(monkeypatch):
    import app.index as m
    from llama_index.core.schema import NodeWithScore, TextNode

    captured = {}

    class _Coll:
        def count(self): return 1

    class _Retriever:
        def retrieve(self, q): return [NodeWithScore(node=TextNode(text="x"), score=0.5)]

    class _Index:
        def as_retriever(self, **k): return _Retriever()

    def _fake_fusion(retrievers, **kwargs):
        captured.update(kwargs)
        return _Retriever()

    monkeypatch.setattr(m, "ENABLE_HYBRID", True)
    monkeypatch.setattr(m, "ENABLE_MULTI_QUERY", True)
    monkeypatch.setattr(m, "MULTI_QUERY_NUM", 4)
    monkeypatch.setattr(m, "get_chroma_collection", lambda: _Coll())
    monkeypatch.setattr(m, "get_index", lambda: _Index())
    monkeypatch.setattr(m, "get_bm25_retriever", lambda dt="all": _Retriever())
    monkeypatch.setattr(m, "get_llm", lambda: object())
    monkeypatch.setattr(m, "get_reranker", lambda: None)
    monkeypatch.setattr(m, "QueryFusionRetriever", _fake_fusion)

    m.retrieve_nodes("q", top_k=2, doc_type="all")
    assert captured["num_queries"] == 4


# --- F2: 真实 QueryFusionRetriever 在 num_queries=1 下离线可跑、且不调用 LLM ---


def test_real_fusion_num_queries_1_does_not_call_llm():
    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.retrievers import QueryFusionRetriever as RealQueryFusionRetriever

    class _NoCallLLM(MockLLM):
        def complete(self, *args, **kwargs):
            raise AssertionError("num_queries=1 时不应调用 LLM")

        def chat(self, *args, **kwargs):
            raise AssertionError("num_queries=1 时不应调用 LLM")

    class _MemRetriever(BaseRetriever):
        def __init__(self, nodes):
            self._nodes = nodes
            super().__init__()

        def _retrieve(self, query_bundle):
            return self._nodes

    r1 = _MemRetriever([NodeWithScore(node=TextNode(text="a", id_="1"), score=0.9)])
    r2 = _MemRetriever([NodeWithScore(node=TextNode(text="b", id_="2"), score=0.8)])

    fusion = RealQueryFusionRetriever(
        [r1, r2],
        llm=_NoCallLLM(),
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    out = fusion.retrieve("hello")  # 若触发 LLM，_NoCallLLM 会抛断言
    assert len(out) >= 1
    assert all(isinstance(n, NodeWithScore) for n in out)
