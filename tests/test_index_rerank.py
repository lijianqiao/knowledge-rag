"""重排编排逻辑测试（不加载真实模型）。"""

from llama_index.core.schema import NodeWithScore, TextNode

from app.index import apply_rerank


def _node(text: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text), score=score)


class _FakeReranker:
    """按文本长度倒序重排并切到 top_n（新契约：reranker.rerank 负责切片）。"""

    def rerank(self, query, nodes, top_n):
        return sorted(nodes, key=lambda n: len(n.get_content()), reverse=True)[:top_n]


def test_apply_rerank_none_reranker_truncates_only():
    nodes = [_node("a", 0.9), _node("bb", 0.8), _node("ccc", 0.7)]
    out = apply_rerank(None, "q", nodes, top_n=2)
    assert len(out) == 2
    assert out[0].get_content() == "a"  # 关闭重排时保持原顺序


def test_apply_rerank_empty_nodes_safe():
    assert apply_rerank(_FakeReranker(), "q", [], top_n=5) == []


def test_apply_rerank_uses_reranker_order_and_slices():
    # reranker 返回全部 3 条重排结果，apply_rerank 必须按 top_n 切到 2（S1 回归）
    nodes = [_node("a", 0.9), _node("bbb", 0.1), _node("cc", 0.5), _node("dddd", 0.4)]
    out = apply_rerank(_FakeReranker(), "q", nodes, top_n=2)
    assert [n.get_content() for n in out] == ["dddd", "bbb"]


def test_apply_rerank_respects_top_n_above_default():
    # top_n=3 时应返回 3 条，不被任何固定 RERANK_TOP_N 截到更少（S1 回归）
    nodes = [_node("a", 0.9), _node("bbb", 0.1), _node("cc", 0.5), _node("dddd", 0.4)]
    out = apply_rerank(_FakeReranker(), "q", nodes, top_n=3)
    assert len(out) == 3
