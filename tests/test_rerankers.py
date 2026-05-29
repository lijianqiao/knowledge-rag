"""可插拔 reranker 测试（注入假 HTTP，不联网）。"""

from llama_index.core.schema import NodeWithScore, TextNode

import app.rerankers as rr


def _nodes():
    return [NodeWithScore(node=TextNode(text=t), score=0.0) for t in ("a", "bb", "ccc")]


def test_api_reranker_reorders_and_truncates(monkeypatch):
    # 假装 API 把第 2 篇(index=2)判为最相关
    def _fake_post(url, json, timeout):
        class _Resp:
            def raise_for_status(self): ...
            def json(self):
                return {"results": [{"index": 2, "relevance_score": 0.9},
                                    {"index": 0, "relevance_score": 0.5},
                                    {"index": 1, "relevance_score": 0.1}]}
        return _Resp()

    monkeypatch.setattr(rr.httpx, "post", _fake_post)
    reranker = rr.ApiReranker(base_url="http://x", model="m")
    out = reranker.rerank("q", _nodes(), top_n=2)
    assert [n.get_content() for n in out] == ["ccc", "a"]


def test_api_reranker_falls_back_on_error(monkeypatch):
    def _boom(*a, **k):
        raise rr.httpx.HTTPError("down")

    monkeypatch.setattr(rr.httpx, "post", _boom)
    reranker = rr.ApiReranker(base_url="http://x", model="m")
    out = reranker.rerank("q", _nodes(), top_n=2)
    assert len(out) == 2  # 降级为原序截断，不抛


def test_api_reranker_handles_malformed_index(monkeypatch):
    # 服务返回越界/非整数 index 不应抛 IndexError，整体降级（F-2）
    def _fake_post(url, json, timeout):
        class _Resp:
            def raise_for_status(self): ...
            def json(self):
                return {"results": [{"index": 99, "relevance_score": 0.9},
                                    {"index": "x", "relevance_score": 0.5}]}
        return _Resp()

    monkeypatch.setattr(rr.httpx, "post", _fake_post)
    out = rr.ApiReranker(base_url="http://x", model="m").rerank("q", _nodes(), top_n=2)
    assert len(out) == 2  # 全部非法 → 降级为原序截断


def test_get_reranker_none_when_disabled(monkeypatch):
    monkeypatch.setattr(rr, "ENABLE_RERANK", False)
    assert rr.get_reranker() is None
