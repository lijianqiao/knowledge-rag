"""可插拔 Reranker：local（cross-encoder）/ api（HTTP rerank 服务）。"""

from typing import Protocol

import httpx
from llama_index.core.schema import NodeWithScore

from app.config import (
    ENABLE_RERANK,
    RERANK_API_MODEL,
    RERANK_BACKEND,
    RERANK_BASE_URL,
    RERANK_MODEL,
    RETRIEVE_CANDIDATE_K,
)


class Reranker(Protocol):
    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]: ...


class ApiReranker:
    """调用 HTTP rerank 服务（TEI/Jina 风格）；失败降级为原序截断。"""

    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self._url = base_url.rstrip("/") + "/rerank"
        self._model = model
        self._timeout = timeout

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        if not nodes:
            return []
        docs = [n.get_content() for n in nodes]
        try:
            resp = httpx.post(
                self._url,
                json={"model": self._model, "query": query, "documents": docs, "top_n": top_n},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        except (httpx.HTTPError, KeyError, ValueError):
            return nodes[:top_n]
        # F-2：下标校验在容错块外会因畸形响应抛 IndexError，故逐条校验、非法跳过
        ordered: list[NodeWithScore] = []
        for item in results[:top_n]:
            idx = item.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(nodes)):
                continue
            node = nodes[idx]
            node.score = item.get("relevance_score", node.score)
            ordered.append(node)
        return ordered or nodes[:top_n]  # 全部非法时整体降级


class LocalReranker:
    """本地 cross-encoder（SentenceTransformerRerank 包一层统一接口）。"""

    def __init__(self, model: str, top_n: int):
        from llama_index.core.postprocessor import SentenceTransformerRerank

        self._impl = SentenceTransformerRerank(model=model, top_n=top_n)

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        if not nodes:
            return []
        return self._impl.postprocess_nodes(nodes, query_str=query)[:top_n]


_reranker: Reranker | None = None


def get_reranker() -> Reranker | None:
    """按后端返回 reranker 单例；关闭时 None。"""
    global _reranker
    if not ENABLE_RERANK:
        return None
    if _reranker is None:
        if RERANK_BACKEND == "local":
            _reranker = LocalReranker(RERANK_MODEL, top_n=RETRIEVE_CANDIDATE_K)
        else:
            _reranker = ApiReranker(RERANK_BASE_URL, RERANK_API_MODEL)
    return _reranker
