"""检索路由：按问题类型选择 graph / vector 通道。"""

from app.config import ENABLE_GRAPH, ROUTE_CLASSIFY_PROMPT
from app.graph_store import graph_store_exists
from app.index import get_llm


def classify_route(question: str) -> str:
    """返回 'graph' 或 'vector'。

    仅当 ENABLE_GRAPH 且图已构建时才尝试 LLM 分类；
    分类非 'graph'、出错、或前置条件不满足，一律回退 'vector'（R4/R5）。
    """
    if not ENABLE_GRAPH or not graph_store_exists():
        return "vector"
    try:
        out = str(get_llm().complete(ROUTE_CLASSIFY_PROMPT.format(question=question))).strip().lower()
    except Exception:
        return "vector"
    return "graph" if "graph" in out else "vector"
