"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: graph.py
@DateTime: 2026-05-28
@Docs: LangGraph RAG 工作流：检索 → 生成 → 输出 →（不足时）重检索
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from llama_index.core.schema import NodeWithScore

from app.config import ENABLE_QUERY_REWRITE, MAX_RETRIEVE_RETRIES, RETRIEVE_TOP_K
from app.index import (
    build_context,
    format_nodes,
    generate_answer,
    is_retrieval_weak,
    retrieve_with_diagnostics,
    rewrite_query,
)


class RAGState(TypedDict):
    """LangGraph RAG 状态。"""

    question: str
    doc_type: str
    top_k: int
    retry_count: int
    search_query: str
    nodes: list[NodeWithScore]
    best_vector_score: float
    context: str
    answer: str
    sources: str
    result: str


def _retrieve(state: RAGState) -> dict:
    """节点：LlamaIndex 检索 + 组装 context。"""
    nodes, best_vector_score = retrieve_with_diagnostics(
        query=state["search_query"],
        top_k=state["top_k"],
        doc_type=state["doc_type"],
    )
    return {
        "nodes": nodes,
        "best_vector_score": best_vector_score,
        "context": build_context(nodes),
        "sources": format_nodes(nodes),
    }


def _generate(state: RAGState) -> dict:
    """节点：LLM 生成答案。"""
    answer = generate_answer(state["question"], state["context"])
    return {"answer": answer}


def _output(state: RAGState) -> dict:
    """节点：LangGraph 组装最终输出（答案 + 参考来源）。"""
    return {"result": f"{state['answer']}\n\n--- 参考来源 ---\n{state['sources']}"}


def _prepare_retry(state: RAGState) -> dict:
    """节点：改写查询（或扩大检索）后重试。"""
    if ENABLE_QUERY_REWRITE:
        new_query = rewrite_query(state["question"], state["search_query"])
    else:
        new_query = state["question"]
    return {
        "retry_count": state["retry_count"] + 1,
        "top_k": state["top_k"] + 3,
        "doc_type": "all",
        "search_query": new_query,
    }


def _route_after_output(state: RAGState) -> Literal["prepare_retry", "__end__"]:
    """条件边：输出后判断，信息不足或检索偏弱时回到检索。"""
    if state["retry_count"] >= MAX_RETRIEVE_RETRIES:
        return END

    answer = state.get("answer") or ""

    if is_retrieval_weak(state.get("best_vector_score", 0.0)):
        return "prepare_retry"
    if "信息不足" in answer:
        return "prepare_retry"
    return END


def build_rag_graph():
    """
    构建 LangGraph RAG 工作流。

    Returns:
        编译后的 StateGraph
    """
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)
    builder.add_node("output", _output)
    builder.add_node("prepare_retry", _prepare_retry)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "output")
    builder.add_conditional_edges("output", _route_after_output)
    builder.add_edge("prepare_retry", "retrieve")

    return builder.compile()


_rag_graph = None


def get_rag_graph():
    """获取单例 RAG 图。"""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


def run_ask(question: str, top_k: int = RETRIEVE_TOP_K, doc_type: str = "all") -> str:
    """
    执行 LangGraph RAG 问答。

    Args:
        question: 用户问题
        top_k: 检索条数
        doc_type: 文档类型过滤

    Returns:
        答案与参考来源
    """
    graph = get_rag_graph()
    result = graph.invoke(
        {
            "question": question,
            "doc_type": doc_type,
            "top_k": top_k,
            "retry_count": 0,
            "search_query": question,
            "nodes": [],
            "best_vector_score": 0.0,
            "context": "",
            "answer": "",
            "sources": "",
            "result": "",
        }
    )
    return result["result"]
