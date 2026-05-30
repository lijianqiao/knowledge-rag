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

from app.config import ENABLE_GRAPH, ENABLE_QUERY_REWRITE, MAX_RETRIEVE_RETRIES, RETRIEVE_TOP_K
from app.graph_index import graph_retrieve
from app.index import (
    build_context,
    format_nodes,
    generate_answer,
    generate_answer_stream,
    is_retrieval_weak,
    retrieve_with_diagnostics,
    rewrite_query,
)
from app.router import classify_route
from app.trace import log_event, new_trace_id


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
    trace_id: str
    allowed_sources: list | None


def _do_retrieve(
    search_query: str, top_k: int, doc_type: str, allowed_sources: list | None = None
) -> tuple[list[NodeWithScore], float, str, str]:
    """按路由选 graph / vector 检索并组装 context / sources（无 trace、无重试）。

    供图节点 `_retrieve` 与单趟流式 `run_ask_stream` 共用，保证两条路径检索逻辑一致。

    Returns:
        (nodes, best_vector_score, context, sources)
    """
    route = classify_route(search_query) if ENABLE_GRAPH else "vector"
    if route == "graph":
        nodes = graph_retrieve(search_query, top_k=top_k)
        best_vector_score = 1.0  # 图路径不参与余弦 weak 判断，置高分避免误触发重检索
    else:
        nodes, best_vector_score = retrieve_with_diagnostics(
            query=search_query, top_k=top_k, doc_type=doc_type, allowed_sources=allowed_sources
        )
    return nodes, best_vector_score, build_context(nodes), format_nodes(nodes)


def _retrieve(state: RAGState) -> dict:
    """节点：按路由选 graph / vector 检索 + 组装 context。"""
    nodes, best_vector_score, context, sources = _do_retrieve(
        state["search_query"], state["top_k"], state["doc_type"], state.get("allowed_sources")
    )
    log_event(
        state.get("trace_id", ""),
        "retrieve",
        {
            "best_vector_score": best_vector_score,
            "top_k": state["top_k"],
            "topk": [(n.metadata.get("source", "?"), round(n.score or 0.0, 4)) for n in nodes[:5]],
        },
    )
    return {
        "nodes": nodes,
        "best_vector_score": best_vector_score,
        "context": context,
        "sources": sources,
    }


def _generate(state: RAGState) -> dict:
    """节点：LLM 生成答案。"""
    answer = generate_answer(state["question"], state["context"])
    log_event(state.get("trace_id", ""), "answer", {"len": len(answer)})
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
    log_event(
        state.get("trace_id", ""),
        "rewrite",
        {"from": state["search_query"], "to": new_query, "retry": state["retry_count"] + 1},
    )
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


def _build_rag_builder() -> StateGraph:
    """组装 RAG 工作流节点与边（未编译），供无状态图与会话图共用。"""
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

    return builder


def build_rag_graph():
    """
    构建 LangGraph RAG 工作流（无状态，无 checkpointer）。

    Returns:
        编译后的 StateGraph
    """
    return _build_rag_builder().compile()


def build_session_graph(checkpointer, interrupt_before: list[str] | None = None):
    """
    构建带 checkpointer 的会话图：复用无状态图的同一套节点/边，额外挂持久化与可选 HITL 中断。

    与 `build_rag_graph` 并存——无状态 `run_ask` 路径不受影响。

    Args:
        checkpointer: LangGraph checkpointer（如 SqliteSaver），按 thread_id 持久化会话状态
        interrupt_before: 在这些节点前中断以供人工审批（如 ["generate"]）；None 表示直通

    Returns:
        编译后的 StateGraph（支持 thread_id 与 interrupt/resume）
    """
    builder = _build_rag_builder()
    if interrupt_before:
        return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    return builder.compile(checkpointer=checkpointer)


_rag_graph = None


def get_rag_graph():
    """获取单例 RAG 图。"""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


def run_ask(
    question: str,
    top_k: int = RETRIEVE_TOP_K,
    doc_type: str = "all",
    allowed_sources: list | None = None,
) -> str:
    """
    执行 LangGraph RAG 问答。

    Args:
        question: 用户问题
        top_k: 检索条数
        doc_type: 文档类型过滤
        allowed_sources: 允许访问的 source 白名单（应用层 RBAC）；None 表示不限来源

    Returns:
        答案与参考来源
    """
    graph = get_rag_graph()
    trace_id = new_trace_id()
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
            "trace_id": trace_id,
            "allowed_sources": allowed_sources,
        }
    )
    return result["result"]


def run_ask_stream(
    question: str,
    top_k: int = RETRIEVE_TOP_K,
    doc_type: str = "all",
    allowed_sources: list | None = None,
):
    """
    流式 RAG 问答：单趟检索（不走 LangGraph 重试循环）+ 逐 token yield 答案。

    流式是 CLI 输出关注点，刻意与图的重试循环解耦：检索只做一次（best-effort 单趟），
    先 yield 答案增量，答案耗尽后再 yield 末尾的参考来源块。

    Args:
        question: 用户问题
        top_k: 检索条数
        doc_type: 文档类型过滤
        allowed_sources: 允许访问的 source 白名单（应用层 RBAC）；None 表示不限来源

    Yields:
        答案增量片段，最后一项为「--- 参考来源 ---」块
    """
    trace_id = new_trace_id()
    _, best_vector_score, context, sources = _do_retrieve(
        question, top_k, doc_type, allowed_sources
    )
    log_event(trace_id, "retrieve", {"best_vector_score": best_vector_score, "top_k": top_k})
    yield from generate_answer_stream(question, context)
    yield f"\n\n--- 参考来源 ---\n{sources}"
