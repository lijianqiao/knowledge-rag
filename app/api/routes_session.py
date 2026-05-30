"""会话 / 人机循环端点：SqliteSaver checkpointer 持久化 thread + 可选 interrupt/resume。

与无状态 `/ask` 并存：会话图 thread_id = 会话 id，按 thread 持久化多轮状态、支持重启恢复。
`require_approval=true` 时在「生成答案前」中断（HITL），经 `/sessions/{id}/resume` 审批继续。
"""

import uuid

from fastapi import APIRouter, Depends

from app.api.auth import Principal, require_principal
from app.api.checkpoint import get_checkpointer
from app.graph import build_session_graph
from app.trace import new_trace_id

router = APIRouter(prefix="/sessions")

# 按是否启用审批缓存会话图（图对象无状态，状态由 checkpointer 持有）。
_graphs: dict[bool, object] = {}


def get_session_graph(require_approval: bool):
    """缓存的会话图工厂：按 require_approval 复用编译后的图。"""
    if require_approval not in _graphs:
        _graphs[require_approval] = build_session_graph(
            get_checkpointer(),
            interrupt_before=["generate"] if require_approval else None,
        )
    return _graphs[require_approval]


def reset_session_graphs() -> None:
    """清空会话图缓存（供测试，配合 reset_checkpointer 切换后端）。"""
    _graphs.clear()


def _initial_state(question: str, top_k: int, doc_type: str, allowed_sources: list | None) -> dict:
    """构造会话图的初始 RAGState。"""
    return {
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
        "trace_id": new_trace_id(),
        "allowed_sources": allowed_sources,
    }


@router.post("")
def create_session(body: dict, principal: Principal = Depends(require_principal)) -> dict:
    """开新会话并跑一轮：命中中断返回 interrupted，否则返回 done + 答案。"""
    question = body["question"]
    top_k = body.get("top_k", 5)
    doc_type = body.get("doc_type", "all")
    require_approval = bool(body.get("require_approval", False))

    thread_id = uuid.uuid4().hex
    graph = get_session_graph(require_approval)
    config = {"configurable": {"thread_id": thread_id}}

    state = _initial_state(question, top_k, doc_type, principal.allowed_sources)
    graph.invoke(state, config=config)

    snapshot = graph.get_state(config)
    if snapshot.next:
        return {"status": "interrupted", "thread_id": thread_id, "pending": list(snapshot.next)}
    return {"status": "done", "thread_id": thread_id, "answer": snapshot.values.get("result", "")}


@router.post("/{thread_id}/resume")
def resume_session(
    thread_id: str, body: dict | None = None, principal: Principal = Depends(require_principal)
) -> dict:
    """审批通过后继续：从中断点续跑；再次中断则仍返回 interrupted。"""
    require_approval = bool((body or {}).get("require_approval", True))
    graph = get_session_graph(require_approval)
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(None, config=config)

    snapshot = graph.get_state(config)
    if snapshot.next:
        return {"status": "interrupted", "thread_id": thread_id, "pending": list(snapshot.next)}
    return {"status": "done", "thread_id": thread_id, "answer": snapshot.values.get("result", "")}


@router.get("/{thread_id}")
def get_session(
    thread_id: str, require_approval: bool = False, principal: Principal = Depends(require_principal)
) -> dict:
    """读取会话当前快照：next（待执行节点）与已生成答案。"""
    graph = get_session_graph(require_approval)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    return {
        "thread_id": thread_id,
        "next": list(snapshot.next),
        "answer": snapshot.values.get("result", ""),
    }
