"""跨文档推理 Agent：LangGraph 显式 plan→act→reflect 循环（不依赖原生 function-calling）。"""

import json
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from llama_index.core.schema import NodeWithScore

from app.config import AGENT_ANSWER_PROMPT, AGENT_DECIDE_PROMPT, MAX_AGENT_STEPS
from app.graph_index import graph_retrieve
from app.index import build_context, format_nodes, get_llm, retrieve_nodes
from app.trace import log_event, new_trace_id


class AgentState(TypedDict):
    question: str
    top_k: int
    step: int
    decision: dict
    evidence: list[NodeWithScore]
    answer: str
    trace_id: str
    allowed_sources: list | None


def _parse_decision(text: str) -> dict:
    """解析 LLM 决策 JSON；失败安全降级为 answer（R2）。"""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"action": "answer"}
    if data.get("action") not in ("search", "answer"):
        return {"action": "answer"}
    return data


def _evidence_text(evidence: list[NodeWithScore]) -> str:
    return build_context(evidence) if evidence else "（暂无）"


def _decide(state: AgentState) -> dict:
    prompt = AGENT_DECIDE_PROMPT.format(question=state["question"], evidence=_evidence_text(state["evidence"]))
    try:
        raw = str(get_llm().complete(prompt))
    except Exception:
        log_event(state.get("trace_id", ""), "decide", {"step": state["step"] + 1, "action": "answer"})
        return {"decision": {"action": "answer"}, "step": state["step"] + 1}
    decision = _parse_decision(raw)
    log_event(state.get("trace_id", ""), "decide", {"step": state["step"] + 1, "action": decision.get("action")})
    return {"decision": decision, "step": state["step"] + 1}


def _act(state: AgentState) -> dict:
    d = state["decision"]
    query = d.get("query") or state["question"]
    top_k = state["top_k"]
    # graph 路径无 source 过滤 → RBAC 为 best-effort（KB 关系图通常非敏感）；向量路径透传白名单。
    nodes = (
        graph_retrieve(query, top_k=top_k)
        if d.get("tool") == "graph"
        else retrieve_nodes(query, top_k=top_k, allowed_sources=state.get("allowed_sources"))
    )
    seen = {n.node.node_id for n in state["evidence"]}
    merged = list(state["evidence"]) + [n for n in nodes if n.node.node_id not in seen]
    log_event(
        state.get("trace_id", ""),
        "act",
        {"tool": d.get("tool"), "query": query, "evidence_count": len(merged)},
    )
    return {"evidence": merged}


def _answer(state: AgentState) -> dict:
    prompt = AGENT_ANSWER_PROMPT.format(question=state["question"], evidence=_evidence_text(state["evidence"]))
    try:
        ans = str(get_llm().complete(prompt)).strip()
    except Exception as exc:
        raise ValueError(f"Agent 生成答案失败: {exc}") from exc
    log_event(state.get("trace_id", ""), "answer", {"len": len(ans)})
    return {"answer": ans}


def _route_after_decide(state: AgentState) -> Literal["act", "answer"]:
    if state["step"] >= MAX_AGENT_STEPS:
        return "answer"
    return "act" if state["decision"].get("action") == "search" else "answer"


def build_agent_graph():
    b = StateGraph(AgentState)
    b.add_node("decide", _decide)
    b.add_node("act", _act)
    b.add_node("answer", _answer)
    b.add_edge(START, "decide")
    b.add_conditional_edges("decide", _route_after_decide)
    b.add_edge("act", "decide")
    b.add_edge("answer", END)
    return b.compile()


_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(question: str, top_k: int = 5, allowed_sources: list | None = None) -> str:
    """执行跨文档推理 Agent，返回答案 + 引用来源。

    allowed_sources：应用层 RBAC 白名单，透传至向量检索；graph 检索不支持 source 过滤，
    其 RBAC 为 best-effort。None 表示不限来源（行为不变）。
    """
    result = get_agent_graph().invoke(
        {
            "question": question,
            "step": 0,
            "decision": {},
            "evidence": [],
            "answer": "",
            "top_k": top_k,
            "trace_id": new_trace_id(),
            "allowed_sources": allowed_sources,
        }
    )
    return f"{result['answer']}\n\n--- 证据来源 ---\n{format_nodes(result['evidence'])}"
