"""埋点测试：run_ask / run_agent 是否在关键节点发出 log_event（全程离线）。"""

import app.agent as ag
import app.graph as g


def test_run_ask_emits_retrieve_and_answer(monkeypatch):
    from llama_index.core.schema import NodeWithScore, TextNode

    events: list[str] = []
    monkeypatch.setattr(g, "log_event", lambda tid, event, payload: events.append(event))

    node = NodeWithScore(node=TextNode(text="内容", id_="1"), score=0.9)
    # ENABLE_GRAPH 默认 false → route=vector，不调 classify_route
    monkeypatch.setattr(g, "retrieve_with_diagnostics", lambda query, top_k, doc_type: ([node], 0.9))
    monkeypatch.setattr(g, "build_context", lambda nodes: "ctx")
    monkeypatch.setattr(g, "format_nodes", lambda nodes: "src")
    monkeypatch.setattr(g, "generate_answer", lambda q, c: "答案")
    monkeypatch.setattr(g, "is_retrieval_weak", lambda score: False)  # 不触发重试

    out = g.run_ask("问题")
    assert "答案" in out
    assert "retrieve" in events
    assert "answer" in events


def test_run_agent_emits_decide_act_answer(monkeypatch):
    from llama_index.core.schema import NodeWithScore, TextNode

    events: list[str] = []
    monkeypatch.setattr(ag, "log_event", lambda tid, event, payload: events.append(event))

    class _LLM:
        def complete(self, prompt):
            if "只输出 JSON" in prompt:
                return '{"action": "search", "tool": "vector", "query": "Redis"}'
            return "答案 [1]"

    monkeypatch.setattr(ag, "get_llm", lambda: _LLM())
    # MAX_AGENT_STEPS=2：decide(step1,search)→act→decide(step2)→answer，确保 _act 真正执行
    monkeypatch.setattr(ag, "MAX_AGENT_STEPS", 2)
    monkeypatch.setattr(
        ag, "retrieve_nodes",
        lambda q, top_k, doc_type="all": [NodeWithScore(node=TextNode(text="内容", id_="1"), score=0.9)],
    )
    monkeypatch.setattr(ag, "graph_retrieve", lambda q, top_k: [])

    ag.run_agent("问题")
    assert "decide" in events
    assert "act" in events
    assert "answer" in events
