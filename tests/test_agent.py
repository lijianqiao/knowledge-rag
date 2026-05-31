"""Agent 循环控制纯逻辑测试（注入假 LLM/工具）。"""

import app.agent as ag


def test_parse_decision_search():
    d = ag._parse_decision('{"action": "search", "tool": "graph", "query": "X 的根因"}')
    assert d == {"action": "search", "tool": "graph", "query": "X 的根因"}


def test_parse_decision_answer():
    assert ag._parse_decision('{"action": "answer"}')["action"] == "answer"


def test_parse_decision_garbage_defaults_answer():
    # 解析失败 → 安全终止（R2）
    assert ag._parse_decision("这不是JSON")["action"] == "answer"


def test_should_continue_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(ag, "MAX_AGENT_STEPS", 2)
    ev = [object()]  # 非空证据
    # 步数上限优先：即便有证据/想继续，达上限也终止
    assert ag._route_after_decide({"step": 2, "evidence": ev, "decision": {"action": "search"}}) == "answer"
    # 有证据：按决策走
    assert ag._route_after_decide({"step": 1, "evidence": ev, "decision": {"action": "search"}}) == "act"
    assert ag._route_after_decide({"step": 1, "evidence": ev, "decision": {"action": "answer"}}) == "answer"


def test_route_forces_initial_retrieval_when_no_evidence(monkeypatch):
    """实跑调参（R2）：尚无证据时，即便模型误选 answer 也强制先 act 一次。"""
    monkeypatch.setattr(ag, "MAX_AGENT_STEPS", 4)
    assert ag._route_after_decide({"step": 1, "evidence": [], "decision": {"action": "answer"}}) == "act"


def test_act_appends_unique_evidence(monkeypatch):
    from llama_index.core.schema import NodeWithScore, TextNode

    n = NodeWithScore(node=TextNode(text="订单服务依赖Redis", id_="1"), score=0.9)
    monkeypatch.setattr(ag, "retrieve_nodes", lambda q, top_k, doc_type="all", allowed_sources=None: [n])
    monkeypatch.setattr(ag, "graph_retrieve", lambda q, top_k: [])

    state = {"evidence": [], "decision": {"action": "search", "tool": "vector", "query": "Redis"}, "top_k": 5}
    out = ag._act(state)
    assert len(out["evidence"]) == 1
    # 再次相同节点不重复累计
    state2 = {"evidence": out["evidence"], "decision": {"action": "search", "tool": "vector", "query": "Redis"}, "top_k": 5}
    out2 = ag._act(state2)
    assert len(out2["evidence"]) == 1


def test_run_agent_returns_answer(monkeypatch):
    from llama_index.core.schema import NodeWithScore, TextNode

    # LLM 第一次说 search，第二次（达上限后）走 answer 提示词产出答案
    calls = {"n": 0}

    class _LLM:
        def complete(self, prompt):
            if "只输出 JSON" in prompt:
                calls["n"] += 1
                return '{"action": "search", "tool": "vector", "query": "Redis"}'
            return "订单服务依赖 Redis [1]"

    monkeypatch.setattr(ag, "get_llm", lambda: _LLM())
    monkeypatch.setattr(ag, "MAX_AGENT_STEPS", 1)
    monkeypatch.setattr(
        ag, "retrieve_nodes",
        lambda q, top_k, doc_type="all", allowed_sources=None: [NodeWithScore(node=TextNode(text="订单服务依赖Redis", id_="1"), score=0.9)],
    )
    monkeypatch.setattr(ag, "graph_retrieve", lambda q, top_k: [])

    out = ag.run_agent("订单服务依赖什么")
    assert "Redis" in out
