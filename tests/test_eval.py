"""评估纯逻辑：context_recall 与报告聚合。"""

import app.eval as ev


def test_context_recall_full():
    retrieved_sources = ["运维文档/redis.md", "运维文档/order.md"]
    expected = ["redis", "nginx"]
    # 命中 1/2
    assert ev.context_recall(retrieved_sources, expected) == 0.5


def test_context_recall_empty_expected_is_one():
    assert ev.context_recall(["a"], []) == 1.0  # 无期望来源视为满分（不惩罚）


def test_aggregate_means():
    rows = [{"recall": 1.0, "faithfulness": 1.0, "relevancy": 0.0},
            {"recall": 0.0, "faithfulness": 1.0, "relevancy": 1.0}]
    agg = ev.aggregate(rows)
    assert agg["recall"] == 0.5 and agg["faithfulness"] == 1.0 and agg["relevancy"] == 0.5


def test_judge_faithfulness_parses_score(monkeypatch):
    import app.eval as ev
    monkeypatch.setattr(ev, "get_llm", lambda: type("L", (), {"complete": lambda s, p: "0.8"})())
    assert ev.judge_faithfulness("答案", "上下文") == 0.8


def test_judge_handles_garbage(monkeypatch):
    import app.eval as ev
    monkeypatch.setattr(ev, "get_llm", lambda: type("L", (), {"complete": lambda s, p: "无法判断"})())
    assert ev.judge_faithfulness("答案", "上下文") == 0.0  # 解析失败给 0


def test_score_ignores_irrelevant_numbers(monkeypatch):
    import app.eval as ev
    # 含无关数字「2025」，应取末尾的真实分 0.9，而非匹配到 0
    monkeypatch.setattr(ev, "get_llm", lambda: type("L", (), {"complete": lambda s, p: "依据2025年数据，分数：0.9"})())
    assert ev.judge_faithfulness("答案", "上下文") == 0.9
