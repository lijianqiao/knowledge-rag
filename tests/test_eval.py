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
