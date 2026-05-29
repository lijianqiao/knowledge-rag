"""eval CLI 运行器：run_eval 用注入的 ask_fn / judge 函数，离线产出报告。"""

import json

import app.eval as ev


def _write_goldset(tmp_path):
    rows = [
        {"question": "订单服务 502 怎么排查", "expected_source_substrings": ["502"]},
        {"question": "Redis 内存溢出如何处理", "expected_source_substrings": ["redis"]},
    ]
    path = tmp_path / "goldset.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_run_eval_report(tmp_path):
    path = _write_goldset(tmp_path)

    def fake_ask(question):
        # 返回 (answer, sources, context)，命中各自期望子串
        if "502" in question:
            return "排查步骤……", ["运维文档/order-502.md"], "ctx502"
        return "调大内存……", ["运维文档/redis.md"], "ctxredis"

    report = ev.run_eval(
        path,
        ask_fn=fake_ask,
        faithfulness_fn=lambda answer, context: 1.0,
        relevancy_fn=lambda answer, question: 0.5,
    )

    assert isinstance(report, str)
    # 两个问题都出现
    assert "订单服务 502 怎么排查" in report
    assert "Redis 内存溢出如何处理" in report
    # 聚合分（recall 全命中 = 1.0，faithfulness 1.0，relevancy 0.5）
    assert "recall" in report
    assert "1.0" in report
    assert "0.5" in report
