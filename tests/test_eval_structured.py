import json
from pathlib import Path

import app.eval as ev


def test_run_eval_json_output(tmp_path: Path):
    gold = tmp_path / "gold.json"
    gold.write_text(
        '[{"question":"q","expected_source_substrings":["a.md"]}]',
        encoding="utf-8",
    )

    report = ev.run_eval(
        gold,
        ask_fn=lambda q: ("answer", ["a.md"], "context"),
        faithfulness_fn=lambda a, c: 1.0,
        relevancy_fn=lambda a, q: 0.5,
        output_format="json",
    )

    data = json.loads(report)
    assert data["aggregate"]["recall"] == 1.0
    assert data["rows"][0]["sources"] == ["a.md"]
    assert data["rows"][0]["answer_chars"] == len("answer")
