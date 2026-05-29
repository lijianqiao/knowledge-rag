import json

import app.trace as t


def test_trace_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "TRACE_DIR", str(tmp_path))
    tid = t.new_trace_id()
    t.log_event(tid, "route", {"choice": "vector"})
    t.log_event(tid, "answer", {"len": 42})
    lines = [json.loads(x) for x in (tmp_path / f"trace-{tid}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["event"] for l in lines] == ["route", "answer"]
    assert all(l["trace_id"] == tid for l in lines)
