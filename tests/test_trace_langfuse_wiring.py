"""trace.log_event 接入 Langfuse 转发的测试（默认关；离线，注入假 forward）。"""

import app.trace as t


def test_log_event_forwards_when_langfuse_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(t, "ENABLE_LANGFUSE", True)
    captured = []
    # _forward_langfuse 内部 from app.api.observability import forward_trace → 打桩该模块
    import app.api.observability as obs

    monkeypatch.setattr(obs, "forward_trace", lambda tid, events: captured.append((tid, events)))

    t.log_event("tid1", "retrieve", {"k": 1})
    assert captured == [("tid1", [{"event": "retrieve", "data": {"k": 1}}])]


def test_log_event_no_forward_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(t, "ENABLE_LANGFUSE", False)
    import app.api.observability as obs

    monkeypatch.setattr(obs, "forward_trace", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应转发")))

    t.log_event("tid2", "answer", {"len": 3})  # 关闭时不触达 forward_trace
    # 仍写了 JSONL
    assert (tmp_path / "trace-tid2.jsonl").exists()
