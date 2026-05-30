"""Langfuse 适配器（默认关，离线，SDK 未安装亦可测）。"""

import app.api.observability as obs


def test_forward_disabled_returns_false(monkeypatch):
    monkeypatch.setattr(obs, "ENABLE_LANGFUSE", False)

    def _boom():
        raise AssertionError("关闭时不应构造客户端")

    monkeypatch.setattr(obs, "_get_client", _boom)
    assert obs.forward_trace("t", []) is False


def test_forward_enabled_calls_client(monkeypatch):
    calls = []

    class _FakeClient:
        def event(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(obs, "ENABLE_LANGFUSE", True)
    monkeypatch.setattr(obs, "_get_client", lambda: _FakeClient())
    events = [
        {"event": "retrieve", "data": {"k": 5}},
        {"event": "answer", "data": {"len": 12}},
    ]
    assert obs.forward_trace("trace-123", events) is True
    assert len(calls) == 2
    assert all(c["trace_id"] == "trace-123" for c in calls)
    assert calls[0]["name"] == "retrieve" and calls[0]["metadata"] == {"k": 5}


def test_forward_enabled_no_sdk_returns_false(monkeypatch):
    monkeypatch.setattr(obs, "ENABLE_LANGFUSE", True)
    monkeypatch.setattr(obs, "_get_client", lambda: None)
    assert obs.forward_trace("t", [{"event": "x"}]) is False
