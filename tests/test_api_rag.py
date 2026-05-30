"""RAG 端点 /ask /query /status（TestClient，离线，monkeypatch 业务函数）。"""

from fastapi.testclient import TestClient

import app.api.routes_rag as rr
from app.api.app import create_app


def test_ask_calls_run_ask_with_allowed_sources(monkeypatch):
    captured = {}
    monkeypatch.setattr(rr, "run_ask", lambda q, top_k, doc_type, allowed_sources: captured.update(
        q=q, top_k=top_k, doc_type=doc_type, allowed_sources=allowed_sources) or "答案+来源")
    c = TestClient(create_app())
    r = c.post("/ask", json={"question": "X", "top_k": 3, "doc_type": "all"})
    assert r.status_code == 200
    assert r.json()["answer"] == "答案+来源"
    assert captured["q"] == "X" and captured["top_k"] == 3
    assert captured["allowed_sources"] is None  # 开放模式


def test_status_ok(monkeypatch):
    monkeypatch.setattr(rr, "get_status", lambda: "状态摘要")
    c = TestClient(create_app())
    assert c.get("/status").json()["status"] == "状态摘要"
