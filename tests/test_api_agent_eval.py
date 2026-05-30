"""/agent（按 ENABLE_AGENT 门控）+ /eval 端点（TestClient，离线，monkeypatch 业务函数）。"""

from fastapi.testclient import TestClient

import app.api.routes_rag as rr
from app.api.app import create_app


def test_agent_gated_off(monkeypatch):
    monkeypatch.setattr(rr, "ENABLE_AGENT", False)
    c = TestClient(create_app())
    r = c.post("/agent", json={"question": "X"})
    assert r.status_code == 403


def test_agent_runs_when_enabled(monkeypatch):
    monkeypatch.setattr(rr, "ENABLE_AGENT", True)
    monkeypatch.setattr(rr, "run_agent", lambda q, top_k, allowed_sources: "agent答案")
    c = TestClient(create_app())
    r = c.post("/agent", json={"question": "X", "top_k": 6})
    assert r.status_code == 200
    assert r.json()["answer"] == "agent答案"


def test_eval_runs(monkeypatch):
    monkeypatch.setattr(rr, "run_eval", lambda path: "报告")
    c = TestClient(create_app())
    r = c.post("/eval", json={"goldset": "eval/goldset.example.json"})
    assert r.status_code == 200
    assert "报告" in r.json()["report"]
