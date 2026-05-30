"""会话 / 人机循环端点（TestClient，全离线）。

monkeypatch 图节点的模型依赖（检索/生成/上下文），checkpointer 指向 tmp sqlite，
不连真实 LLM/embedding/网络。验证：直通 done、interrupt → resume → done。
"""

from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

import app.api.checkpoint as cp
import app.api.routes_session as rs
import app.graph as g
from app.api.app import create_app


def _patch_nodes(monkeypatch):
    """让图节点不调真实模型：假检索 + 固定答案 + 弱召回判否。"""
    fake_node = NodeWithScore(node=TextNode(text="资料", metadata={"source": "运维文档/x.md"}))
    monkeypatch.setattr(g, "retrieve_with_diagnostics", lambda **kw: ([fake_node], 0.9))
    monkeypatch.setattr(g, "build_context", lambda nodes: "上下文")
    monkeypatch.setattr(g, "format_nodes", lambda nodes: "运维文档/x.md")
    monkeypatch.setattr(g, "generate_answer", lambda q, ctx: "答案")
    monkeypatch.setattr(g, "is_retrieval_weak", lambda score: False)


def _client(monkeypatch, tmp_path):
    """全离线 client：tmp sqlite checkpointer + 重置图缓存。"""
    _patch_nodes(monkeypatch)
    monkeypatch.setattr(cp, "CHECKPOINT_DB", str(tmp_path / "s.sqlite"))
    cp.reset_checkpointer()
    rs.reset_session_graphs()
    return TestClient(create_app())


def test_session_done_without_approval(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/sessions", json={"question": "X", "require_approval": False})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert "答案" in body["answer"]
    assert body["thread_id"]
    cp.reset_checkpointer()
    rs.reset_session_graphs()


def test_session_interrupt_then_resume(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/sessions", json={"question": "X", "require_approval": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "interrupted"
    assert "generate" in body["pending"]
    tid = body["thread_id"]

    r2 = c.post(f"/sessions/{tid}/resume", json={})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["status"] == "done"
    assert "答案" in body2["answer"]

    r3 = c.get(f"/sessions/{tid}")
    assert r3.status_code == 200
    assert "答案" in r3.json()["answer"]

    cp.reset_checkpointer()
    rs.reset_session_graphs()
