from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import app.api.auth as auth


def _app(monkeypatch, keys):
    monkeypatch.setattr(auth, "API_KEYS", keys)
    app = FastAPI()

    @app.get("/whoami")
    def whoami(p: auth.Principal = Depends(auth.require_principal)):
        return {"user": p.user, "sources": p.allowed_sources}

    return TestClient(app, raise_server_exceptions=True)


def test_open_mode_when_no_keys(monkeypatch):
    c = _app(monkeypatch, {})
    r = c.get("/whoami")  # 无 key 配置 → 开放模式
    assert r.status_code == 200
    assert r.json()["sources"] is None  # None = 全部可见


def test_valid_key(monkeypatch):
    c = _app(monkeypatch, {"sk-a": {"user": "alice", "allowed_sources": ["运维文档"]}})
    r = c.get("/whoami", headers={"X-API-Key": "sk-a"})
    assert r.json() == {"user": "alice", "sources": ["运维文档"]}


def test_invalid_key_401(monkeypatch):
    c = _app(monkeypatch, {"sk-a": {"user": "alice", "allowed_sources": None}})
    r = c.get("/whoami", headers={"X-API-Key": "bad"})
    assert r.status_code == 401
