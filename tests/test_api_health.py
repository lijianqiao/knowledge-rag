"""/health 端点（TestClient，离线）。"""

from fastapi.testclient import TestClient

from app.api.app import create_app


def test_health_ok():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
