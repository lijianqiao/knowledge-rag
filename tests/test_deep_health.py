import app.health as h


def test_deep_health_reports_components(monkeypatch):
    monkeypatch.setattr(h, "_check_chroma", lambda: {"ok": True, "count": 3})
    monkeypatch.setattr(h, "_check_graph", lambda: {"ok": False, "reason": "missing"})
    monkeypatch.setattr(h, "_check_models", lambda: {"ok": True})

    result = h.deep_health()

    assert result["ok"] is False
    assert result["components"]["chroma"]["count"] == 3
    assert result["components"]["graph"]["reason"] == "missing"


def test_deep_health_api(monkeypatch):
    from fastapi.testclient import TestClient

    import app.api.app as api_app

    monkeypatch.setattr(api_app, "deep_health", lambda: {"ok": True, "components": {}})
    client = TestClient(api_app.create_app())

    assert client.get("/health/deep").json()["ok"] is True


def test_health_cli_prints_json(monkeypatch, capsys):
    import app.health as health
    import main

    monkeypatch.setattr(health, "deep_health", lambda: {"ok": True, "components": {}})
    monkeypatch.setattr("sys.argv", ["main.py", "health"])

    main.main()

    output = capsys.readouterr().out
    assert '"ok": true' in output
