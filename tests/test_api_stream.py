def test_ask_stream_sse(monkeypatch):
    import app.api.routes_rag as rr
    monkeypatch.setattr(rr, "run_ask_stream", lambda q, top_k, doc_type, allowed_sources: iter(["A", "B", "\n\n--- 参考来源 ---\nsrc"]))
    from fastapi.testclient import TestClient
    from app.api.app import create_app
    c = TestClient(create_app())
    with c.stream("POST", "/ask/stream", json={"question": "X"}) as r:
        body = "".join(chunk for chunk in r.iter_text())
    assert "A" in body and "B" in body and "参考来源" in body
