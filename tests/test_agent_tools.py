import app.agent_tools as tools


def test_tool_schema_contains_vector_and_graph_tools():
    schemas = tools.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"vector_search", "graph_search"}


def test_dispatch_vector_search(monkeypatch):
    called = {}

    def fake_retrieve(query, top_k, allowed_sources=None):
        called["args"] = (query, top_k, allowed_sources)
        return []

    monkeypatch.setattr(tools, "retrieve_nodes", fake_retrieve)

    tools.dispatch_tool("vector_search", "redis", 3, allowed_sources=["a.md"])

    assert called["args"] == ("redis", 3, ["a.md"])


def test_dispatch_unknown_tool_raises():
    import pytest

    with pytest.raises(ValueError):
        tools.dispatch_tool("bad", "q", 5)


def test_agent_uses_json_fallback_when_native_tool_calling_disabled(monkeypatch):
    import app.agent as ag

    monkeypatch.setattr(ag, "ENABLE_NATIVE_TOOL_CALLING", False, raising=False)

    assert ag._decide_with_native_tools({"question": "q", "top_k": 5, "allowed_sources": None}) is None
