from llama_index.core.schema import NodeWithScore, TextNode

import app.index as idx


def test_retrieve_uses_automerge_when_enabled_and_store_exists(monkeypatch):
    expected = [
        NodeWithScore(
            node=TextNode(text="parent answer", metadata={"source": "a.md"}),
            score=0.9,
        )
    ]

    monkeypatch.setattr(idx, "ENABLE_AUTO_MERGE", True, raising=False)
    monkeypatch.setattr(idx, "automerge_store_exists", lambda: True, raising=False)
    monkeypatch.setattr(
        idx,
        "auto_merge_retrieve",
        lambda query, top_k: expected,
        raising=False,
    )

    assert idx.retrieve_nodes("question", top_k=3) == expected


def test_force_import_builds_automerge_when_enabled(monkeypatch):
    import app.import_docs as imp
    from app.loader import ChunkRecord

    records = [
        ChunkRecord(
            chunk_id="doc_a__0",
            text="alpha text",
            metadata={"source": "a.md", "doc_type": "doc"},
        )
    ]
    built = {}

    monkeypatch.setattr(imp, "ENABLE_AUTO_MERGE", True, raising=False)
    monkeypatch.setattr(imp, "load_chunks", lambda names: records)
    monkeypatch.setattr(imp, "delete_chroma_collection", lambda: True)
    monkeypatch.setattr(imp, "reset_index_cache", lambda: None)
    monkeypatch.setattr(imp, "insert_text_nodes", lambda nodes: None)
    monkeypatch.setattr(
        imp,
        "build_automerge_index",
        lambda texts: built.setdefault("texts", texts),
        raising=False,
    )

    assert imp.run_import(force=True) == 1
    assert built["texts"] == ["alpha text"]
