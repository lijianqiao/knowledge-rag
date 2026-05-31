from pathlib import Path


def test_production_boundaries_doc_names_deferred_work():
    text = Path("docs/production-boundaries.md").read_text(encoding="utf-8")
    assert "Multimodal RAG" in text
    assert "speculative decoding" in text
    assert "Matryoshka" in text
    assert "not implemented in this app" in text
