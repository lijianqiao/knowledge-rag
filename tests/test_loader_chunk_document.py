"""按格式分块与 chunk_id 稳定性测试。"""

from app.connectors import RawDocument
from app.loader import chunk_document


def test_md_uses_heading_chunking():
    text = "## 一\n" + "内容一" * 5 + "\n\n## 二\n" + "内容二" * 5
    doc = RawDocument(doc_id="doc_a", text=text, metadata={"doc_type": "doc"}, fmt="md")
    records = chunk_document(doc)
    assert records[0].chunk_id == "doc_a__0"
    assert all(r.metadata["chunk_total"] == len(records) for r in records)


def test_non_md_uses_window_split():
    doc = RawDocument(doc_id="doc_b", text="a" * 2000, metadata={"doc_type": "doc"}, fmt="pdf")
    records = chunk_document(doc)
    assert len(records) > 1
    assert all(len(r.text) <= 800 for r in records)
    assert records[1].chunk_id == "doc_b__1"


def test_metadata_merged_with_chunk_index():
    doc = RawDocument(doc_id="x", text="短", metadata={"doc_type": "doc", "title": "T"}, fmt="md")
    records = chunk_document(doc)
    assert records[0].metadata["title"] == "T"
    assert records[0].metadata["chunk_index"] == 0
