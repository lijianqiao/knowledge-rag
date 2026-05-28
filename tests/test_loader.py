"""loader 分块纯逻辑测试。"""

from app.loader import chunk_markdown


def test_short_text_single_chunk():
    text = "## 标题\n短内容"
    chunks = chunk_markdown(text, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert "短内容" in chunks[0]


def test_empty_text_returns_empty():
    assert chunk_markdown("   ") == []


def test_long_section_is_window_split():
    body = "a" * 2000
    text = f"## 大段\n{body}"
    chunks = chunk_markdown(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_splits_on_h2_headings():
    text = "## 一\n内容一\n\n## 二\n内容二"
    chunks = chunk_markdown(text, chunk_size=20, overlap=5)
    joined = "\n".join(chunks)
    assert "内容一" in joined and "内容二" in joined
