"""filesystem 连接器与注册表测试（用临时目录，不连服务）。"""

from pathlib import Path

from app.connectors import RawDocument, build_connector
from app.sources import SourceConfig


def _fs_cfg(root: Path) -> SourceConfig:
    return SourceConfig(
        name="docs",
        type="filesystem",
        doc_type="doc",
        params={"root": str(root), "glob": "**/*.md", "exclude": ["README.md"]},
    )


def test_filesystem_connector_loads_md(tmp_path: Path):
    (tmp_path / "a.md").write_text("# 标题\n正文", encoding="utf-8")
    (tmp_path / "README.md").write_text("skip me", encoding="utf-8")
    docs = list(build_connector(_fs_cfg(tmp_path)).load())
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, RawDocument)
    assert doc.fmt == "md"
    assert doc.metadata["doc_type"] == "doc"
    assert doc.metadata["source"] == "a.md"
    assert doc.doc_id == "doc_a"  # 与第一轮 id 规则一致


def test_filesystem_connector_marks_format_by_ext(tmp_path: Path):
    (tmp_path / "note.txt").write_text("纯文本", encoding="utf-8")
    cfg = SourceConfig("d", "filesystem", "doc", {"root": str(tmp_path), "glob": "**/*.txt"})
    docs = list(build_connector(cfg).load())
    assert docs[0].fmt == "txt"


def test_build_connector_unknown_type_raises():
    cfg = SourceConfig("x", "nosuch", "doc", {})
    try:
        build_connector(cfg)
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_filesystem_connector_dedupes_multi_doc_per_file(tmp_path, monkeypatch):
    """同一文件多 Document（如 PDF 每页）必须产生唯一 doc_id（F-1）。"""
    import app.connectors.filesystem as fs

    class _Doc:
        def __init__(self, text, file_path, page):
            self.text = text
            self.metadata = {"file_path": file_path, "page_label": page}

    fp = str((tmp_path / "manual.pdf").resolve())

    class _Reader:
        def __init__(self, **kwargs):
            pass

        def load_data(self):
            return [_Doc("第一页", fp, "1"), _Doc("第二页", fp, "2")]

    monkeypatch.setattr(fs, "SimpleDirectoryReader", _Reader)
    cfg = SourceConfig("d", "filesystem", "doc", {"root": str(tmp_path), "glob": "**/*.pdf"})
    docs = list(fs.FilesystemConnector.from_config(cfg).load())
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)), f"doc_id 冲突: {ids}"
    assert all(i.startswith("doc_manual") for i in ids)
