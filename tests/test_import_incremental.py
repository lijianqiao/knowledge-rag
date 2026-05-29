"""增量导入：仅重导变更/新增文件，删除消失文件的 chunk。"""

import app.import_docs as im
from app.manifest import content_hash, load_manifest


class _Rec:
    """最小 ChunkRecord 替身（chunk_id / text / metadata）。"""

    def __init__(self, chunk_id, text, source):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = {"source": source}


def test_incremental_only_reimports_changed_and_deletes_gone(monkeypatch, tmp_path):
    records = [
        _Rec("fileA__0", "AAA-new", "运维文档/fileA.md"),
        _Rec("fileB__0", "BBB", "运维文档/fileB.md"),
    ]
    monkeypatch.setattr(im, "load_chunks", lambda names: records)

    # 临时清单：fileB 与本次内容同 hash（未变），fileC 现已消失
    manifest_path = tmp_path / ".rag_manifest.json"
    preexisting = {
        "运维文档/fileA.md": "OLD_HASH_A",
        "运维文档/fileB.md": content_hash("BBB"),
        "运维文档/fileC.md": "HASH_C_GONE",
    }
    manifest_path.write_text(__import__("json").dumps(preexisting, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(im, "MANIFEST_PATH", str(manifest_path))

    inserted_ids: list[str] = []
    removed_sources: list[str] = []
    monkeypatch.setattr(im, "_to_text_nodes", lambda recs: list(recs))
    monkeypatch.setattr(im, "insert_text_nodes", lambda nodes: inserted_ids.extend(n.chunk_id for n in nodes))
    monkeypatch.setattr(im, "delete_chunks_by_sources", lambda srcs: removed_sources.extend(srcs))

    n = im.run_import(incremental=True)

    # 仅 fileA 重导
    assert inserted_ids == ["fileA__0"]
    assert n == 1
    # 仅 fileC 删除
    assert removed_sources == ["运维文档/fileC.md"]
    # 清单写回 fileA + fileB（fileC 已移除）
    saved = load_manifest(manifest_path)
    assert set(saved) == {"运维文档/fileA.md", "运维文档/fileB.md"}
    assert saved["运维文档/fileB.md"] == content_hash("BBB")


def test_incremental_rejects_force(monkeypatch):
    monkeypatch.setattr(im, "load_chunks", lambda names: [])
    import pytest

    with pytest.raises(ValueError):
        im.run_import(force=True, incremental=True)
