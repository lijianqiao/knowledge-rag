"""图谱持久化目录存在性测试（用临时目录，不连模型）。"""

import app.graph_store as gs


def test_graph_store_missing_dir_false(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "GRAPH_PERSIST_DIR", str(tmp_path / "nope"))
    assert gs.graph_store_exists() is False


def test_graph_store_empty_dir_false(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setattr(gs, "GRAPH_PERSIST_DIR", str(d))
    assert gs.graph_store_exists() is False  # 空目录视为未构建


def test_graph_store_dir_with_files_true(tmp_path, monkeypatch):
    d = tmp_path / "g"
    d.mkdir()
    (d / "property_graph_store.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gs, "GRAPH_PERSIST_DIR", str(d))
    assert gs.graph_store_exists() is True
