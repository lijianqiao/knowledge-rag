"""CLI --source 动态来源测试。"""

import app.import_docs as import_mod


def test_resolve_source_names_all_means_empty():
    # "all" → 空列表（load_chunks 空列表 = 全部源）
    assert import_mod._resolve_source_names("all") == []


def test_resolve_source_names_specific():
    assert import_mod._resolve_source_names("docs") == ["docs"]


def test_run_import_rejects_unknown_source(monkeypatch):
    # load_chunks 抛 ValueError 应被 run_import 透传
    def _boom(names):
        raise ValueError("未知数据源: ghost")

    monkeypatch.setattr(import_mod, "load_chunks", _boom)
    try:
        import_mod.run_import(source="ghost")
        assert False, "应抛 ValueError"
    except ValueError:
        pass
