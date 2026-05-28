"""声明式源配置解析测试。"""

from pathlib import Path

from app.sources import SourceConfig, load_source_configs


def test_load_source_configs_parses_filesystem(tmp_path: Path):
    toml = tmp_path / "sources.toml"
    toml.write_text(
        '[[sources]]\n'
        'name = "docs"\n'
        'type = "filesystem"\n'
        'doc_type = "doc"\n'
        'root = "运维文档"\n'
        'glob = "**/*.md"\n'
        'exclude = ["README.md"]\n',
        encoding="utf-8",
    )
    cfgs = load_source_configs(toml)
    assert len(cfgs) == 1
    cfg = cfgs[0]
    assert isinstance(cfg, SourceConfig)
    assert cfg.name == "docs"
    assert cfg.type == "filesystem"
    assert cfg.doc_type == "doc"
    assert cfg.params["root"] == "运维文档"
    assert cfg.params["exclude"] == ["README.md"]


def test_load_source_configs_missing_file_raises(tmp_path: Path):
    try:
        load_source_configs(tmp_path / "nope.toml")
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_load_source_configs_rejects_missing_required_keys(tmp_path: Path):
    toml = tmp_path / "sources.toml"
    toml.write_text('[[sources]]\nname = "x"\n', encoding="utf-8")  # 缺 type/doc_type
    try:
        load_source_configs(toml)
        assert False, "应抛 ValueError"
    except ValueError:
        pass
