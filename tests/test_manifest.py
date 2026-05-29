"""清单 diff：新增/变更/删除。"""

import app.manifest as mf


def test_diff_detects_changes():
    old = {"a.md": "h1", "b.md": "h2", "c.md": "h3"}
    new = {"a.md": "h1", "b.md": "H2new", "d.md": "h4"}
    changed, removed = mf.diff_manifest(old, new)
    assert set(changed) == {"b.md", "d.md"}  # 变更 + 新增
    assert set(removed) == {"c.md"}          # 消失
