"""SQLite checkpointer 封装（真实建 SqliteSaver，仅 sqlite，不连模型/网络）。"""

from langgraph.checkpoint.sqlite import SqliteSaver

import app.api.checkpoint as cp


def test_get_checkpointer_singleton(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "CHECKPOINT_DB", str(tmp_path / "s.sqlite"))
    cp.reset_checkpointer()
    first = cp.get_checkpointer()
    assert isinstance(first, SqliteSaver)
    assert cp.get_checkpointer() is first  # 同一实例
    cp.reset_checkpointer()
