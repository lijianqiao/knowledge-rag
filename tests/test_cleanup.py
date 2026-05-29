"""死代码清理与 ENABLE_AGENT 门控。"""

import pytest

import app.loader as loader
import main


def test_to_batches_removed():
    assert not hasattr(loader, "to_batches")


def test_agent_cli_gated(monkeypatch):
    """ENABLE_AGENT=false 时跑 agent 子命令应退出码 1（透传 ValueError）。"""
    monkeypatch.setattr(main, "run_agent", lambda *a, **k: "X")
    monkeypatch.setattr(main, "ENABLE_AGENT", False)
    monkeypatch.setattr("sys.argv", ["main.py", "agent", "q"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1
