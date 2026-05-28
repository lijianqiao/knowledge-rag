"""查询改写编排测试（注入假 LLM）。"""

import app.index as index_mod
from app.index import rewrite_query


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt):
        # 校验提示词带上了原始问题
        assert "原始问题" in prompt
        return self._text


def test_rewrite_query_returns_trimmed(monkeypatch):
    monkeypatch.setattr(index_mod, "get_llm", lambda: _FakeLLM("  重启 nginx 服务 命令  "))
    out = rewrite_query("nginx咋重启", "nginx咋重启")
    assert out == "重启 nginx 服务 命令"


def test_rewrite_query_falls_back_on_empty(monkeypatch):
    monkeypatch.setattr(index_mod, "get_llm", lambda: _FakeLLM("   "))
    out = rewrite_query("原问题", "上次查询")
    assert out == "原问题"  # 模型空输出时回退到原始问题
