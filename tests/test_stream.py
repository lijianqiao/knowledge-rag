"""流式输出：generate_answer_stream 增量分片 + run_ask_stream 末尾追加来源块（全程离线）。"""

import openai
import pytest

import app.graph as g
import app.index as m


class _StreamLLM:
    """假 LLM：stream_complete 产出带 .delta 的分片对象。"""

    def stream_complete(self, prompt):
        for delta in ("部分1", "部分2"):
            yield type("Resp", (), {"delta": delta})()


class _TimeoutLLM:
    """假 LLM：stream_complete 抛 APITimeoutError。"""

    def stream_complete(self, prompt):
        raise openai.APITimeoutError(request=None)
        yield  # pragma: no cover  使其成为生成器


def test_generate_answer_stream_concats(monkeypatch):
    monkeypatch.setattr(m, "get_llm", lambda: _StreamLLM())
    assert "".join(m.generate_answer_stream("问题", "上下文")) == "部分1部分2"


def test_generate_answer_stream_timeout_raises(monkeypatch):
    monkeypatch.setattr(m, "get_llm", lambda: _TimeoutLLM())
    with pytest.raises(ValueError):
        list(m.generate_answer_stream("问题", "上下文"))


def test_run_ask_stream_appends_sources(monkeypatch):
    monkeypatch.setattr(g, "_do_retrieve", lambda sq, tk, dt, allowed_sources=None: ([], 0.9, "ctx", "src行"))
    monkeypatch.setattr(g, "generate_answer_stream", lambda q, c: iter(["A", "B"]))

    out = list(g.run_ask_stream("问题"))
    assert out == ["A", "B", "\n\n--- 参考来源 ---\nsrc行"]
