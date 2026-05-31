"""测试夹具：每个测试前重置所有模块级单例，保证测试顺序无关。

本仓库多处用模块级单例缓存（模型客户端 / 向量索引 / BM25 / 图索引 / 编译图 /
reranker / checkpointer）。这些缓存若跨测试残留，会造成顺序相关的偶发失败
（如某测试用假对象填充单例后，后续测试读到脏缓存）。autouse 夹具在每个测试前
统一清零，使整套测试与执行顺序解耦。
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """每个测试前重置已知模块单例（导入失败的模块忽略）。"""

    def _clear() -> None:
        try:
            import app.models as _models

            _models._llm = None
            _models._embed_model = None
        except Exception:
            pass
        try:
            import app.index as _index

            _index._index = None
            _index._bm25_retrievers = {}
        except Exception:
            pass
        try:
            import app.rerankers as _rr

            _rr._reranker = None
        except Exception:
            pass
        try:
            import app.graph_index as _gi

            _gi._graph_index = None
        except Exception:
            pass
        try:
            import app.graph as _graph

            _graph._rag_graph = None
        except Exception:
            pass
        try:
            import app.agent as _agent

            _agent._agent_graph = None
        except Exception:
            pass

    _clear()
    yield
    _clear()
