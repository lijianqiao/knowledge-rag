"""graph-build 编排测试（注入假构建，不联网）。"""

import app.import_docs as im


def test_run_graph_build_uses_same_nodes(monkeypatch):
    from app.loader import ChunkRecord

    records = [ChunkRecord(chunk_id="doc_a__0", text="订单服务依赖Redis", metadata={"doc_type": "doc"})]
    monkeypatch.setattr(im, "load_chunks", lambda names: records)

    captured = {}
    monkeypatch.setattr(im, "build_graph_index", lambda nodes: captured.update(n=len(nodes)))

    im.run_graph_build(source="all")
    assert captured["n"] == 1


def test_run_graph_build_empty_raises(monkeypatch):
    monkeypatch.setattr(im, "load_chunks", lambda names: [])
    # load_chunks 自身对空已抛 ValueError；此处模拟其向上透传
    monkeypatch.setattr(im, "load_chunks", lambda names: (_ for _ in ()).throw(ValueError("无文档")))
    try:
        im.run_graph_build(source="all")
        assert False, "应抛 ValueError"
    except ValueError:
        pass
