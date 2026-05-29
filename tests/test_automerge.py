"""auto-merging 父子索引：解析层级（纯逻辑离线）+ 装配/持久化（注入假模型）。"""

import json
from pathlib import Path

import app.automerge as am


def test_parse_hierarchical_makes_leaves_and_parents():
    # 纯逻辑：HierarchicalNodeParser 用 sentence splitter，不加载模型 → 离线可跑。
    text = "段落。" * 500  # 足够长以触发多层切分
    all_nodes, leaf = am.parse_hierarchical([text])
    assert len(leaf) >= 1
    assert len(all_nodes) > len(leaf)  # 存在父节点
    leaf_ids = {n.node_id for n in leaf}
    all_ids = {n.node_id for n in all_nodes}
    assert leaf_ids <= all_ids  # 叶子是全集子集


def test_automerge_store_exists_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "AUTO_MERGE_PERSIST_DIR", str(tmp_path / "nope"))
    assert am.automerge_store_exists() is False


def test_auto_merge_retrieve_empty_when_not_built(monkeypatch):
    monkeypatch.setattr(am, "automerge_store_exists", lambda: False)
    assert am.auto_merge_retrieve("任意问题", top_k=5) == []


def test_build_automerge_index_persists_docstore(tmp_path, monkeypatch):
    from llama_index.core.schema import TextNode

    persist_dir = tmp_path / "automerge_store"
    monkeypatch.setattr(am, "AUTO_MERGE_PERSIST_DIR", str(persist_dir))

    fake_all = [TextNode(text="父", id_="p1"), TextNode(text="子", id_="c1")]
    fake_leaf = [TextNode(text="子", id_="c1")]
    monkeypatch.setattr(am, "parse_hierarchical", lambda texts: (fake_all, fake_leaf))
    monkeypatch.setattr(am, "get_embed_model", lambda: object())

    captured = {}

    def _fake_build(leaf_nodes, storage_context):
        captured["n_leaf"] = len(leaf_nodes)
        captured["docstore_has_all"] = len(storage_context.docstore.docs)
        return object()

    monkeypatch.setattr(am, "_build_index", _fake_build)

    am.build_automerge_index(["原始文本"])

    assert captured["n_leaf"] == 1
    assert captured["docstore_has_all"] == 2  # 父+子都进了 docstore
    assert (persist_dir / "docstore.json").exists()
    data = json.loads((persist_dir / "docstore.json").read_text(encoding="utf-8"))
    assert data  # 非空
