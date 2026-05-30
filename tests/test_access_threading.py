"""把 allowed_sources 下推检索链路：_filter_by_access 兜底过滤（全程离线）。"""

import app.index as m
from llama_index.core.schema import NodeWithScore, TextNode


def test_filter_by_access_keeps_allowed():
    nodes = [
        NodeWithScore(node=TextNode(text="a", metadata={"source": "运维文档/x.md"})),
        NodeWithScore(node=TextNode(text="b", metadata={"source": "私密/y.md"})),
    ]
    out = m._filter_by_access(nodes, ["运维文档/x.md"])
    assert [n.get_content() for n in out] == ["a"]


def test_filter_by_access_none_passthrough():
    nodes = [NodeWithScore(node=TextNode(text="a", metadata={"source": "z"}))]
    assert m._filter_by_access(nodes, None) == nodes
