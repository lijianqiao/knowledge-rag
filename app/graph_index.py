"""GraphRAG：PropertyGraphIndex 构建与图检索（复用本地 LLM/Embedding）。"""

from typing import Literal

from llama_index.core import PropertyGraphIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.schema import NodeWithScore, TextNode

from app.config import (
    GRAPH_ENTITIES,
    GRAPH_MAX_PATHS_PER_CHUNK,
    GRAPH_PERSIST_DIR,
    GRAPH_RELATIONS,
    GRAPH_RETRIEVE_TOP_K,
)
from app.graph_store import graph_store_exists
from app.index import get_embed_model, get_llm

_graph_index: PropertyGraphIndex | None = None


def _make_extractor() -> SchemaLLMPathExtractor:
    """运维领域 schema 抽取器（注入本地 LLM）。

    G-A：possible_entities/relations 形参类型是 Type[Any]，需传 Literal 类型，
    不能传 list[str]。Literal[tuple(...)] 在运行期等价于 Literal['a','b',...]。
    """
    return SchemaLLMPathExtractor(
        llm=get_llm(),
        possible_entities=Literal[tuple(GRAPH_ENTITIES)],
        possible_relations=Literal[tuple(GRAPH_RELATIONS)],
        max_triplets_per_chunk=GRAPH_MAX_PATHS_PER_CHUNK,
        strict=False,
    )


def _build_property_graph_index(nodes, kg_extractors, embed_model, show_progress):
    """隔离真实构建调用，便于测试 monkeypatch。

    不传 property_graph_store，让 PropertyGraphIndex 自建默认 SimplePropertyGraphStore +
    （因其不支持向量查询）独立 vector store 存 kg 节点 embedding。
    """
    return PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=kg_extractors,
        embed_model=embed_model,
        show_progress=show_progress,
    )


def _persist_index(index: PropertyGraphIndex) -> None:
    """G-C：持久化整个 StorageContext 目录（图存储 + embedding vector store + docstore）。"""
    index.storage_context.persist(persist_dir=GRAPH_PERSIST_DIR)


def build_graph_index(nodes: list[TextNode]) -> PropertyGraphIndex:
    """从 TextNode 抽取实体关系建图并持久化（graph-build 调用，慢、调 LLM）。"""
    index = _build_property_graph_index(
        nodes=nodes,
        kg_extractors=[_make_extractor()],
        embed_model=get_embed_model(),
        show_progress=True,
    )
    _persist_index(index)
    global _graph_index
    _graph_index = index
    return index


def get_graph_index() -> PropertyGraphIndex | None:
    """加载已构建的图索引（整目录恢复）；未构建返回 None（R5）。"""
    global _graph_index
    if _graph_index is None:
        if not graph_store_exists():
            return None
        storage_context = StorageContext.from_defaults(persist_dir=GRAPH_PERSIST_DIR)
        _graph_index = load_index_from_storage(
            storage_context, embed_model=get_embed_model(), llm=get_llm()
        )
    return _graph_index


def graph_retrieve(query: str, top_k: int = GRAPH_RETRIEVE_TOP_K) -> list[NodeWithScore]:
    """图检索：用索引默认检索器（同义词扩展 + 向量上下文，已正确接好内部 vector store）。

    G-B：不手搓 VectorContextRetriever（SimplePropertyGraphStore 不支持向量查询）。
    """
    index = get_graph_index()
    if index is None:
        return []
    return index.as_retriever(similarity_top_k=top_k, include_text=True).retrieve(query)


def reset_graph_index_cache() -> None:
    """清空图索引单例（重建后调用）。"""
    global _graph_index
    _graph_index = None
