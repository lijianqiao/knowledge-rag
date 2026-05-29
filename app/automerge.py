"""父子 / auto-merging 索引：层级解析 + 持久化 + 合并检索（默认关闭）。

镜像 graph_index/graph_store 的隔离方式：把模型/Chroma 接触面收进可注入的内部
helper（_build_index / _retrieve_nodes），纯逻辑（parse_hierarchical）离线可测。

持久化方案：
- 父节点（含中间层）存独立 SimpleDocumentStore → docstore.json（AUTO_MERGE_PERSIST_DIR）。
- 叶子节点的向量存独立 chroma collection（f"{COLLECTION_NAME}_automerge"），不与主库冲突。
- AutoMergingRetriever 命中叶子后，经 storage_context.docstore 查父块按层级合并。

本任务仅提供模块 + 持久化；未接入 _do_retrieve / import 热路径（wiring 延后，见报告）。
"""

import os
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import AUTO_MERGE_CHUNK_SIZES, AUTO_MERGE_PERSIST_DIR, COLLECTION_NAME
from app.index import get_chroma_client, get_embed_model

_AUTOMERGE_COLLECTION = f"{COLLECTION_NAME}_automerge"
_DOCSTORE_FILE = "docstore.json"


def _automerge_collection():
    """auto-merge 专用 chroma collection（与主库 collection 隔离）。"""
    return get_chroma_client().get_or_create_collection(name=_AUTOMERGE_COLLECTION)


def parse_hierarchical(texts: list[str]) -> tuple[list, list]:
    """把文本切成层级节点，返回 (all_nodes, leaf_nodes)。

    纯逻辑：HierarchicalNodeParser 用句子切分器，不加载模型 → 离线可测。
    """
    docs = [Document(text=t) for t in texts]
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=AUTO_MERGE_CHUNK_SIZES)
    all_nodes = parser.get_nodes_from_documents(docs)
    leaf = get_leaf_nodes(all_nodes)
    return all_nodes, leaf


def automerge_store_exists() -> bool:
    """持久化目录是否已含构建产物（非空目录）。镜像 graph_store_exists。"""
    p = Path(AUTO_MERGE_PERSIST_DIR)
    return p.is_dir() and any(p.iterdir())


def _build_index(leaf_nodes, storage_context) -> VectorStoreIndex:
    """隔离真实构建调用（接触 embedding 模型 + Chroma），便于测试 monkeypatch。"""
    return VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(),
        show_progress=True,
    )


def build_automerge_index(texts: list[str]) -> None:
    """构建并持久化父子索引：叶子入 Chroma 向量库，全部节点入 docstore.json。"""
    all_nodes, leaf = parse_hierarchical(texts)

    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)  # 父+子全部，供合并时查父块

    vector_store = ChromaVectorStore(chroma_collection=_automerge_collection())
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )

    _build_index(leaf, storage_context)

    os.makedirs(AUTO_MERGE_PERSIST_DIR, exist_ok=True)
    docstore.persist(persist_path=os.path.join(AUTO_MERGE_PERSIST_DIR, _DOCSTORE_FILE))


def _retrieve_nodes(query: str, top_k: int) -> list[NodeWithScore]:
    """隔离真实检索调用（接触 embedding 模型 + Chroma），便于测试 monkeypatch。"""
    docstore = SimpleDocumentStore.from_persist_path(
        os.path.join(AUTO_MERGE_PERSIST_DIR, _DOCSTORE_FILE)
    )
    vector_store = ChromaVectorStore(chroma_collection=_automerge_collection())
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=get_embed_model()
    )
    base = index.as_retriever(similarity_top_k=top_k)
    retriever = AutoMergingRetriever(base, index.storage_context)
    return retriever.retrieve(query)


def auto_merge_retrieve(query: str, top_k: int) -> list[NodeWithScore]:
    """合并检索：未构建则返回 []（镜像 graph_retrieve 的安全回退）。"""
    if not automerge_store_exists():
        return []
    return _retrieve_nodes(query, top_k)
