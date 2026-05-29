"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: index.py
@DateTime: 2026-05-28
@Docs: LlamaIndex 索引层：Embedding、ChromaDB 向量库、检索与 context 组装
"""

import chromadb
import jieba
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.config import (
    API_KEY,
    BM25_TOP_K,
    CHAT_BASE_URL,
    CHAT_MODEL,
    COLLECTION_NAME,
    CONTEXT_CHUNK_MAX_CHARS,
    DB_PATH,
    EMBED_BASE_URL,
    EMBED_MODEL,
    ENABLE_HYBRID,
    ENABLE_MULTI_QUERY,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    MULTI_QUERY_NUM,
    MULTI_QUERY_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_SYSTEM_PROMPT,
    RETRIEVE_CANDIDATE_K,
    RETRIEVE_SCORE_THRESHOLD,
)
from app.rerankers import get_reranker

_embed_model: OpenAIEmbedding | None = None
_llm: OpenAILike | None = None
_index: VectorStoreIndex | None = None
_bm25_retrievers: dict[str, BM25Retriever] = {}


def get_embed_model() -> OpenAIEmbedding:
    """获取 LlamaIndex Embedding 模型（连接本地 llama.cpp）。"""
    global _embed_model
    if _embed_model is None:
        _embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            model_name=EMBED_MODEL,
            api_base=EMBED_BASE_URL,
            api_key=API_KEY,
        )
    return _embed_model


def get_llm() -> OpenAILike:
    """获取 OpenAI 兼容 Chat 模型（连接本地 llama.cpp）。"""
    global _llm
    if _llm is None:
        _llm = OpenAILike(
            model=CHAT_MODEL,
            api_base=CHAT_BASE_URL,
            api_key=API_KEY,
            is_chat_model=True,
            temperature=0.2,
            context_window=8192,
            timeout=LLM_TIMEOUT,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm


def reset_index_cache() -> None:
    """清空索引缓存（重建 collection 后调用）。"""
    global _index, _bm25_retrievers
    _index = None
    _bm25_retrievers = {}


def get_chroma_client() -> chromadb.ClientAPI:
    """获取 ChromaDB 持久化客户端。"""
    return chromadb.PersistentClient(path=DB_PATH)


def get_chroma_collection():
    """获取 ChromaDB collection。"""
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def delete_chroma_collection() -> bool:
    """
    删除 ChromaDB collection。

    Returns:
        是否执行了删除
    """
    client = get_chroma_client()
    names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME not in names:
        return False
    client.delete_collection(COLLECTION_NAME)
    reset_index_cache()
    return True


def get_index() -> VectorStoreIndex:
    """
    获取 LlamaIndex 向量索引（绑定已有 ChromaDB collection）。

    Returns:
        VectorStoreIndex 实例
    """
    global _index
    if _index is None:
        vector_store = ChromaVectorStore(chroma_collection=get_chroma_collection())
        _index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=get_embed_model(),
        )
    return _index


def _build_metadata_filters(doc_type: str) -> MetadataFilters | None:
    """构建文档类型过滤条件。"""
    if doc_type == "all":
        return None
    return MetadataFilters(filters=[MetadataFilter(key="doc_type", value=doc_type)])


def _jieba_tokenize(text: str) -> list[str]:
    """中文分词器，供 BM25 使用（B2：默认英文分词对中文无效）。"""
    return [tok for tok in jieba.lcut(text) if tok.strip()]


def nodes_from_chroma_payload(payload: dict) -> list[TextNode]:
    """把 chroma collection.get() 的返回映射为 TextNode（纯函数）。"""
    ids = payload.get("ids") or []
    docs = payload.get("documents") or []
    metas = payload.get("metadatas") or []
    nodes: list[TextNode] = []
    for id_, doc, meta in zip(ids, docs, metas):
        nodes.append(TextNode(text=doc, id_=id_, metadata=meta or {}))
    return nodes


def load_all_nodes() -> list[TextNode]:
    """从 ChromaDB 重建全部节点，供 BM25 使用。"""
    payload = get_chroma_collection().get(include=["documents", "metadatas"])
    return nodes_from_chroma_payload(payload)


def get_bm25_retriever(doc_type: str = "all") -> BM25Retriever:
    """按 doc_type 构建并缓存 BM25（严格过滤，F3 升级；B2：jieba 中文分词）。"""
    if doc_type not in _bm25_retrievers:
        nodes = load_all_nodes()
        if doc_type != "all":
            nodes = [n for n in nodes if (n.metadata or {}).get("doc_type") == doc_type]
        _bm25_retrievers[doc_type] = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=BM25_TOP_K,
            tokenizer=_jieba_tokenize,
        )
    return _bm25_retrievers[doc_type]


def apply_rerank(
    reranker,
    query: str,
    nodes: list[NodeWithScore],
    top_n: int,
) -> list[NodeWithScore]:
    """对召回节点重排并截断到 top_n；reranker 为 None 时仅按原序截断。

    切片交由 reranker.rerank 负责（其契约保证返回条数等于 top_n）。
    """
    if not nodes:
        return []
    if reranker is None:
        return nodes[:top_n]
    return reranker.rerank(query, nodes, top_n)


def _filter_by_doc_type(nodes: list[NodeWithScore], doc_type: str) -> list[NodeWithScore]:
    """按 doc_type 过滤节点（F3：BM25 通道不走 Chroma metadata 过滤，需在融合后兜底）。"""
    if doc_type == "all":
        return nodes
    return [node for node in nodes if (node.metadata or {}).get("doc_type") == doc_type]


def _retrieve_core(
    query: str,
    top_k: int,
    doc_type: str,
    *,
    want_vector_score: bool,
) -> tuple[list[NodeWithScore], float | None]:
    """
    检索核心：宽召回 →（Hybrid）稠密+BM25 融合 → 按类型过滤 → cross-encoder 重排。

    Args:
        query: 检索文本
        top_k: 返回条数
        doc_type: 文档类型过滤
        want_vector_score: 是否额外计算向量召回最高余弦分（F1：weak 判断用，避免量纲混淆）

    Returns:
        (最终节点, 向量召回最高余弦分或 None)

    Raises:
        ValueError: 向量库为空时
    """
    if get_chroma_collection().count() == 0:
        raise ValueError("向量库为空，请先执行 import")

    candidate_k = max(top_k, RETRIEVE_CANDIDATE_K)
    vector_retriever = get_index().as_retriever(
        similarity_top_k=candidate_k,
        filters=_build_metadata_filters(doc_type),
    )

    best_vector_score: float | None = None
    if ENABLE_HYBRID:
        # Hybrid 下融合分是 RRF，量纲与余弦不同；weak 判断需单独取一次向量召回的余弦分
        if want_vector_score:
            vector_nodes = vector_retriever.retrieve(query)
            best_vector_score = max((n.score or 0.0) for n in vector_nodes) if vector_nodes else 0.0
        num_queries = MULTI_QUERY_NUM if ENABLE_MULTI_QUERY else 1
        retriever = QueryFusionRetriever(
            [vector_retriever, get_bm25_retriever(doc_type)],
            llm=get_llm(),            # S2：显式传 LLM，避免回退到未配置的 Settings.llm
            similarity_top_k=candidate_k,
            num_queries=num_queries,
            query_gen_prompt=MULTI_QUERY_PROMPT,  # F-3：中文扩展提示词（num_queries=1 时不生效，无害）
            mode="reciprocal_rerank", # RRF 融合
            use_async=ENABLE_MULTI_QUERY,  # 多查询时并行
        )
        nodes = retriever.retrieve(query)
    else:
        nodes = vector_retriever.retrieve(query)
        best_vector_score = max((n.score or 0.0) for n in nodes) if nodes else 0.0

    nodes = _filter_by_doc_type(nodes, doc_type)
    final = apply_rerank(get_reranker(), query, nodes, top_n=top_k)
    return final, best_vector_score


def retrieve_nodes(
    query: str,
    top_k: int = 5,
    doc_type: str = "all",
) -> list[NodeWithScore]:
    """通用检索入口（CLI query / 图首检索复用）。"""
    nodes, _ = _retrieve_core(query, top_k, doc_type, want_vector_score=False)
    return nodes


def retrieve_with_diagnostics(
    query: str,
    top_k: int = 5,
    doc_type: str = "all",
) -> tuple[list[NodeWithScore], float]:
    """
    图重检索用：额外返回向量召回最高余弦分，供 weak 判断（F1）。

    Returns:
        (最终节点, 向量召回最高余弦分)
    """
    nodes, best_vector_score = _retrieve_core(query, top_k, doc_type, want_vector_score=True)
    return nodes, best_vector_score or 0.0


def build_context(nodes: list[NodeWithScore]) -> str:
    """
    LlamaIndex 组装 RAG context。

    Args:
        nodes: 检索结果节点

    Returns:
        供 LLM 使用的参考资料文本
    """
    blocks: list[str] = []
    for index, node in enumerate(nodes, start=1):
        meta = node.metadata or {}
        header = f"[{index}] {meta.get('title', '未知')} — {meta.get('source', '未知')}"
        content = node.get_content().strip()
        if len(content) > CONTEXT_CHUNK_MAX_CHARS:
            content = content[:CONTEXT_CHUNK_MAX_CHARS].rstrip() + "..."
        blocks.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(blocks)


def format_nodes(nodes: list[NodeWithScore]) -> str:
    """格式化检索结果为调试输出。"""
    lines: list[str] = []
    for index, node in enumerate(nodes, start=1):
        meta = node.metadata or {}
        score = node.score if node.score is not None else 0.0
        lines.append(
            f"{index}. [{meta.get('doc_type', '?')}] {meta.get('title', '?')} "
            f"| {meta.get('source', '?')} "
            f"| chunk {meta.get('chunk_index', '?')}/{meta.get('chunk_total', '?')} "
            f"| 分数: {score:.4f}"
        )
    return "\n".join(lines)


def is_retrieval_weak(best_vector_score: float) -> bool:
    """
    判断检索结果是否偏弱，用于 LangGraph 重检索分支。

    Args:
        best_vector_score: 向量召回最高余弦分（F1：必须是余弦分，而非融合/重排分）

    Returns:
        是否需要扩大检索
    """
    return best_vector_score < RETRIEVE_SCORE_THRESHOLD


def generate_answer(question: str, context: str) -> str:
    """
    调用 LLM 基于 context 生成答案。

    Args:
        question: 用户问题
        context: 检索组装的参考资料

    Returns:
        模型回答
    """
    llm = get_llm()
    prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"参考资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请回答："
    )
    try:
        response = llm.complete(prompt)
    except APITimeoutError as exc:
        raise ValueError(
            f"Chat 请求超时（>{LLM_TIMEOUT}s）。本地模型生成较慢，"
            f"可增大 .env 中 LLM_TIMEOUT 或减小 LLM_MAX_TOKENS。详情: {exc}"
        ) from exc
    except (APIConnectionError, APIStatusError) as exc:
        raise ValueError(
            f"Chat 服务不可用 ({CHAT_BASE_URL})，请确认 llama.cpp 对话模型已启动: {exc}"
        ) from exc
    return str(response).strip()


def rewrite_query(question: str, prev_query: str) -> str:
    """LLM 改写检索查询；空输出时回退原始问题。"""
    prompt = QUERY_REWRITE_PROMPT.format(question=question, prev_query=prev_query)
    try:
        rewritten = str(get_llm().complete(prompt)).strip()
    except Exception:
        return question
    return rewritten or question


def insert_text_nodes(nodes: list[TextNode]) -> None:
    """
    将分块节点写入向量库。

    Args:
        nodes: TextNode 列表
    """
    index = get_index()
    index.insert_nodes(nodes)


def get_status() -> str:
    """返回向量库状态摘要。"""
    client = get_chroma_client()
    names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME not in names:
        return f"数据库: {DB_PATH}\ncollection '{COLLECTION_NAME}' 不存在，请先 import"

    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    sample = collection.get(limit=min(total, 200), include=["metadatas"])

    type_counts: dict[str, int] = {}
    for meta in sample.get("metadatas", []):
        doc_type = meta.get("doc_type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

    lines = [f"数据库: {DB_PATH}", f"collection: {COLLECTION_NAME}", f"chunk 总数: {total}"]
    for doc_type, count in sorted(type_counts.items()):
        lines.append(f"  - {doc_type}: {count}（采样）")
    return "\n".join(lines)
