"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: index.py
@DateTime: 2026-05-28
@Docs: LlamaIndex 索引层：Embedding、ChromaDB 向量库、检索与 context 组装
"""

import re

import bm25s
import chromadb
import jieba
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.utils import node_to_metadata_dict
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
    CHAT_PROVIDER,
    CLOUD_CHAT_API_KEY,
    CLOUD_CHAT_BASE_URL,
    CLOUD_CHAT_MODEL,
    CLOUD_EMBED_API_KEY,
    CLOUD_EMBED_BASE_URL,
    CLOUD_EMBED_MODEL,
    COLLECTION_NAME,
    CONTEXT_CHUNK_MAX_CHARS,
    DB_PATH,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
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
    """获取 LlamaIndex Embedding 模型（按 EMBED_PROVIDER 选 local/cloud）。"""
    global _embed_model
    if _embed_model is None:
        if EMBED_PROVIDER == "cloud":
            base, model, key = CLOUD_EMBED_BASE_URL, CLOUD_EMBED_MODEL, CLOUD_EMBED_API_KEY
        else:
            base, model, key = EMBED_BASE_URL, EMBED_MODEL, API_KEY
        _embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",  # 占位，真实模型由 model_name 决定
            model_name=model, api_base=base, api_key=key,
        )
    return _embed_model


def get_llm() -> OpenAILike:
    """获取 OpenAI 兼容 Chat 模型（按 CHAT_PROVIDER 选 local/cloud）。"""
    global _llm
    if _llm is None:
        if CHAT_PROVIDER == "cloud":
            base, model, key = CLOUD_CHAT_BASE_URL, CLOUD_CHAT_MODEL, CLOUD_CHAT_API_KEY
        else:
            base, model, key = CHAT_BASE_URL, CHAT_MODEL, API_KEY
        _llm = OpenAILike(
            model=model, api_base=base, api_key=key,
            is_chat_model=True, temperature=0.2, context_window=8192,
            timeout=LLM_TIMEOUT, max_tokens=LLM_MAX_TOKENS,
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


def delete_chunks_by_sources(sources: list[str]) -> None:
    """按 source 元数据删除 chunk（增量导入清理消失的文件）。"""
    if not sources:
        return
    collection = get_chroma_collection()
    for src in sources:
        collection.delete(where={"source": src})
    reset_index_cache()


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


def build_access_filters(
    doc_type: str,
    allowed_sources: list[str] | None,
) -> MetadataFilters | None:
    """构建向量通道的元数据过滤条件（应用层 RBAC 钩子）。

    Chroma 无原生 RBAC：上层按 user→allowed_sources 把可见 source 白名单传入，
    与 doc_type 叠加为 Chroma metadata 过滤。无任何限制时返回 None（行为不变）。

    Args:
        doc_type: 文档类型；"all" 表示不限类型
        allowed_sources: 允许访问的 source 白名单；None 表示不限来源

    Returns:
        叠加后的 MetadataFilters；无过滤项时为 None
    """
    filters: list[MetadataFilter] = []
    if doc_type != "all":
        filters.append(MetadataFilter(key="doc_type", value=doc_type))
    if allowed_sources:
        filters.append(
            MetadataFilter(key="source", value=allowed_sources, operator=FilterOperator.IN)
        )
    if not filters:
        return None
    return MetadataFilters(filters=filters)


def _build_metadata_filters(doc_type: str) -> MetadataFilters | None:
    """构建文档类型过滤条件（build_access_filters 的无来源限制特例）。"""
    return build_access_filters(doc_type, None)


# 中文 BM25 用 jieba 词级分词。BM25Retriever.from_defaults 的 tokenizer= 形参已废弃且被忽略，
# 故走官方扩展点：自建 existing_bm25（语料经 jieba 预切分），查询端在子类 _retrieve 里同样预切分，
# 两端统一用空白分词 token_pattern，保证一致。
_BM25_WS_PATTERN = r"(?u)\S+"


def _jieba_tokenize(text: str) -> str:
    """jieba 分词后空格连接（供 bm25s 按空白再切，实现词级匹配）。

    用 jieba.cut（全版本可用）而非 lcut（旧版缺失）。
    """
    return " ".join(tok for tok in jieba.cut(text) if tok.strip())


class _JiebaBM25Retriever(BM25Retriever):
    """查询端也用 jieba 预切分，与 jieba 切分的语料对齐。"""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return super()._retrieve(QueryBundle(query_str=_jieba_tokenize(query_bundle.query_str)))


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


def _build_jieba_bm25(nodes: list[TextNode], top_k: int) -> _JiebaBM25Retriever:
    """用 jieba 预切分语料自建 bm25s 索引，包成 retriever（语料保留原文，仅打分文本被切分）。"""
    corpus_records = [node_to_metadata_dict(n) | {"node_id": n.node_id} for n in nodes]
    corpus_tokens = bm25s.tokenize(
        [_jieba_tokenize(n.get_content()) for n in nodes],
        token_pattern=_BM25_WS_PATTERN,
        stemmer=None,  # 与查询端一致：不做英文词干还原
    )
    bm25 = bm25s.BM25(corpus=corpus_records)  # 检索按匹配行号返回 corpus_records[i]（原文节点）
    bm25.index(corpus_tokens)
    return _JiebaBM25Retriever(
        existing_bm25=bm25,
        similarity_top_k=top_k,
        token_pattern=_BM25_WS_PATTERN,
        skip_stemming=True,
    )


def get_bm25_retriever(doc_type: str = "all") -> _JiebaBM25Retriever | None:
    """按 doc_type 构建并缓存 BM25（严格过滤，F3 升级；jieba 词级中文分词）。

    该 doc_type 下无节点时返回 None，调用方退化为纯向量检索。
    """
    if doc_type not in _bm25_retrievers:
        nodes = load_all_nodes()
        if doc_type != "all":
            nodes = [n for n in nodes if (n.metadata or {}).get("doc_type") == doc_type]
        if not nodes:
            _bm25_retrievers[doc_type] = None
        else:
            # k 不得超过节点数，否则 bm25s 会告警并覆盖；显式 clamp 消除告警
            _bm25_retrievers[doc_type] = _build_jieba_bm25(nodes, min(BM25_TOP_K, len(nodes)))
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
        bm25 = get_bm25_retriever(doc_type)
        sub_retrievers = [vector_retriever] + ([bm25] if bm25 is not None else [])
        if len(sub_retrievers) > 1 or ENABLE_MULTI_QUERY:
            num_queries = MULTI_QUERY_NUM if ENABLE_MULTI_QUERY else 1
            retriever = QueryFusionRetriever(
                sub_retrievers,
                llm=get_llm(),            # S2：显式传 LLM，避免回退到未配置的 Settings.llm
                similarity_top_k=candidate_k,
                num_queries=num_queries,
                query_gen_prompt=MULTI_QUERY_PROMPT,  # F-3：中文扩展提示词（num_queries=1 时不生效，无害）
                mode="reciprocal_rerank", # RRF 融合
                use_async=ENABLE_MULTI_QUERY,  # 多查询时并行
            )
            nodes = retriever.retrieve(query)
        else:
            # 该 doc_type 无 BM25 节点且未开多查询 → 退化为纯向量
            nodes = vector_retriever.retrieve(query)
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


# 常见提示词注入标记（中英，命中即整段替换为占位符）。
# 仅匹配祈使型越权短语，避免误伤正常运维文本（如单独的「指令」一词）。
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"忽略(以上|上述|前面|之前|前述).{0,8}(指令|提示|要求|规则|内容)",
        r"(无视|忽略)上述",
        r"你现在是",
        r"从现在起你是",
        r"disregard\s+(the\s+)?(above|previous|prior)",
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)",
        r"system\s+prompt",
        r"override.*instructions",
    )
]

_INJECTION_PLACEHOLDER = "[已移除可疑指令]"


def sanitize_context(text: str) -> str:
    """
    中和检索内容中的提示词注入标记。

    剥离常见越权指令短语（中英、大小写不敏感），替换为占位符，
    保留其余事实内容。纯函数，不调用模型。

    Args:
        text: 单段检索内容原文

    Returns:
        中和注入标记后的文本
    """
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub(_INJECTION_PLACEHOLDER, cleaned)
    return cleaned


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
        # 先中和注入（避免短语被截断错过），再截断长度
        content = sanitize_context(node.get_content().strip())
        if len(content) > CONTEXT_CHUNK_MAX_CHARS:
            content = content[:CONTEXT_CHUNK_MAX_CHARS].rstrip() + "..."
        blocks.append(f"{header}\n<<DOC>>\n{content}\n<</DOC>>")
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
