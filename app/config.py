"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: config.py
@DateTime: 2026-05-28
@Docs: 应用配置与文档源定义
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DocumentSource:
    """
    文档源配置。

    Attributes:
        key: 源标识
        root: 文档根目录
        doc_type: 文档类型标签
        skip_files: 跳过的文件名
    """

    key: str
    root: Path
    doc_type: str
    skip_files: frozenset[str]


DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ops_knowledge")

API_KEY = os.getenv("OPENAI_API_KEY", "dummy")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1"))
EMBED_MODEL = os.getenv("EMBED_MODEL", os.getenv("LLM_MODEL", "Qwen3-Embedding-0.6B"))

CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "http://127.0.0.1:8081/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen3-4B-Instruct")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
CONTEXT_CHUNK_MAX_CHARS = int(os.getenv("CONTEXT_CHUNK_MAX_CHARS", "1200"))

RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))
MAX_RETRIEVE_RETRIES = int(os.getenv("MAX_RETRIEVE_RETRIES", "2"))
RETRIEVE_SCORE_THRESHOLD = float(os.getenv("RETRIEVE_SCORE_THRESHOLD", "0.35"))

# 宽召回 → 重排（2026 基线：先宽召回再 cross-encoder 重排）
RETRIEVE_CANDIDATE_K = int(os.getenv("RETRIEVE_CANDIDATE_K", "30"))
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# 注：重排后保留条数由调用方 top_k（CLI -n）决定，不再单独配 RERANK_TOP_N，避免覆盖 -n。

# Hybrid 检索（稠密 + BM25 稀疏融合）
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "30"))

# Query Rewriting（LLM 改写检索查询，替换弱重试）
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

SOURCES_CONFIG_PATH = os.getenv("SOURCES_CONFIG_PATH", "sources.toml")

DOCUMENT_SOURCES: dict[str, DocumentSource] = {
    "prompts": DocumentSource(
        key="prompts",
        root=Path("运维prompt库"),
        doc_type="prompt",
        skip_files=frozenset({"README.md", "Prompt模板规范.md"}),
    ),
    "docs": DocumentSource(
        key="docs",
        root=Path("运维文档"),
        doc_type="doc",
        skip_files=frozenset({"README.md"}),
    ),
}

RAG_SYSTEM_PROMPT = """你是企业运维知识库助手。请严格基于提供的参考资料回答问题。

要求：
1. 仅使用参考资料中的信息，不得编造
2. 信息不足时明确说明「信息不足」并列出还需哪些信息
3. 使用中文，技术术语保留英文
4. 回答末尾列出引用来源编号，如 [1][2]
"""

QUERY_REWRITE_PROMPT = """你是检索查询优化助手。下面的查询召回结果不理想，请改写成更利于向量检索与关键词检索的查询。
要求：保留原始意图，补全可能的同义词/术语，去掉口语化与无关词，只输出改写后的查询本身，不要解释。

原始问题：{question}
上次查询：{prev_query}
改写后的查询："""
