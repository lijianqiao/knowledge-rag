"""模型客户端层：Chat / Embedding 的 OpenAI 兼容单例，按 provider 选 local / cloud。

从 index.py 拆出（结构债治理）：模型客户端与"索引/检索"是独立关注点，且其单例状态
（`_llm`/`_embed_model`）与检索层的缓存互不耦合。index.py 仍 re-export 这两个函数，
故 `from app.index import get_llm` 等既有导入与 monkeypatch 不受影响。
"""

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from app.config import (
    API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    CHAT_PROVIDER,
    CLOUD_CHAT_API_KEY,
    CLOUD_CHAT_BASE_URL,
    CLOUD_CHAT_MODEL,
    CLOUD_EMBED_API_KEY,
    CLOUD_EMBED_BASE_URL,
    CLOUD_EMBED_MODEL,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)

_embed_model: OpenAIEmbedding | None = None
_llm: OpenAILike | None = None


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
