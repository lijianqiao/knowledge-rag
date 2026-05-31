"""Context 组装与展示层（纯函数，无模型/状态）。

从 index.py 拆出（结构债治理）：提示词注入中和、`<<DOC>>` 包裹的 context 组装、
检索结果格式化都是无状态纯函数，与检索缓存解耦。index.py re-export 这些函数，
故 `from app.index import build_context, format_nodes, sanitize_context` 兼容不变。
"""

import re

from llama_index.core.schema import NodeWithScore

from app.config import CONTEXT_CHUNK_MAX_CHARS

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
