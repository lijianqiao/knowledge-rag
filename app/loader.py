"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: loader.py
@DateTime: 2026-05-28
@Docs: Markdown 扫描与分块
"""

import re
from dataclasses import dataclass

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENT_SOURCES, DocumentSource


@dataclass(frozen=True)
class ChunkRecord:
    """
    单条待入库分块。

    Attributes:
        chunk_id: 向量库唯一 ID
        text: 分块正文
        metadata: 元数据
    """

    chunk_id: str
    text: str
    metadata: dict[str, str | int]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符窗口切分超长文本。"""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_markdown(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    按 Markdown 二级标题优先分块，过长段落再滑窗切分。

    Args:
        text: Markdown 全文
        chunk_size: 单块最大字符数
        overlap: 滑窗重叠字符数

    Returns:
        分块文本列表
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    sections = re.split(r"(?=\n## )", cleaned)
    chunks: list[str] = []
    buffer = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(buffer) + len(section) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{section}".strip() if buffer else section
            continue

        if buffer:
            chunks.append(buffer)

        if len(section) <= chunk_size:
            buffer = section
        else:
            chunks.extend(_split_long_text(section, chunk_size, overlap))
            buffer = ""

    if buffer:
        chunks.append(buffer)

    return chunks or [cleaned]


def _load_source(source: DocumentSource) -> list[ChunkRecord]:
    """从单个文档源加载并分块。"""
    if not source.root.exists():
        return []

    records: list[ChunkRecord] = []
    for md_file in source.root.rglob("*.md"):
        if md_file.name in source.skip_files:
            continue

        rel_path = md_file.relative_to(source.root).as_posix()
        category = rel_path.split("/")[0] if "/" in rel_path else "根目录"
        base_id = f"{source.doc_type}_{rel_path.replace('/', '_').replace('.md', '')}"
        parts = chunk_markdown(md_file.read_text(encoding="utf-8"))
        total = len(parts)

        for index, part in enumerate(parts):
            records.append(
                ChunkRecord(
                    chunk_id=f"{base_id}__{index}",
                    text=part,
                    metadata={
                        "source": rel_path,
                        "category": category,
                        "title": md_file.stem,
                        "doc_type": source.doc_type,
                        "library": source.root.name,
                        "chunk_index": index,
                        "chunk_total": total,
                    },
                )
            )
    return records


def load_chunks(source_keys: list[str]) -> list[ChunkRecord]:
    """
    按源标识加载分块记录。

    Args:
        source_keys: 如 ["prompts", "docs"]

    Returns:
        分块列表

    Raises:
        ValueError: 源无效或无文档时
    """
    if not source_keys:
        raise ValueError("至少需要指定一个文档源")

    records: list[ChunkRecord] = []
    for key in source_keys:
        source = DOCUMENT_SOURCES.get(key)
        if source is None:
            raise ValueError(f"未知文档源: {key}")
        records.extend(_load_source(source))

    if not records:
        roots = ", ".join(str(DOCUMENT_SOURCES[k].root) for k in source_keys)
        raise ValueError(f"未找到可导入文档: {roots}")

    return records


def to_batches(records: list[ChunkRecord]) -> tuple[list[str], list[str], list[dict[str, str | int]]]:
    """转为 ChromaDB 写入批次。"""
    return (
        [r.chunk_id for r in records],
        [r.text for r in records],
        [r.metadata for r in records],
    )
