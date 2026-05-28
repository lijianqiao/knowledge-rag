"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: loader.py
@DateTime: 2026-05-28
@Docs: Markdown 扫描与分块
"""

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, SOURCES_CONFIG_PATH
from app.connectors import RawDocument, build_connector
from app.sources import load_source_configs


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


def chunk_document(doc: RawDocument) -> list[ChunkRecord]:
    """按格式分块：md 走标题分块，其余走滑窗。"""
    parts = chunk_markdown(doc.text) if doc.fmt == "md" else _split_long_text(doc.text, CHUNK_SIZE, CHUNK_OVERLAP)
    total = len(parts)
    return [
        ChunkRecord(
            chunk_id=f"{doc.doc_id}__{i}",
            text=part,
            metadata={**doc.metadata, "chunk_index": i, "chunk_total": total},
        )
        for i, part in enumerate(parts)
    ]


def load_chunks(source_names: list[str]) -> list[ChunkRecord]:
    """按源名加载分块（source_names 为空表示全部）。

    Raises:
        ValueError: 源名无效或无文档时
    """
    configs = load_source_configs(Path(SOURCES_CONFIG_PATH))
    by_name = {c.name: c for c in configs}
    selected = configs if not source_names else []
    for name in source_names:
        if name not in by_name:
            raise ValueError(f"未知数据源: {name}（可用: {sorted(by_name)}）")
        selected.append(by_name[name])

    records: list[ChunkRecord] = []
    for cfg in selected:
        for doc in build_connector(cfg).load():
            records.extend(chunk_document(doc))

    if not records:
        raise ValueError("未找到可导入文档，请检查 sources.toml 的 root/urls")
    return records


def to_batches(records: list[ChunkRecord]) -> tuple[list[str], list[str], list[dict[str, str | int]]]:
    """转为 ChromaDB 写入批次。"""
    return (
        [r.chunk_id for r in records],
        [r.text for r in records],
        [r.metadata for r in records],
    )
