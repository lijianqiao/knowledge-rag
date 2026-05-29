"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: import_docs.py
@DateTime: 2026-05-28
@Docs: 文档导入向量库（LlamaIndex TextNode → ChromaDB）
"""

from pathlib import Path

from llama_index.core.schema import TextNode

from app.config import COLLECTION_NAME, MANIFEST_PATH
from app.graph_index import build_graph_index
from app.index import (
    delete_chroma_collection,
    delete_chunks_by_sources,
    get_chroma_collection,
    insert_text_nodes,
    reset_index_cache,
)
from app.loader import load_chunks
from app.manifest import content_hash, diff_manifest, load_manifest, save_manifest


def _resolve_source_names(source: str) -> list[str]:
    """把 CLI --source 解析为 load_chunks 的源名列表（"all" → 空列表 = 全部）。"""
    if source == "all":
        return []
    return [source]


def _to_text_nodes(records) -> list[TextNode]:
    """将分块记录转为 LlamaIndex TextNode。"""
    nodes: list[TextNode] = []
    for record in records:
        metadata = {key: str(value) for key, value in record.metadata.items()}
        nodes.append(
            TextNode(
                text=record.text,
                id_=record.chunk_id,
                metadata=metadata,
            )
        )
    return nodes


def _run_incremental(source: str) -> int:
    """按源文件内容 hash 增量导入：仅重导变更/新增文件，删除消失文件的 chunk。"""
    records = load_chunks(_resolve_source_names(source))

    by_source: dict[str, list] = {}
    for record in records:
        by_source.setdefault(record.metadata["source"], []).append(record)

    new_manifest = {src: content_hash("".join(r.text for r in recs)) for src, recs in by_source.items()}
    old_manifest = load_manifest(Path(MANIFEST_PATH))
    changed, removed = diff_manifest(old_manifest, new_manifest)

    if removed:
        delete_chunks_by_sources(removed)

    changed_set = set(changed)
    changed_records = [r for r in records if r.metadata["source"] in changed_set]
    nodes = _to_text_nodes(changed_records)
    if nodes:
        insert_text_nodes(nodes)

    save_manifest(Path(MANIFEST_PATH), new_manifest)
    print(f"增量导入：变更 {len(changed)} 个文件、删除 {len(removed)} 个文件、upsert {len(nodes)} 个 chunk → {COLLECTION_NAME}")
    return len(nodes)


def run_import(source: str = "all", force: bool = False, upsert: bool = False, incremental: bool = False) -> int:
    """
    导入 Markdown 分块到向量库。

    Args:
        source: "all" 或 sources.toml 中的源名
        force: 删除 collection 后全量重建
        upsert: 更新已有分块（LlamaIndex insert 覆盖同 id）
        incremental: 按内容 hash 清单增量导入（仅重导变更、清理消失文件）

    Returns:
        写入条数
    """
    if force and upsert:
        raise ValueError("force 与 upsert 不能同时使用")
    if force and incremental:
        raise ValueError("force 与 incremental 不能同时使用")

    if incremental:
        return _run_incremental(source)

    records = load_chunks(_resolve_source_names(source))
    nodes = _to_text_nodes(records)

    if force:
        delete_chroma_collection()
        reset_index_cache()
        insert_text_nodes(nodes)
        print(f"已重建并导入 {len(nodes)} 个 chunk → {COLLECTION_NAME}")
        return len(nodes)

    if get_chroma_collection().count() == 0:
        insert_text_nodes(nodes)
        print(f"首次导入 {len(nodes)} 个 chunk → {COLLECTION_NAME}")
        return len(nodes)

    if upsert:
        insert_text_nodes(nodes)
        print(f"已 upsert {len(nodes)} 个 chunk → {COLLECTION_NAME}")
        return len(nodes)

    print(f"已有 {get_chroma_collection().count()} 个 chunk，跳过。使用 --upsert 或 --force")
    return 0


def run_graph_build(source: str = "all") -> int:
    """从文档源抽取实体关系构建知识图谱（慢，调 LLM）。

    Returns:
        参与构建的节点数
    """
    records = load_chunks(_resolve_source_names(source))
    nodes = _to_text_nodes(records)
    build_graph_index(nodes)
    print(f"已构建图谱：{len(nodes)} 个节点参与抽取")
    return len(nodes)
