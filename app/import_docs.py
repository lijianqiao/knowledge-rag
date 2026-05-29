"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: import_docs.py
@DateTime: 2026-05-28
@Docs: 文档导入向量库（LlamaIndex TextNode → ChromaDB）
"""

from llama_index.core.schema import TextNode

from app.config import COLLECTION_NAME
from app.index import delete_chroma_collection, get_chroma_collection, insert_text_nodes, reset_index_cache
from app.loader import load_chunks


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


def run_import(source: str = "all", force: bool = False, upsert: bool = False) -> int:
    """
    导入 Markdown 分块到向量库。

    Args:
        source: "all" 或 sources.toml 中的源名
        force: 删除 collection 后全量重建
        upsert: 更新已有分块（LlamaIndex insert 覆盖同 id）

    Returns:
        写入条数
    """
    if force and upsert:
        raise ValueError("force 与 upsert 不能同时使用")

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
