"""图谱持久化目录的存在性判定（持久化/加载由 graph_index 经 StorageContext 完成）。"""

from pathlib import Path

from app.config import GRAPH_PERSIST_DIR


def graph_store_exists() -> bool:
    """持久化目录是否已含构建产物（非空目录）。决定能否走图检索（R5）。"""
    p = Path(GRAPH_PERSIST_DIR)
    return p.is_dir() and any(p.iterdir())
