"""LangGraph SQLite checkpointer 封装：会话/HITL 的持久化后端。

`SqliteSaver(conn)` 需一个 `sqlite3.Connection`（`from_conn_string` 返回的是
上下文管理器，不适合长生命周期单例）。FastAPI 多线程 → `check_same_thread=False`。
连接在首次 `get_checkpointer()` 时惰性创建，模块 import 不建连接。
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import CHECKPOINT_DB

_checkpointer: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    """返回模块级 SqliteSaver 单例（连 CHECKPOINT_DB）。"""
    global _checkpointer
    if _checkpointer is None:
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer


def reset_checkpointer() -> None:
    """清空单例（供测试）。"""
    global _checkpointer
    _checkpointer = None
