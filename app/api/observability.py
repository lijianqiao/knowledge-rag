"""Langfuse 可观测适配器（可选，默认关）。

自写 app/trace.py（JSONL，零依赖）仍是默认观测手段，本模块不影响它。
ENABLE_LANGFUSE=true 且 langfuse SDK 可用时，forward_trace 会尽力把一次链路的
事件转发到 Langfuse；任何 import/运行期错误都被吞掉并返回 False，绝不破坏请求路径。

wiring deferred：本期仅提供独立适配器，未接入检索热路径（self-trace 仍为默认）。
"""

from app.config import (
    ENABLE_LANGFUSE,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)


def _get_client():
    """惰性构造 Langfuse 客户端；SDK 缺失或构造失败返回 None。"""
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    try:
        return Langfuse(
            host=LANGFUSE_HOST or None,
            public_key=LANGFUSE_PUBLIC_KEY or None,
            secret_key=LANGFUSE_SECRET_KEY or None,
        )
    except Exception:
        return None


def forward_trace(trace_id: str, events: list[dict]) -> bool:
    """把一次链路的事件转发到 Langfuse（尽力而为）。

    返回 True 仅当确实转发成功；ENABLE_LANGFUSE 关、SDK 缺失或任何异常 → False。
    自写 JSONL trace 不受影响，仍是默认。
    """
    if not ENABLE_LANGFUSE:
        return False
    client = _get_client()
    if client is None:
        return False
    try:
        for event in events:
            client.event(
                trace_id=trace_id,
                name=event.get("event", "event"),
                metadata=event.get("data", {}),
            )
        return True
    except Exception:
        return False
