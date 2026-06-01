"""轻量结构化链路追踪（JSON Lines，零依赖）。可用 ENABLE_TRACE 关闭。"""

import json
import time
import uuid
from pathlib import Path

from app.config import ENABLE_LANGFUSE, ENABLE_TRACE, TRACE_DIR


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _forward_langfuse(trace_id: str, event: str, payload: dict) -> None:
    """可选转发到 Langfuse（默认关；惰性导入；任何异常吞掉，绝不破坏 trace/请求）。"""
    if not ENABLE_LANGFUSE:
        return
    try:
        from app.api.observability import forward_trace

        forward_trace(trace_id, [{"event": event, "data": payload}])
    except Exception:
        pass


def log_event(trace_id: str, event: str, payload: dict) -> None:
    if not ENABLE_TRACE:
        return
    Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
    rec = {"ts": round(time.time(), 3), "trace_id": trace_id, "event": event, **{"data": payload}}
    with open(Path(TRACE_DIR) / f"trace-{trace_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _forward_langfuse(trace_id, event, payload)
