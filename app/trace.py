"""轻量结构化链路追踪（JSON Lines，零依赖）。可用 ENABLE_TRACE 关闭。"""

import json
import os
import time
import uuid
from pathlib import Path

from app.config import ENABLE_TRACE, TRACE_DIR


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(trace_id: str, event: str, payload: dict) -> None:
    if not ENABLE_TRACE:
        return
    Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
    rec = {"ts": round(time.time(), 3), "trace_id": trace_id, "event": event, **{"data": payload}}
    with open(Path(TRACE_DIR) / f"trace-{trace_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
