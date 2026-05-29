"""失败复现：按 trace_id 回放一次问答/Agent 的完整链路。

用法：
    uv run python scripts/replay.py <trace_id>
    uv run python scripts/replay.py <trace_id> --dir ./logs

读取 {TRACE_DIR}/trace-<trace_id>.jsonl（由 app.trace.log_event 写入），
按时间顺序打印 route / rewrite / retrieve TopK / answer 等事件，便于调试。
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import TRACE_DIR


def _format_event(rec: dict) -> str:
    """把一条 trace 记录格式化为可读行。"""
    event = rec.get("event", "?")
    ts = rec.get("ts", "")
    data = rec.get("data", {})
    return f"[{ts}] {event}: {json.dumps(data, ensure_ascii=False)}"


def replay(trace_id: str, trace_dir: str) -> str:
    """读取并格式化某条 trace 的全部事件；找不到时返回提示。"""
    path = Path(trace_dir) / f"trace-{trace_id}.jsonl"
    if not path.exists():
        return f"未找到 trace 文件: {path}"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return f"trace 文件为空: {path}"
    out = [f"=== trace {trace_id}（{len(lines)} 个事件）==="]
    out.extend(_format_event(json.loads(ln)) for ln in lines)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="按 trace_id 回放问答链路")
    parser.add_argument("trace_id")
    parser.add_argument("--dir", dest="trace_dir", default=TRACE_DIR, help="trace 目录（默认取 TRACE_DIR）")
    args = parser.parse_args()
    report = replay(args.trace_id, args.trace_dir)
    print(report)
    if report.startswith("未找到") or report.startswith("trace 文件为空"):
        sys.exit(1)


if __name__ == "__main__":
    main()
