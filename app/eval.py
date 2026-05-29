"""RAG 评估：自写 LLM-as-judge + 确定性指标（不依赖 ragas）。"""

import json
from pathlib import Path


def context_recall(retrieved_sources: list[str], expected_substrings: list[str]) -> float:
    """期望来源子串在检索到的 source 列表里的命中比例。"""
    if not expected_substrings:
        return 1.0
    blob = " ".join(retrieved_sources).lower()
    hit = sum(1 for e in expected_substrings if e.lower() in blob)
    return hit / len(expected_substrings)


def aggregate(rows: list[dict]) -> dict:
    """对各指标求均值。"""
    keys = ("recall", "faithfulness", "relevancy")
    n = len(rows) or 1
    return {k: round(sum(r.get(k, 0.0) for r in rows) / n, 4) for k in keys}


def load_goldset(path: Path) -> list[dict]:
    """读评测集 [{question, expected_source_substrings, ...}]。"""
    if not path.exists():
        raise ValueError(f"评测集不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("评测集应为非空 JSON 数组")
    return data
