"""RAG 评估：自写 LLM-as-judge + 确定性指标（不依赖 ragas）。"""

import json
import re
from pathlib import Path

from app.index import get_llm
from app.config import EVAL_FAITHFULNESS_PROMPT, EVAL_RELEVANCY_PROMPT


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


def _score(text: str) -> float:
    """从 LLM 输出抽 0–1 分；失败给 0。

    先试整体 float（提示词要求只输出数字）；否则取所有数字里**最后一个**并 clamp，
    避免命中「2025」之类无关数字的前缀（取最后一个更接近"分数：X"结尾习惯）。
    """
    s = text.strip()
    try:
        return max(0.0, min(1.0, float(s)))
    except ValueError:
        nums = re.findall(r"\d+\.\d+|\d+", s)
        if not nums:
            return 0.0
        try:
            return max(0.0, min(1.0, float(nums[-1])))
        except ValueError:
            return 0.0


def judge_faithfulness(answer: str, context: str) -> float:
    try:
        return _score(str(get_llm().complete(EVAL_FAITHFULNESS_PROMPT.format(answer=answer, context=context))))
    except Exception:
        return 0.0


def judge_relevancy(answer: str, question: str) -> float:
    try:
        return _score(str(get_llm().complete(EVAL_RELEVANCY_PROMPT.format(answer=answer, question=question))))
    except Exception:
        return 0.0
