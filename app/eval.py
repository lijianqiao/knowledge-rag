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


def _default_ask(question: str) -> tuple[str, list[str], str]:
    """默认问答 seam：真实检索 + 生成（仅 run_eval 实际调用时触发，不在 import/测试时执行）。"""
    from app.config import RETRIEVE_TOP_K
    from app.index import build_context, generate_answer, retrieve_nodes

    nodes = retrieve_nodes(question, top_k=RETRIEVE_TOP_K)
    context = build_context(nodes)
    answer = generate_answer(question, context)
    sources = [n.metadata.get("source", "") for n in nodes]
    return answer, sources, context


def run_eval(
    goldset_path: Path,
    ask_fn=None,
    faithfulness_fn=None,
    relevancy_fn=None,
    output_format: str = "text",
) -> str:
    """跑评测集，返回人类可读报告字符串。

    Args:
        goldset_path: 评测集 JSON 路径（[{question, expected_source_substrings, ...}]）。
        ask_fn: question -> (answer, sources, context)，默认走真实检索+生成。
        faithfulness_fn: (answer, context) -> float，默认 judge_faithfulness。
        relevancy_fn: (answer, question) -> float，默认 judge_relevancy。
    """
    ask = ask_fn or _default_ask
    judge_f = faithfulness_fn or judge_faithfulness
    judge_r = relevancy_fn or judge_relevancy

    goldset = load_goldset(goldset_path)
    rows: list[dict] = []
    for item in goldset:
        question = item["question"]
        answer, sources, context = ask(question)
        recall = context_recall(sources, item.get("expected_source_substrings", []))
        faithfulness = judge_f(answer, context)
        relevancy = judge_r(answer, question)
        rows.append(
            {
                "question": question,
                "sources": sources,
                "answer_chars": len(answer),
                "recall": recall,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
            }
        )

    agg = aggregate(rows)
    if output_format == "json":
        return json.dumps({"rows": rows, "aggregate": agg}, ensure_ascii=False, indent=2)

    lines = [f"评测集: {goldset_path}（共 {len(rows)} 条）", ""]
    for r in rows:
        lines.append(
            f"- {r['question']}  recall={r['recall']:.2f} "
            f"faithfulness={r['faithfulness']:.2f} relevancy={r['relevancy']:.2f}"
        )
    lines += [
        "",
        "--- 聚合 ---",
        f"recall={agg['recall']} faithfulness={agg['faithfulness']} relevancy={agg['relevancy']}",
    ]
    return "\n".join(lines)
