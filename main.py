"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: main.py
@DateTime: 2026-05-28
@Docs: CLI 入口：import / query / ask / status
"""

import argparse
import sys

from app.graph import run_ask
from app.import_docs import run_import
from app.index import format_nodes, get_status, retrieve_nodes


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="运维知识库 RAG（LangGraph + LlamaIndex）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="导入文档（分块向量化）")
    p_import.add_argument("--source", choices=["all", "prompts", "docs"], default="all")
    p_import.add_argument("--force", action="store_true", help="重建 collection")
    p_import.add_argument("--upsert", action="store_true", help="更新已有分块")

    p_query = sub.add_parser("query", help="调试检索（LlamaIndex → ChromaDB）")
    p_query.add_argument("text")
    p_query.add_argument("-n", type=int, default=5)
    p_query.add_argument("--type", choices=["all", "prompt", "doc"], default="all")

    p_ask = sub.add_parser("ask", help="RAG 问答（LangGraph 编排）")
    p_ask.add_argument("text")
    p_ask.add_argument("-n", type=int, default=5)
    p_ask.add_argument("--type", choices=["all", "prompt", "doc"], default="all")

    sub.add_parser("status", help="向量库状态")
    return parser


def main() -> None:
    """CLI 主入口。"""
    args = build_parser().parse_args()

    try:
        if args.command == "import":
            run_import(source=args.source, force=args.force, upsert=args.upsert)
        elif args.command == "query":
            nodes = retrieve_nodes(args.text, top_k=args.n, doc_type=args.type)
            print(format_nodes(nodes))
        elif args.command == "ask":
            print(run_ask(args.text, top_k=args.n, doc_type=args.type))
        elif args.command == "status":
            print(get_status())
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
