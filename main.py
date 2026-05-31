"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: main.py
@DateTime: 2026-05-28
@Docs: CLI 入口：import / query / ask / status
"""

import argparse
import sys
from pathlib import Path

from app.agent import run_agent
from app.config import ENABLE_AGENT, SERVE_HOST, SERVE_PORT, SOURCES_CONFIG_PATH
from app.eval import run_eval
from app.graph import run_ask, run_ask_stream
from app.import_docs import run_graph_build, run_import
from app.index import format_nodes, get_status, retrieve_nodes
from app.sources import load_source_configs


def _source_choices() -> list[str]:
    """从 sources.toml 读取可用源名供 --source；读取失败时退回仅 all。"""
    try:
        names = [c.name for c in load_source_configs(Path(SOURCES_CONFIG_PATH))]
    except ValueError:
        names = []
    return ["all", *names]


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="运维知识库 RAG（LangGraph + LlamaIndex）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="导入文档（分块向量化）")
    p_import.add_argument("--source", choices=_source_choices(), default="all")
    p_import.add_argument("--force", action="store_true", help="重建 collection")
    p_import.add_argument("--upsert", action="store_true", help="更新已有分块")
    p_import.add_argument("--incremental", action="store_true", help="按内容 hash 清单增量导入（仅重导变更、清理消失文件）")

    p_query = sub.add_parser("query", help="调试检索（LlamaIndex → ChromaDB）")
    p_query.add_argument("text")
    p_query.add_argument("-n", type=int, default=5)
    p_query.add_argument("--type", choices=["all", "prompt", "doc"], default="all")

    p_ask = sub.add_parser("ask", help="RAG 问答（LangGraph 编排）")
    p_ask.add_argument("text")
    p_ask.add_argument("-n", type=int, default=5)
    p_ask.add_argument("--type", choices=["all", "prompt", "doc"], default="all")
    p_ask.add_argument("--stream", action="store_true", help="流式增量输出（单趟检索，无重试）")

    p_graph = sub.add_parser("graph-build", help="构建知识图谱（GraphRAG，慢）")
    p_graph.add_argument("--source", choices=_source_choices(), default="all")

    p_agent = sub.add_parser("agent", help="跨文档推理问答（多步 Agent）")
    p_agent.add_argument("text")
    p_agent.add_argument("-n", type=int, default=5)

    p_eval = sub.add_parser("eval", help="跑评测集（recall + LLM-as-judge faithfulness/relevancy）")
    p_eval.add_argument("--set", dest="goldset", default="eval/goldset.example.json", help="评测集 JSON 路径")
    p_eval.add_argument("--format", dest="fmt", choices=["text", "json"], default="text", help="输出格式（json 为机器可读）")

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务（API-only）")
    p_serve.add_argument("--host", default=SERVE_HOST, help=f"监听地址（默认 {SERVE_HOST}）")
    p_serve.add_argument("--port", type=int, default=SERVE_PORT, help=f"监听端口（默认 {SERVE_PORT}）")

    sub.add_parser("status", help="向量库状态")
    sub.add_parser("health", help="Deep health check")
    return parser


def main() -> None:
    """CLI 主入口。"""
    args = build_parser().parse_args()

    try:
        if args.command == "import":
            run_import(source=args.source, force=args.force, upsert=args.upsert, incremental=args.incremental)
        elif args.command == "query":
            nodes = retrieve_nodes(args.text, top_k=args.n, doc_type=args.type)
            print(format_nodes(nodes))
        elif args.command == "ask":
            if args.stream:
                for chunk in run_ask_stream(args.text, top_k=args.n, doc_type=args.type):
                    print(chunk, end="", flush=True)
                print()
            else:
                print(run_ask(args.text, top_k=args.n, doc_type=args.type))
        elif args.command == "graph-build":
            run_graph_build(source=args.source)
        elif args.command == "agent":
            if not ENABLE_AGENT:
                raise ValueError("跨文档推理 Agent 未启用，请设 ENABLE_AGENT=true")
            print(run_agent(args.text, top_k=args.n))
        elif args.command == "eval":
            print(run_eval(Path(args.goldset), output_format=args.fmt))
        elif args.command == "serve":
            import uvicorn

            from app.api.app import create_app

            uvicorn.run(create_app(), host=args.host, port=args.port)
        elif args.command == "status":
            print(get_status())
        elif args.command == "health":
            import json

            from app.health import deep_health

            print(json.dumps(deep_health(), ensure_ascii=False, indent=2))
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
