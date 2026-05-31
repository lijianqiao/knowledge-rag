"""serve 子命令参数解析（不启动 uvicorn）。"""

from app.config import SERVE_HOST, SERVE_PORT
from main import build_parser


def test_serve_defaults():
    args = build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == SERVE_HOST
    assert args.port == SERVE_PORT


def test_serve_override_port():
    args = build_parser().parse_args(["serve", "--port", "9001"])
    assert args.port == 9001
