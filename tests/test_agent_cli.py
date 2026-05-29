"""agent 子命令解析测试。"""

from main import build_parser


def test_agent_subcommand_parses():
    args = build_parser().parse_args(["agent", "订单服务依赖什么", "-n", "6"])
    assert args.command == "agent"
    assert args.text == "订单服务依赖什么"
    assert args.n == 6
