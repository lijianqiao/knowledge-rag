"""声明式数据源配置：解析 sources.toml。"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_REQUIRED = ("name", "type", "doc_type")


@dataclass(frozen=True)
class SourceConfig:
    """单个数据源声明。

    Attributes:
        name: 源唯一标识（CLI --source 用）
        type: 连接器类型（filesystem / web）
        doc_type: 文档类型标签（写入 metadata，用于检索过滤）
        params: 类型相关参数（root/glob/exclude 或 urls 等）
    """

    name: str
    type: str
    doc_type: str
    params: dict


def load_source_configs(path: Path) -> list[SourceConfig]:
    """解析 sources.toml。

    Raises:
        ValueError: 文件不存在、格式错误或缺必填字段时
    """
    if not path.exists():
        raise ValueError(f"源配置文件不存在: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"源配置解析失败: {exc}") from exc

    configs: list[SourceConfig] = []
    for raw in data.get("sources", []):
        for key in _REQUIRED:
            if key not in raw:
                raise ValueError(f"源缺少必填字段 '{key}': {raw}")
        params = {k: v for k, v in raw.items() if k not in _REQUIRED}
        configs.append(
            SourceConfig(name=raw["name"], type=raw["type"], doc_type=raw["doc_type"], params=params)
        )
    if not configs:
        raise ValueError(f"源配置为空: {path}")
    return configs
