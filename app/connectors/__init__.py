"""连接器：把各类数据源取成统一的 RawDocument，按 type 注册分发。"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.sources import SourceConfig


@dataclass(frozen=True)
class RawDocument:
    """连接器输出的标准化原始文档（未分块）。

    Attributes:
        doc_id: 稳定基 id（分块时拼 __{i}）
        text: 正文
        metadata: 元数据（含 doc_type/source/title/category 等）
        fmt: 格式标签（md/pdf/docx/txt/web），决定分块策略
    """

    doc_id: str
    text: str
    metadata: dict[str, str | int]
    fmt: str


@runtime_checkable
class Connector(Protocol):
    def load(self) -> Iterable[RawDocument]: ...


# type -> 工厂。新连接器在此注册即可，核心流程不感知。
_FACTORIES: dict[str, Callable[[SourceConfig], Connector]] = {}


def register_connector(type_name: str, factory: Callable[[SourceConfig], Connector]) -> None:
    _FACTORIES[type_name] = factory


def build_connector(cfg: SourceConfig) -> Connector:
    """按 cfg.type 构造连接器。

    Raises:
        ValueError: 未知 type
    """
    factory = _FACTORIES.get(cfg.type)
    if factory is None:
        raise ValueError(f"未知数据源类型: {cfg.type}（已注册: {sorted(_FACTORIES)}）")
    return factory(cfg)


# 注册内置连接器（放末尾避免循环导入）
from app.connectors.filesystem import FilesystemConnector  # noqa: E402

register_connector("filesystem", FilesystemConnector.from_config)
