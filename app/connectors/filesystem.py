"""本地目录连接器：用 LlamaIndex SimpleDirectoryReader 读多格式。"""

from collections.abc import Iterable
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from app.connectors import RawDocument
from app.sources import SourceConfig


class FilesystemConnector:
    """从本地目录按 glob 读取文件，产出 RawDocument。"""

    def __init__(self, root: Path, doc_type: str, glob: str, exclude: list[str]):
        self._root = root
        self._doc_type = doc_type
        self._glob = glob
        self._exclude = exclude

    @classmethod
    def from_config(cls, cfg: SourceConfig) -> "FilesystemConnector":
        params = cfg.params
        return cls(
            root=Path(params["root"]),
            doc_type=cfg.doc_type,
            glob=params.get("glob", "**/*.md"),
            exclude=list(params.get("exclude", [])),
        )

    def load(self) -> Iterable[RawDocument]:
        if not self._root.exists():
            return []
        reader = SimpleDirectoryReader(
            input_dir=str(self._root),
            recursive=True,
            exclude=self._exclude,
            required_exts=_exts_from_glob(self._glob),
            filename_as_id=False,
        )
        # 同一文件可能产出多个 Document（如 PDF 默认每页一个），按 file_path 分组去重 id（F-1）
        by_path: dict[str, list] = {}
        for d in reader.load_data():
            by_path.setdefault(d.metadata.get("file_path", ""), []).append(d)

        docs: list[RawDocument] = []
        root_resolved = self._root.resolve()
        for fp, group in by_path.items():
            file_path = Path(fp)
            try:
                rel = file_path.resolve().relative_to(root_resolved).as_posix()
            except ValueError:
                rel = file_path.name
            category = rel.split("/")[0] if "/" in rel else "根目录"
            stem = file_path.stem
            fmt = file_path.suffix.lstrip(".").lower() or "txt"
            base = f"{self._doc_type}_{rel.rsplit('.', 1)[0].replace('/', '_')}"
            multi = len(group) > 1  # 单 Document 文件（md/txt）保持原 id，守住 chunk_id 稳定性
            for i, d in enumerate(group):
                page = d.metadata.get("page_label")
                doc_id = base if not multi else f"{base}_p{page or i}"
                docs.append(
                    RawDocument(
                        doc_id=doc_id,
                        text=d.text,
                        metadata={
                            "source": rel,
                            "category": category,
                            "title": stem,
                            "doc_type": self._doc_type,
                            "library": self._root.name,
                            **({"page": str(page or i)} if multi else {}),
                        },
                        fmt=fmt,
                    )
                )
        return docs


def _exts_from_glob(glob: str) -> list[str] | None:
    """从 glob 提取扩展名列表给 SimpleDirectoryReader 的 required_exts。

    支持 '**/*.md' 与 '**/*.{md,pdf,docx}' 两种写法；无法解析时返回 None（不限制）。
    """
    if "*." not in glob:
        return None
    tail = glob.rsplit("*.", 1)[1]
    if tail.startswith("{") and tail.endswith("}"):
        return [f".{e.strip()}" for e in tail[1:-1].split(",") if e.strip()]
    return [f".{tail}"] if tail else None
