"""网页连接器：抓取 URL 列表为 RawDocument。"""

import hashlib
from collections.abc import Iterable

from app.connectors import RawDocument
from app.sources import SourceConfig


def _make_reader():
    """构造 LlamaIndex 网页 reader（隔离以便测试 monkeypatch）。"""
    from llama_index.readers.web import SimpleWebPageReader

    return SimpleWebPageReader(html_to_text=True)


class WebConnector:
    def __init__(self, urls: list[str], doc_type: str):
        self._urls = urls
        self._doc_type = doc_type

    @classmethod
    def from_config(cls, cfg: SourceConfig) -> "WebConnector":
        return cls(urls=list(cfg.params.get("urls", [])), doc_type=cfg.doc_type)

    def load(self) -> Iterable[RawDocument]:
        if not self._urls:
            return []
        reader = _make_reader()
        docs: list[RawDocument] = []
        for d in reader.load_data(self._urls):
            url = d.metadata.get("url") if getattr(d, "metadata", None) else None
            url = url or "unknown"
            doc_id = f"{self._doc_type}_web_{hashlib.md5(url.encode()).hexdigest()[:12]}"
            docs.append(
                RawDocument(
                    doc_id=doc_id,
                    text=d.text,
                    metadata={
                        "source": url,
                        "category": "web",
                        "title": url,
                        "doc_type": self._doc_type,
                        "library": "web",
                    },
                    fmt="web",
                )
            )
        return docs
