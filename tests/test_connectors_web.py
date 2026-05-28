"""web 连接器测试（注入假 reader，不联网）。"""

import app.connectors.web as web_mod
from app.connectors import RawDocument
from app.sources import SourceConfig


def test_web_connector_maps_documents(monkeypatch):
    class _Doc:
        def __init__(self, text, url):
            self.text = text
            self.metadata = {"url": url}

    class _FakeReader:
        def load_data(self, urls):
            return [_Doc("网页正文", urls[0])]

    monkeypatch.setattr(web_mod, "_make_reader", lambda: _FakeReader())

    cfg = SourceConfig("wiki", "web", "doc", {"urls": ["https://example.com/a"]})
    docs = list(web_mod.WebConnector.from_config(cfg).load())
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].fmt == "web"
    assert docs[0].metadata["doc_type"] == "doc"
    assert docs[0].metadata["source"] == "https://example.com/a"


def test_web_connector_no_urls_returns_empty(monkeypatch):
    monkeypatch.setattr(web_mod, "_make_reader", lambda: None)
    cfg = SourceConfig("wiki", "web", "doc", {})
    assert list(web_mod.WebConnector.from_config(cfg).load()) == []
