"""非流式端点：threadpool 卸载 + 请求超时（TestClient，离线，monkeypatch）。

超时验证说明：anyio.fail_after 包住 anyio.to_thread.run_sync 时，能在 *可取消*
的 await 上抛内建 TimeoutError，但无法强行中断已在工作线程里跑的同步阻塞（线程不可
强杀）——genuine time.sleep 在 to_thread 下不会被 fail_after 打断（已实测：慢调用
返回 200 而非 504）。故 504 用例采用「模拟」：让业务函数（在线程内执行）直接抛内建
TimeoutError，等价于 fail_after 触发的效果，精确验证 _run_blocking 的 except
TimeoutError → 504 映射；正常用例则走完整真实路径断言 200。
"""

from fastapi.testclient import TestClient

import app.api.routes_rag as rr
from app.api.app import create_app


def test_timeout_returns_504(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        # 在线程池内抛 TimeoutError，模拟 fail_after 超时被 _run_blocking 捕获
        raise TimeoutError

    monkeypatch.setattr(rr, "run_ask", _raise_timeout)

    c = TestClient(create_app())
    r = c.post("/ask", json={"question": "X", "top_k": 3, "doc_type": "all"})
    assert r.status_code == 504
    assert r.json()["detail"] == "请求超时"


def test_normal_request_ok(monkeypatch):
    monkeypatch.setattr(
        rr, "run_ask", lambda q, top_k, doc_type, allowed_sources: "答案"
    )
    c = TestClient(create_app())
    r = c.post("/ask", json={"question": "X", "top_k": 3, "doc_type": "all"})
    assert r.status_code == 200
    assert r.json()["answer"] == "答案"
