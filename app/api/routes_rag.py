"""RAG 端点：/ask /query /status。薄服务层，仅鉴权 + 透传到现有业务函数。

非流式端点为 async def：阻塞型业务调用（同步 LLM/HTTP/Chroma）经
`anyio.to_thread.run_sync` 卸载到线程池，避免阻塞事件循环；并用
`anyio.fail_after(REQUEST_TIMEOUT)` 包裹，超时返回 504。
流式 `/ask/stream` 维持 StreamingResponse，自行处理生成器内阻塞，不改。
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import anyio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent import run_agent
from app.api.auth import Principal, require_principal
from app.api.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    EvalRequest,
    EvalResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from app.config import ENABLE_AGENT, REQUEST_TIMEOUT
from app.eval import run_eval
from app.graph import run_ask, run_ask_stream
from app.index import format_nodes, get_status, retrieve_nodes

router = APIRouter()

_T = TypeVar("_T")


async def _run_blocking(fn: Callable[[], _T]) -> _T:
    """在线程池跑零参阻塞调用，超 REQUEST_TIMEOUT 抛 HTTP 504。

    fn 须为零参可调用（用 functools.partial 绑定实参），因
    anyio.to_thread.run_sync 仅按位置传参、不支持 kwargs。
    anyio.fail_after 超时抛内建 TimeoutError（anyio 4.x）。
    """
    try:
        with anyio.fail_after(REQUEST_TIMEOUT):
            return await anyio.to_thread.run_sync(fn)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="请求超时")


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, principal: Principal = Depends(require_principal)) -> AskResponse:
    answer = await _run_blocking(
        functools.partial(
            run_ask,
            req.question,
            req.top_k,
            req.doc_type,
            allowed_sources=principal.allowed_sources,
        )
    )
    return AskResponse(answer=answer)


@router.post("/ask/stream")
def ask_stream(
    req: AskRequest, principal: Principal = Depends(require_principal)
) -> StreamingResponse:
    def gen():
        for chunk in run_ask_stream(
            req.question,
            req.top_k,
            req.doc_type,
            allowed_sources=principal.allowed_sources,
        ):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest, principal: Principal = Depends(require_principal)
) -> QueryResponse:
    nodes = await _run_blocking(
        functools.partial(
            retrieve_nodes,
            req.question,
            top_k=req.top_k,
            doc_type=req.doc_type,
            allowed_sources=principal.allowed_sources,
        )
    )
    return QueryResponse(sources=format_nodes(nodes))


@router.get("/status", response_model=StatusResponse)
async def status(principal: Principal = Depends(require_principal)) -> StatusResponse:
    return StatusResponse(status=await _run_blocking(functools.partial(get_status)))


@router.post("/agent", response_model=AgentResponse)
async def agent(
    req: AgentRequest, principal: Principal = Depends(require_principal)
) -> AgentResponse:
    if not ENABLE_AGENT:
        raise HTTPException(status_code=403, detail="跨文档推理 Agent 未启用，请设 ENABLE_AGENT=true")
    answer = await _run_blocking(
        functools.partial(
            run_agent,
            req.question,
            req.top_k,
            allowed_sources=principal.allowed_sources,
        )
    )
    return AgentResponse(answer=answer)


@router.post("/eval", response_model=EvalResponse)
async def eval(
    req: EvalRequest, principal: Principal = Depends(require_principal)
) -> EvalResponse:
    report = await _run_blocking(functools.partial(run_eval, Path(req.goldset)))
    return EvalResponse(report=report)
