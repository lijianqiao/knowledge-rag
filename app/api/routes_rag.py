"""RAG 端点：/ask /query /status。薄服务层，仅鉴权 + 透传到现有业务函数。"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.auth import Principal, require_principal
from app.api.schemas import (
    AskRequest,
    AskResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from app.graph import run_ask, run_ask_stream
from app.index import format_nodes, get_status, retrieve_nodes

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, principal: Principal = Depends(require_principal)) -> AskResponse:
    answer = run_ask(
        req.question, req.top_k, req.doc_type, allowed_sources=principal.allowed_sources
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
def query(req: QueryRequest, principal: Principal = Depends(require_principal)) -> QueryResponse:
    nodes = retrieve_nodes(
        req.question,
        top_k=req.top_k,
        doc_type=req.doc_type,
        allowed_sources=principal.allowed_sources,
    )
    return QueryResponse(sources=format_nodes(nodes))


@router.get("/status", response_model=StatusResponse)
def status(principal: Principal = Depends(require_principal)) -> StatusResponse:
    return StatusResponse(status=get_status())
