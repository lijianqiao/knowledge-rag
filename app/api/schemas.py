"""Pydantic 请求/响应模型：薄服务层的输入校验与序列化。"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_type: str = "all"


class AskResponse(BaseModel):
    answer: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_type: str = "all"


class QueryResponse(BaseModel):
    sources: str


class StatusResponse(BaseModel):
    status: str


class AgentRequest(BaseModel):
    question: str
    top_k: int = 5


class AgentResponse(BaseModel):
    answer: str


class EvalRequest(BaseModel):
    goldset: str = "eval/goldset.example.json"


class EvalResponse(BaseModel):
    report: str
