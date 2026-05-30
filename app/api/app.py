"""FastAPI app 工厂：薄服务层，仅挂载路由与生命周期。"""

from fastapi import FastAPI

from app.api import routes_rag, routes_session


def create_app() -> FastAPI:
    app = FastAPI(title="运维知识库 RAG API", version="5.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(routes_rag.router)
    app.include_router(routes_session.router)
    return app
