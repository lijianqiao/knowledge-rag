"""FastAPI app 工厂：薄服务层，仅挂载路由与生命周期。"""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="运维知识库 RAG API", version="5.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # 路由在后续 Task 挂载：app.include_router(...)
    return app
