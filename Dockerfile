# 运维知识库 RAG —— python:3.14-slim + uv 多阶段构建
# 注意：本镜像只含应用代码与 Python 依赖；LLM/Embedding/Reranker 由外部
# llama.cpp 或云端 OpenAI 兼容端点提供（容器内不跑模型）。

# ---------- builder：用 uv 解析并安装锁定依赖到 /app/.venv ----------
FROM python:3.14-slim AS builder

# 装 uv（从官方镜像拷二进制，免 pip）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 先拷依赖清单，利用缓存层（仅 lock/pyproject 变化才重装）
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# 再拷源码并安装项目本身
COPY . .
RUN uv sync --no-dev --frozen

# ---------- runtime：仅拷虚拟环境 + 源码，体积小 ----------
FROM python:3.14-slim AS runtime

WORKDIR /app

# 拷 builder 的 venv 与应用代码
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# 用 venv 内解释器，免装 uv
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "main.py"]
CMD ["status"]
