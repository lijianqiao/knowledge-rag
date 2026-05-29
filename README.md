# 运维知识库 RAG（LangGraph + LlamaIndex + ChromaDB）

把本地 Markdown 运维文档（`运维prompt库/`、`运维文档/`）分块向量化入 ChromaDB，再用 LangGraph 编排检索与问答的命令行 RAG 工具。所有 LLM / Embedding 调用走 OpenAI 兼容接口，默认指向本地 llama.cpp 服务。

## 特性

- **文档导入**：扫描 Markdown，按二级标题 + 字符滑窗分块后写入向量库，支持增量 / 重建 / upsert。
- **声明式数据源**：源清单写在 `sources.toml`，增删源不改 Python 代码即可生效。
- **多格式 + 网页接入**：连接器注册表分发 `filesystem`（md/pdf/docx/txt 等）与 `web`（URL 抓取）。
- **检索调试**：直接查看 LlamaIndex → ChromaDB 的召回结果与相似度分数。
- **RAG 问答**：LangGraph 编排「检索 → 生成 → 输出」，召回偏弱时自动扩大范围重检索。
- **可插拔 Reranker**：默认走 HTTP rerank API（免本地大模型下载），可切回本地 cross-encoder。
- **多查询扩展**：可选用 LLM 把原问题扩展成多条变体并行检索后融合。
- **类型过滤**：按 `prompt` / `doc` 文档类型过滤检索范围。

## 技术栈

Python ≥ 3.14、[uv](https://docs.astral.sh/uv/)、LlamaIndex、ChromaDB、LangGraph、OpenAI 兼容客户端（本地 llama.cpp）。

## 安装

```bash
uv sync
```

在项目根目录创建 `.env`（所有变量均有默认值，按需覆盖）：

```dotenv
# 向量库
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=ops_knowledge

# Embedding 模型（本地 llama.cpp）
EMBED_BASE_URL=http://127.0.0.1:8080/v1
EMBED_MODEL=Qwen3-Embedding-0.6B

# 对话模型（本地 llama.cpp）
CHAT_BASE_URL=http://127.0.0.1:8081/v1
CHAT_MODEL=Qwen3-4B-Instruct
LLM_TIMEOUT=600
LLM_MAX_TOKENS=1024

# 检索 / 分块
RETRIEVE_TOP_K=5
RETRIEVE_SCORE_THRESHOLD=0.35
MAX_RETRIEVE_RETRIES=2
RETRIEVE_CANDIDATE_K=30
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# 数据源
SOURCES_CONFIG_PATH=sources.toml

# Reranker（默认 api：需 RERANK_BASE_URL 处有 rerank HTTP 服务）
ENABLE_RERANK=true
RERANK_BACKEND=api                       # api | local
RERANK_BASE_URL=http://127.0.0.1:8082
RERANK_API_MODEL=Qwen3-Reranker-0.6B
RERANK_MODEL=BAAI/bge-reranker-v2-m3     # 仅 RERANK_BACKEND=local 时使用

# Hybrid 检索（稠密 + BM25 融合）
ENABLE_HYBRID=true
BM25_TOP_K=30

# 查询改写（弱重试时）/ 多查询扩展（首检索时）
ENABLE_QUERY_REWRITE=true
ENABLE_MULTI_QUERY=false
MULTI_QUERY_NUM=3
```

Reranker 默认后端为 `api`，需在 `RERANK_BASE_URL` 处提供 TEI/Jina 风格的 `/rerank` HTTP 服务；设 `RERANK_BACKEND=local` 则改用本地 cross-encoder（首次使用会下载 `BAAI/bge-reranker-v2-m3`，约 2.27GB）。离线且无 rerank 服务时设 `ENABLE_RERANK=false`。`ENABLE_RERANK`/`ENABLE_HYBRID`/`ENABLE_QUERY_REWRITE`/`ENABLE_MULTI_QUERY` 全设 `false` 即退回纯稠密 top_k 旧行为。

## 使用

```bash
# 导入文档
uv run python main.py import                 # 首次 / 增量导入全部
uv run python main.py import --force         # 删除 collection 后全量重建
uv run python main.py import --upsert        # 覆盖更新已有同 id 分块
uv run python main.py import --source docs   # 仅导入某一源（源名取自 sources.toml）

# 调试检索（不调用 LLM）
uv run python main.py query "如何重启服务" -n 5 --type all

# RAG 问答（调用 LLM）
uv run python main.py ask "订单服务 502 怎么排查" -n 5 --type all

# 查看向量库状态
uv run python main.py status
```

`--type` 可选 `all` / `prompt` / `doc`；`-n` 指定返回条数。

## 问答流程

```
用户提问
  → LangGraph 接收并编排
  → LlamaIndex 发起检索（问题向量化 → ChromaDB 相似度召回 → 组装 context）
  → LLM 生成答案
  → LangGraph 输出结果
       └─（召回偏弱或答案含「信息不足」且未超重试上限时）扩大检索范围回到检索
```

重检索由 `RETRIEVE_SCORE_THRESHOLD`（召回最高分阈值）与 `MAX_RETRIEVE_RETRIES`（最大重试次数）控制。

## 项目结构

```
main.py              CLI 入口：import / query / ask / status
sources.toml         声明式数据源清单（filesystem / web）
app/
  config.py          配置（全部来自环境变量）
  sources.py         解析 sources.toml → SourceConfig
  connectors/        连接器注册表：__init__（协议 + RawDocument + 注册表）、filesystem、web
  loader.py          按格式分块（md 走标题分块，其余走滑窗），经连接器加载
  rerankers.py       可插拔 Reranker：api（HTTP）/ local（cross-encoder）
  index.py           LlamaIndex + ChromaDB：Embedding、检索、context 组装、答案生成
  graph.py           LangGraph RAG 工作流
  import_docs.py     分块 → TextNode → 写入向量库
运维prompt库/         prompt 文档源（doc_type=prompt）
运维文档/             运维文档源（doc_type=doc）
```

## 数据源接入

数据源在项目根目录的 `sources.toml` 中声明（路径可由 `SOURCES_CONFIG_PATH` 覆盖）。每个 `[[sources]]` 块的 `name` 即 `import --source` 的取值；增删源不改 Python 代码即可生效。`type` 由连接器注册表分发：`filesystem`（本地多格式）与 `web`（URL 抓取）。

```toml
# 本地目录源（filesystem）
[[sources]]
name = "docs"
type = "filesystem"
doc_type = "doc"
root = "运维文档"
glob = "**/*.md"          # 多格式可写 "**/*.{md,pdf,docx}"
exclude = ["README.md"]

# 网页源（web）
[[sources]]
name = "oncall-wiki"
type = "web"
doc_type = "doc"
urls = ["https://example.com/runbook"]
```

- 必填字段：`name` / `type` / `doc_type`；其余按类型放在同一块（filesystem 用 `root`/`glob`/`exclude`，web 用 `urls`）。
- 分块 `chunk_id` 形如 `{doc_type}_{相对路径下划线化}__{序号}`，`--upsert` 依赖该 id 稳定。
- filesystem 源根目录下每个一级子目录会作为 `category` 元数据。`.md` 走二级标题分块，其余格式走字符滑窗分块。
