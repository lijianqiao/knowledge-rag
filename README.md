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
- **GraphRAG（知识图谱检索）**：可选用本地 LLM 从文档抽取实体关系建知识图谱，按子图召回回答关系 / 影响链 / 根因传播类问题。
- **问题类型路由**：`ask` 按问题类型自动在图检索与向量检索之间选通道，误判或图谱未构建时安全回退向量检索。
- **跨文档推理 Agent**：显式 plan→act→reflect 多步循环（LangGraph，非原生 function-calling），跨文档收集证据后作答。

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
CHAT_MODEL=Qwen3.5-9B
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
RERANK_API_MODEL=bge-reranker-v2-m3      # 单模型 llama.cpp 会忽略该名，仅作标签
RERANK_MODEL=BAAI/bge-reranker-v2-m3     # 仅 RERANK_BACKEND=local 时使用（需 HF 格式，非 GGUF）

# Hybrid 检索（稠密 + BM25 融合）
ENABLE_HYBRID=true
BM25_TOP_K=30

# 查询改写（弱重试时）/ 多查询扩展（首检索时）
ENABLE_QUERY_REWRITE=true
ENABLE_MULTI_QUERY=false
MULTI_QUERY_NUM=3

# GraphRAG（默认关闭：需先 graph-build 构建图谱）
ENABLE_GRAPH=false
GRAPH_PERSIST_DIR=./graph_store
GRAPH_RETRIEVE_TOP_K=8
GRAPH_MAX_PATHS_PER_CHUNK=10

# 跨文档推理 Agent（默认关闭）
ENABLE_AGENT=false
MAX_AGENT_STEPS=4

# LLM / Embedding 提供方切换：local（本地 llama.cpp）| cloud（OpenAI 兼容云端点），两角色独立
CHAT_PROVIDER=local
EMBED_PROVIDER=local

# 云端 Chat（CHAT_PROVIDER=cloud 时生效；任一 OpenAI 兼容厂商）
CLOUD_CHAT_BASE_URL=https://api.deepseek.com/v1
CLOUD_CHAT_MODEL=deepseek-chat
CLOUD_CHAT_API_KEY=

# 云端 Embedding（EMBED_PROVIDER=cloud 时生效；切换会改向量维度，需 import --force 重灌）
CLOUD_EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CLOUD_EMBED_MODEL=text-embedding-v3
CLOUD_EMBED_API_KEY=
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

# 构建知识图谱（GraphRAG，慢，需本地 LLM；与向量 import 解耦）
uv run python main.py graph-build                 # 从全部源抽取实体关系建图
uv run python main.py graph-build --source docs   # 仅从某一源构建（源名取自 sources.toml）

# 跨文档推理问答（多步 Agent，调用 LLM）
uv run python main.py agent "订单故障会牵连哪些服务" -n 6

# 查看向量库状态
uv run python main.py status
```

`--type` 可选 `all` / `prompt` / `doc`；`-n` 指定返回条数（`agent` 的 `-n` 为每步检索条数）。

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

## GraphRAG 与 Agent

GraphRAG 与向量检索是**两套解耦的子系统**，默认全部关闭（`ENABLE_GRAPH`/`ENABLE_AGENT` 默认 `false`，全关时行为与纯向量检索一致）。

- **先 `import` 再 `graph-build`**：向量库由 `import` 构建，知识图谱由 `graph-build` 单独构建，两者独立、各自执行。`graph-build` 用本地 LLM 从文档抽取实体关系建图（运维领域 schema：服务 / 组件 / 故障 / 根因 等），过程慢、通常一次性构建，故不挂在 `import` 上以免拖慢导入。图谱产物写入 `GRAPH_PERSIST_DIR`（默认 `./graph_store`，已在 `.gitignore`）。
- **`ask` 自动路由**：设 `ENABLE_GRAPH=true` 后，`ask` 会按问题类型自动在图检索（关系 / 影响链 / 根因传播 / 跨文档全局关联）与向量检索（单点事实 / 操作步骤 / 定义）之间选通道。路由分类不确定、LLM 出错，或图谱尚未构建（`GRAPH_PERSIST_DIR` 缺失 / 为空）时，一律安全回退到向量检索，不会崩。
- **`agent` 独立多步入口**：跨文档推理 Agent 是独立子命令，用显式 plan→act→reflect 循环（不依赖本地小模型不可靠的原生 function-calling）多步收集证据后作答，步数受 `MAX_AGENT_STEPS`（默认 4）约束，每步可调向量或图检索工具。

| 变量 | 默认 | 说明 |
|---|---|---|
| `ENABLE_GRAPH` | `false` | 开启后 `ask` 启用问题类型路由（图 / 向量） |
| `GRAPH_PERSIST_DIR` | `./graph_store` | 图谱持久化目录（整 StorageContext） |
| `GRAPH_RETRIEVE_TOP_K` | `8` | 图检索默认召回条数 |
| `GRAPH_MAX_PATHS_PER_CHUNK` | `10` | 每个 chunk 抽取的最大三元组数 |
| `ENABLE_AGENT` | `false` | 跨文档推理 Agent 开关 |
| `MAX_AGENT_STEPS` | `4` | Agent 最大决策步数 |

## 云端 LLM

Chat 与 Embedding 可各自在**本地 llama.cpp** 与**云端 OpenAI 兼容端点**之间切换，由 `CHAT_PROVIDER` / `EMBED_PROVIDER` 控制（默认均为 `local`）。两个 provider 相互独立，可混用（如云 chat + 本地 embed）。不配置任何云变量时行为与之前完全一致。

- **切云端 chat**：设 `CHAT_PROVIDER=cloud`，并配 `CLOUD_CHAT_BASE_URL` / `CLOUD_CHAT_MODEL` / `CLOUD_CHAT_API_KEY`。可对接任一 OpenAI 兼容厂商：DeepSeek、通义 DashScope、Moonshot、智谱、OpenAI 等（默认值为 DeepSeek）。
- **切云端 embedding**：设 `EMBED_PROVIDER=cloud`，并配 `CLOUD_EMBED_BASE_URL` / `CLOUD_EMBED_MODEL` / `CLOUD_EMBED_API_KEY`（默认值为通义 DashScope `text-embedding-v3`）。

> **警告**：切换 `EMBED_PROVIDER` 会改变向量维度，旧 Chroma collection 维度不兼容，必须 `uv run python main.py import --force` 重灌向量库后才能查询。仅切 chat（embedding 不变）无此问题。

最小示例（PowerShell，临时环境变量，仅本进程生效；只切云 chat、embedding 仍走本地，无需重灌）：

```powershell
$env:CHAT_PROVIDER = "cloud"
$env:CLOUD_CHAT_API_KEY = "sk-你的key"
uv run python main.py ask "订单服务 502 怎么排查" -n 5 --type all
```

> 真实云 key 的端到端冒烟为可选手测步骤（需自备 key）；不配 key 时本地链路照常工作。

## 项目结构

```
main.py              CLI 入口：import / query / ask / graph-build / agent / status
sources.toml         声明式数据源清单（filesystem / web）
app/
  config.py          配置（全部来自环境变量）
  sources.py         解析 sources.toml → SourceConfig
  connectors/        连接器注册表：__init__（协议 + RawDocument + 注册表）、filesystem、web
  loader.py          按格式分块（md 走标题分块，其余走滑窗），经连接器加载
  rerankers.py       可插拔 Reranker：api（HTTP）/ local（cross-encoder）
  index.py           LlamaIndex + ChromaDB：Embedding、检索、context 组装、答案生成
  graph.py           LangGraph RAG 工作流（含问题类型路由接入）
  import_docs.py     分块 → TextNode → 写入向量库 / 构建知识图谱
  graph_store.py     图谱持久化目录存在性判定
  graph_index.py     GraphRAG：PropertyGraphIndex 构建与图检索
  router.py          检索路由：按问题类型选 graph / vector 通道
  agent.py           跨文档推理 Agent（LangGraph plan→act→reflect 循环）
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

## 扩展性与向量库选型（预案）

> 决定要不要换"重型向量库"的是**数据量与并发**，不是公司人数。当前架构（ChromaDB 稠密 + 进程内 `bm25s`+jieba 稀疏 + reranker）对**几千人规模、低 QPS 的运维知识库足够**，无需升级。

**何时才考虑升级（量化触发阈值，满足任一）**

- chunk 总量 > ~100 万，或单机内存吃紧 / 启动明显变慢；
- 持续并发 QPS > ~20–50；
- 需要**学习式稀疏（BGE-M3）**进一步提升中文召回（ChromaDB 不支持稀疏向量 ANN）；
- 需要多副本 / 高可用 / 在线扩容 / 快照。

**升级路径优先级**

1. **pgvector**（本仓库环境已有 PostgreSQL）→ 零新增服务，复用现有 Postgres 存稠密；要稀疏/BM25 可上 `VectorChord`/`pgvecto.rs`。最低摩擦。
2. **Qdrant**（向量原生、轻量：单二进制 / Docker / 嵌入式 local 模式）→ **仅当要做 BGE-M3 学习式稀疏 hybrid 时选它**（这是它相对 Chroma 的唯一强理由：原生稀疏 + server 端融合）。
3. **Elasticsearch/OpenSearch + IK 分词** → 要做全公司级中文全文检索平台时。
4. **Milvus** → 千万级向量 / 分布式才考虑，最重。

**Qdrant 迁移预案（真要换时照此做）**

- 触发条件：决定上 BGE-M3 学习式稀疏 hybrid。只为"换个更好的稠密库"不值得迁。
- 依赖：`qdrant-client`、`llama-index-vector-stores-qdrant`；可先用 local 模式 `QdrantClient(path=...)`（Windows 免 Docker），生产单机 Docker。
- 改动点（集中在 `app/index.py`）：`get_chroma_*`/`get_index`（`ChromaVectorStore`→`QdrantVectorStore`）、`delete_chroma_collection`、`load_all_nodes`（`chroma.get()`→Qdrant scroll）、`get_status`；新增 `QDRANT_*` 配置；**数据需重灌**。
- 稀疏来源：用 FastEmbed(ONNX) 出 BGE-M3/SPLADE 稀疏 —— 代价是 **~2GB 进程内模型**（与当前 llama.cpp 纯服务化相悖，需接受）。
- 灰度与回退：加 `VECTOR_BACKEND=chroma|qdrant` 开关，保留 Chroma 路径，可回退；用同一批问题对比召回/答案质量，确认提升再全量切。

**结论**：当前规模不换。要升级优先 **pgvector（复用 Postgres）**；只有为 **BGE-M3 学习式稀疏**才上 **Qdrant**；ES/Milvus 留给平台级规模。
