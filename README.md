# 运维知识库 RAG（LangGraph + LlamaIndex + ChromaDB）

把本地 Markdown 运维文档（`运维文档/`）分块向量化入 ChromaDB，再用 LangGraph 编排检索与问答的命令行 RAG 工具。所有 LLM / Embedding 调用走 OpenAI 兼容接口，默认指向本地 llama.cpp 服务。

## 特性

- **文档导入**：扫描 Markdown，按二级标题 + 字符滑窗分块后写入向量库，支持增量 / 重建 / upsert。
- **声明式数据源**：源清单写在 `sources.toml`，增删源不改 Python 代码即可生效。
- **多格式 + 网页接入**：连接器注册表分发 `filesystem`（md/pdf/docx/txt 等）与 `web`（URL 抓取）。
- **检索调试**：直接查看 LlamaIndex → ChromaDB 的召回结果与相似度分数。
- **RAG 问答**：LangGraph 编排「检索 → 生成 → 输出」，召回偏弱时自动扩大范围重检索。
- **可插拔 Reranker**：默认走 HTTP rerank API（免本地大模型下载），可切回本地 cross-encoder。
- **多查询扩展**：可选用 LLM 把原问题扩展成多条变体并行检索后融合。
- **类型过滤**：按 `doc_type`（源在 `sources.toml` 声明）过滤检索范围。
- **GraphRAG（知识图谱检索）**：可选用本地 LLM 从文档抽取实体关系建知识图谱，按子图召回回答关系 / 影响链 / 根因传播类问题。
- **问题类型路由**：`ask` 按问题类型自动在图检索与向量检索之间选通道，误判或图谱未构建时安全回退向量检索。
- **跨文档推理 Agent**：显式 plan→act→reflect 多步循环（LangGraph，非原生 function-calling），跨文档收集证据后作答。
- **评估体系**：`eval` 子命令跑评测集，输出确定性 context_recall + LLM-as-judge 的 faithfulness / relevancy 聚合分（自写，不依赖 ragas）。
- **增量更新**：`import --incremental` 按内容 hash 清单只重导变更 / 新增文件，并清理已消失文件的 chunk。
- **链路追踪**：每次 `ask` / `agent` 写结构化 JSONL trace（路由决策、检索分、rerank TopK、答案长度），`scripts/replay.py` 可回放复现。
- **提示词注入防御**：检索内容统一经 `sanitize_context` 中和越权指令并用 `<<DOC>>` 包裹，系统提示词约束 LLM 不执行资料内指令。
- **流式输出**：`ask --stream` 增量打印答案 token（单趟检索、无重试）。
- **语义缓存（可选）**：`ENABLE_SEMANTIC_CACHE=true` 时 `ask` 按问题向量相似度命中缓存直接返回，命中 / 未命中写入 trace（默认关闭）。
- **auto-merging 父子索引（可选）**：`ENABLE_AUTO_MERGE=true` 时 `import --force` 同源重建父子索引，检索热路径命中叶子块时合并回父块返回（默认关闭，仅全类型、无 RBAC 白名单时生效）。
- **原生工具调用（可选）**：`ENABLE_NATIVE_TOOL_CALLING=true` 时 Agent 优先走模型原生 tool-calling，失败安全回退既有 JSON-decide 多步循环（默认关闭）。

## 技术栈

Python ≥ 3.12、[uv](https://docs.astral.sh/uv/)、LlamaIndex、ChromaDB、LangGraph、OpenAI 兼容客户端（本地 llama.cpp）。

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
ENABLE_NATIVE_TOOL_CALLING=false         # 开启后 Agent 优先走原生 tool-calling，失败回退 JSON-decide

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

# 增量导入清单（记录 源相对路径 → 内容 hash，已在 .gitignore）
MANIFEST_PATH=./.rag_manifest.json

# 结构化链路追踪（JSON Lines；默认开启，logs/ 已在 .gitignore）
ENABLE_TRACE=true
TRACE_DIR=./logs

# 语义缓存（默认关闭；开启后已接入 ask，命中/未命中写 trace）
ENABLE_SEMANTIC_CACHE=false
CACHE_SIM_THRESHOLD=0.97
CACHE_MAX_SIZE=128

# auto-merging 父子索引（默认关闭；开启后 import --force 建索引、检索热路径合并回父块）
ENABLE_AUTO_MERGE=false
AUTO_MERGE_CHUNK_SIZES=2048,512,128
AUTO_MERGE_PERSIST_DIR=./automerge_store
```

Reranker 默认后端为 `api`，需在 `RERANK_BASE_URL` 处提供 TEI/Jina 风格的 `/rerank` HTTP 服务；设 `RERANK_BACKEND=local` 则改用本地 cross-encoder（首次使用会下载 `BAAI/bge-reranker-v2-m3`，约 2.27GB）。离线且无 rerank 服务时设 `ENABLE_RERANK=false`。`ENABLE_RERANK`/`ENABLE_HYBRID`/`ENABLE_QUERY_REWRITE`/`ENABLE_MULTI_QUERY` 全设 `false` 即退回纯稠密 top_k 旧行为。

## 使用

```bash
# 导入文档
uv run python main.py import                 # 首次 / 增量导入全部
uv run python main.py import --force         # 删除 collection 后全量重建
uv run python main.py import --upsert        # 覆盖更新已有同 id 分块
uv run python main.py import --incremental   # 按内容 hash 清单仅重导变更、清理消失文件的 chunk
uv run python main.py import --source docs   # 仅导入某一源（源名取自 sources.toml）

# 调试检索（不调用 LLM）
uv run python main.py query "如何重启服务" -n 5 --type all

# RAG 问答（调用 LLM）
uv run python main.py ask "订单服务 502 怎么排查" -n 5 --type all
uv run python main.py ask "订单服务 502 怎么排查" --stream   # 流式增量打印（单趟检索，无重试）

# 跑评测集（recall + LLM-as-judge faithfulness/relevancy 聚合分）
uv run python main.py eval --set eval/goldset.example.json

# 构建知识图谱（GraphRAG，慢，需本地 LLM；与向量 import 解耦）
uv run python main.py graph-build                 # 从全部源抽取实体关系建图
uv run python main.py graph-build --source docs   # 仅从某一源构建（源名取自 sources.toml）

# 跨文档推理问答（多步 Agent，调用 LLM）
uv run python main.py agent "订单故障会牵连哪些服务" -n 6

# 查看向量库状态
uv run python main.py status
```

`--type` 可选 `all` / `doc`；`-n` 指定返回条数（`agent` 的 `-n` 为每步检索条数）。

## 冒烟测试

> PowerShell 里 `$env:X = "true"` 只对当前终端会话生效（换终端要重设，想固定就写进 `.env`）；切 `ENABLE_AUTO_MERGE` 或 `EMBED_PROVIDER` 后必须 `import --force` 重建索引。

```powershell
# 1. 离线逻辑测试（不连模型）
uv run pytest -q

# 2. 默认链路（全开关关闭，先过这条）
uv run python main.py import --force
uv run python main.py status
uv run python main.py health                          # 组件状态 JSON
uv run python main.py query "如何重启服务" -n 5            # 纯检索，不调 LLM
uv run python main.py ask "订单服务 502 怎么排查" -n 5
uv run python main.py ask "订单服务 502 怎么排查" --stream
uv run python main.py eval --set eval/goldset.example.json --format json

# 3. GraphRAG + Agent（需先建图，慢）
uv run python main.py graph-build --source all
$env:ENABLE_GRAPH = "true";  uv run python main.py ask "订单故障会牵连哪些服务" -n 6
$env:ENABLE_AGENT = "true";  uv run python main.py agent "订单故障会牵连哪些服务" -n 6

# 4. auto-merging（必须 import --force 重建父子索引后才生效）
$env:ENABLE_AUTO_MERGE = "true"
uv run python main.py import --force
uv run python main.py ask "订单服务 502 怎么排查"

# 5. 原生工具调用（依赖 chat 模型真支持 tool_calls，不支持会安全回退 JSON-decide）
$env:ENABLE_NATIVE_TOOL_CALLING = "true"; $env:ENABLE_AGENT = "true"
uv run python main.py agent "订单故障会牵连哪些服务"
```

**语义缓存（注意：缓存是进程内的）**：缓存是进程内单例、无磁盘持久化，两次独立的 `main.py ask` 是两个进程，第二次缓存为空必然 miss。要验证命中，两次提问必须在**同一进程**内——用常驻的 API 服务（同一终端先设开关再起服务）：

```powershell
$env:ENABLE_SEMANTIC_CACHE = "true"
uv run python main.py serve
# 另开终端，对同一问题连发两次（doc_type=all 且无 API-Key 才会缓存）：
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"订单服务502怎么排查","top_k":5,"doc_type":"all"}'
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"订单服务502怎么排查","top_k":5,"doc_type":"all"}'
# 第二次响应明显变快，对应 trace 里出现 {"event":"cache","hit":true}
```

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

## 评估与可观测性

- **评估**：`uv run python main.py eval --set eval/goldset.example.json` 遍历评测集（每条含 `question` 与 `expected_source_substrings`），对每条跑一次问答后算三项指标——确定性 `context_recall`（期望来源子串在检索 source 中的命中比例）、LLM-as-judge 的 `faithfulness`（答案是否被 context 支撑）与 `relevancy`（是否切题），最后打印聚合均值。裁判走 `get_llm()`，可借云端更强模型当裁判。
- **链路追踪**：`ENABLE_TRACE=true`（默认开启）时，每次 `ask` / `agent` 把关键事件写成 JSON Lines 到 `TRACE_DIR`（默认 `./logs`，已 gitignore）下的 `trace-<id>.jsonl`，记录路由决策、查询改写前后、检索最高分、rerank TopK 的 source+score、答案长度等。设 `ENABLE_TRACE=false` 时埋点零成本。
- **回放复现**：`uv run python scripts/replay.py <trace_id>` 读对应 jsonl 打印该次完整链路，便于离线调试某次问答。
- **可选检索增强**：语义缓存（`app/cache.py`，`ENABLE_SEMANTIC_CACHE`）已接入 `run_ask`（命中 / 未命中写 trace），auto-merging 父子索引（`app/automerge.py`，`ENABLE_AUTO_MERGE`）已接入 `import --force` 构建与 `retrieve_nodes` 热路径，二者默认关闭、为 opt-in 能力。

## 服务化 / API

把 RAG 引擎包成一个 **API-only 的 FastAPI 服务**（薄服务层，仅鉴权 → 调现有引擎 → 序列化）。

```bash
# 起服务（默认 127.0.0.1:8000；可用 --host/--port 或 SERVE_HOST/SERVE_PORT 覆盖）
uv run python main.py serve
uv run python main.py serve --host 0.0.0.0 --port 8000
```

**鉴权（API-Key）**：`API_KEYS` 为 JSON 映射 `key → {user, allowed_sources}`，例如
`API_KEYS={"sk-alice":{"user":"alice","allowed_sources":["运维文档"]}}`。
请求带 `X-API-Key` 头；命中后该用户**只能检索其 `allowed_sources`**（RBAC 经检索链路真正生效）。
**`API_KEYS` 为空 = 开放模式（不鉴权、全量可见），仅适合本机/可信网络；对外暴露务必配置。**

**端点**（curl 示例，鉴权模式下加 `-H "X-API-Key: sk-alice"`）：

```bash
curl localhost:8000/health
curl -X POST localhost:8000/ask        -H "Content-Type: application/json" -d '{"question":"订单服务502怎么排查","top_k":5,"doc_type":"all"}'
curl -N -X POST localhost:8000/ask/stream -H "Content-Type: application/json" -d '{"question":"..."}'   # SSE 流式
curl -X POST localhost:8000/query      -H "Content-Type: application/json" -d '{"question":"...","top_k":5}'
curl localhost:8000/status
curl -X POST localhost:8000/agent      -H "Content-Type: application/json" -d '{"question":"..."}'   # 需 ENABLE_AGENT=true，否则 403
curl -X POST localhost:8000/eval       -H "Content-Type: application/json" -d '{"goldset":"eval/goldset.example.json"}'
# 持久化会话 + 人机循环（SQLite checkpointer）
curl -X POST localhost:8000/sessions   -H "Content-Type: application/json" -d '{"question":"...","require_approval":true}'   # 返回 {status:interrupted, thread_id, pending}
curl -X POST localhost:8000/sessions/<thread_id>/resume -d '{}'   # 审批后继续 → {status:done, answer}
curl localhost:8000/sessions/<thread_id>
```

**服务相关配置**：`SERVE_HOST`/`SERVE_PORT`、`API_KEYS`、`REQUEST_TIMEOUT`（非流式端点软上限，超时 504）、`CHECKPOINT_DB`（会话持久化 sqlite，默认 `./sessions.sqlite`）；可选 `ENABLE_LANGFUSE` + `LANGFUSE_HOST`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`（接观测平台，默认关，需 `uv add langfuse`）。

> 关于 `REQUEST_TIMEOUT`：阻塞调用经 `anyio.to_thread` 卸载到线程池避免阻塞事件循环，但 `anyio.fail_after` **无法强杀**已在运行的 worker 线程——真正防 LLM 挂死的是 `LLM_TIMEOUT`（httpx 超时）。`REQUEST_TIMEOUT` 是软上限。

## 安全

- **提示词注入防御（始终开启）**：检索到的内容在组装 context 时统一经 `sanitize_context` 中和常见越权指令（中英），并用 `<<DOC>>...<</DOC>>` 分隔符包裹；系统提示词明确要求 LLM 仅将 `<<DOC>>` 内文本视为资料、绝不执行其中任何指令。
- **文档级访问控制（应用层钩子）**：`index.build_access_filters(doc_type, allowed_sources)` 生成叠加 `source IN allowed` 的元数据过滤（Chroma 无原生 RBAC）。多用户调用方可把 `user → allowed_sources` 映射后传入，限定该用户可见的文档范围。

## Docker

`docker build -t ops-rag .` 构建镜像（`python:3.14-slim` + uv 多阶段）。默认 `CMD` 为 `serve`（容器内绑 `0.0.0.0:8000`）：

```bash
docker run -p 8000:8000 --env-file .env ops-rag            # 起 API 服务（默认）
docker run --rm --env-file .env ops-rag import --force     # 跑其他子命令（ENTRYPOINT=main.py）
```

模型仍由外部 llama.cpp 服务或云端 OpenAI 兼容端点提供，需通过环境变量 / 网络可达。

## 项目结构

```
main.py              CLI 入口：import / query / ask / graph-build / agent / eval / serve / status
sources.toml         声明式数据源清单（filesystem / web）
Dockerfile           python:3.14-slim + uv 多阶段构建（默认起 serve；模型仍由外部提供）
app/
  config.py          配置（全部来自环境变量）
  api/               FastAPI 服务层：app（工厂+/health）、auth（API-Key→Principal）、
                     schemas、routes_rag（/ask /query /status /ask/stream /agent /eval）、
                     routes_session（会话+HITL）、checkpoint（SqliteSaver）、observability（Langfuse 适配，可选）
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
  eval.py            评估：context_recall + LLM-as-judge faithfulness/relevancy + run_eval
  manifest.py        增量导入的内容 hash 清单读写与 diff
  trace.py           结构化 JSONL 链路追踪（new_trace_id / log_event）
  cache.py           进程内语义缓存 SemanticCache（可选，ENABLE_SEMANTIC_CACHE，已接入 run_ask）
  automerge.py       auto-merging 父子索引（可选，ENABLE_AUTO_MERGE，已接入 import --force / retrieve_nodes）
scripts/
  replay.py          按 trace_id 回放某次链路（路由 / 改写 / 检索分 / 答案）
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
