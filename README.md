**English** | [中文](README_zh.md)

# Ops Knowledge Base RAG (LangGraph + LlamaIndex + ChromaDB)

A command-line RAG tool that chunks and vectorizes local Markdown ops docs (`运维文档/`) into ChromaDB, then orchestrates retrieval and Q&A with LangGraph. All LLM / Embedding calls go through an OpenAI-compatible interface, pointing at a local llama.cpp service by default.

## Features

- **Document import**: scans Markdown, chunks by `##` headings + character sliding window, writes to the vector store; supports incremental / rebuild / upsert.
- **Declarative sources**: source list lives in `sources.toml`; add/remove sources without touching Python.
- **Multi-format + web ingest**: a connector registry dispatches `filesystem` (md/pdf/docx/txt …) and `web` (URL fetching).
- **Retrieval debugging**: inspect LlamaIndex → ChromaDB recall results and similarity scores directly.
- **RAG Q&A**: LangGraph orchestrates "retrieve → generate → output", and automatically widens the search and re-retrieves when recall is weak.
- **Pluggable reranker**: defaults to an HTTP rerank API (no local big-model download); can switch back to a local cross-encoder.
- **Multi-query expansion**: optionally let the LLM expand the question into several variants, retrieve in parallel, and fuse.
- **Type filtering**: filter retrieval scope by `doc_type` (declared per source in `sources.toml`).
- **GraphRAG (knowledge-graph retrieval)**: optionally use a local LLM to extract entity-relations from docs into a knowledge graph, retrieving by subgraph to answer relation / impact-chain / root-cause-propagation questions.
- **Question-type routing**: `ask` auto-selects between graph and vector retrieval by question type, safely falling back to vector retrieval on misclassification or when the graph isn't built.
- **Cross-document reasoning Agent**: an explicit plan→act→reflect multi-step loop (LangGraph, not native function-calling) that gathers evidence across docs before answering.
- **Evaluation suite**: the `eval` subcommand runs a gold set and reports deterministic context_recall plus LLM-as-judge faithfulness / relevancy aggregates (self-written, no ragas dependency).
- **Incremental updates**: `import --incremental` re-imports only changed / new files by a content-hash manifest, and cleans up chunks of removed files.
- **Trace logging**: every `ask` / `agent` writes a structured JSONL trace (routing decisions, retrieval scores, rerank TopK, answer length); `scripts/replay.py` replays them.
- **Prompt-injection defense**: retrieved content is neutralized via `sanitize_context` and wrapped in `<<DOC>>`; the system prompt instructs the LLM not to execute instructions inside the material.
- **Streaming output**: `ask --stream` incrementally prints answer tokens (single-pass retrieval, no retry).
- **Semantic cache (optional)**: with `ENABLE_SEMANTIC_CACHE=true`, `ask` returns directly on a question-vector similarity hit; hit/miss is written to trace (off by default).
- **Auto-merging parent-child index (optional)**: with `ENABLE_AUTO_MERGE=true`, `import --force` rebuilds a parent-child index from the same source, and the retrieval hot path merges leaf hits back into the parent chunk (off by default; only when type is `all` and no RBAC allowlist).
- **Native tool calling (optional)**: with `ENABLE_NATIVE_TOOL_CALLING=true`, the Agent prefers the model's native tool-calling, safely falling back to the existing JSON-decide multi-step loop (off by default).

## Tech stack

Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), LlamaIndex, ChromaDB, LangGraph, OpenAI-compatible client (local llama.cpp).

## Installation

```bash
uv sync
```

Create a `.env` in the project root (every variable has a default; override as needed):

```dotenv
# Vector store
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=ops_knowledge

# Embedding model (local llama.cpp)
EMBED_BASE_URL=http://127.0.0.1:8080/v1
EMBED_MODEL=Qwen3-Embedding-0.6B

# Chat model (local llama.cpp)
CHAT_BASE_URL=http://127.0.0.1:8081/v1
CHAT_MODEL=Qwen3.5-9B
LLM_TIMEOUT=600
LLM_MAX_TOKENS=1024

# Retrieval / chunking
RETRIEVE_TOP_K=5
RETRIEVE_SCORE_THRESHOLD=0.35
MAX_RETRIEVE_RETRIES=2
RETRIEVE_CANDIDATE_K=30
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Data sources
SOURCES_CONFIG_PATH=sources.toml

# Reranker (default api: requires a rerank HTTP service at RERANK_BASE_URL)
ENABLE_RERANK=true
RERANK_BACKEND=api                       # api | local
RERANK_BASE_URL=http://127.0.0.1:8082
RERANK_API_MODEL=bge-reranker-v2-m3      # single-model llama.cpp ignores this name; label only
RERANK_MODEL=BAAI/bge-reranker-v2-m3     # used only when RERANK_BACKEND=local (HF format, not GGUF)

# Hybrid retrieval (dense + BM25 fusion)
ENABLE_HYBRID=true
BM25_TOP_K=30

# Query rewrite (on weak retry) / multi-query expansion (on first retrieval)
ENABLE_QUERY_REWRITE=true
ENABLE_MULTI_QUERY=false
MULTI_QUERY_NUM=3

# GraphRAG (off by default: run graph-build first)
ENABLE_GRAPH=false
GRAPH_PERSIST_DIR=./graph_store
GRAPH_RETRIEVE_TOP_K=8
GRAPH_MAX_PATHS_PER_CHUNK=10

# Cross-document reasoning Agent (off by default)
ENABLE_AGENT=false
MAX_AGENT_STEPS=4
ENABLE_NATIVE_TOOL_CALLING=false         # when on, Agent prefers native tool-calling; falls back to JSON-decide

# LLM / Embedding provider switch: local (local llama.cpp) | cloud (OpenAI-compatible endpoint); independent per role
CHAT_PROVIDER=local
EMBED_PROVIDER=local

# Cloud Chat (used when CHAT_PROVIDER=cloud; any OpenAI-compatible vendor)
CLOUD_CHAT_BASE_URL=https://api.deepseek.com/v1
CLOUD_CHAT_MODEL=deepseek-chat
CLOUD_CHAT_API_KEY=

# Cloud Embedding (used when EMBED_PROVIDER=cloud; switching changes vector dim, requires import --force)
CLOUD_EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CLOUD_EMBED_MODEL=text-embedding-v3
CLOUD_EMBED_API_KEY=

# Incremental import manifest (records source-relative-path → content hash; already in .gitignore)
MANIFEST_PATH=./.rag_manifest.json

# Structured trace logging (JSON Lines; on by default, logs/ already in .gitignore)
ENABLE_TRACE=true
TRACE_DIR=./logs

# Semantic cache (off by default; when on, wired into ask, hit/miss written to trace)
ENABLE_SEMANTIC_CACHE=false
CACHE_SIM_THRESHOLD=0.97
CACHE_MAX_SIZE=128

# Auto-merging parent-child index (off by default; when on, import --force builds it, retrieval hot path merges to parent)
ENABLE_AUTO_MERGE=false
AUTO_MERGE_CHUNK_SIZES=2048,512,128
AUTO_MERGE_PERSIST_DIR=./automerge_store
```

The reranker default backend is `api`, requiring a TEI/Jina-style `/rerank` HTTP service at `RERANK_BASE_URL`; set `RERANK_BACKEND=local` to use a local cross-encoder (first use downloads `BAAI/bge-reranker-v2-m3`, ~2.27GB). When offline with no rerank service, set `ENABLE_RERANK=false`. Setting `ENABLE_RERANK`/`ENABLE_HYBRID`/`ENABLE_QUERY_REWRITE`/`ENABLE_MULTI_QUERY` all to `false` reverts to pure dense top_k behavior.

## Usage

```bash
# Import documents
uv run python main.py import                 # first-time / incremental import of everything
uv run python main.py import --force         # drop collection and full rebuild
uv run python main.py import --upsert        # overwrite existing chunks with the same id
uv run python main.py import --incremental   # re-import only changed files by content-hash manifest; clean up removed files
uv run python main.py import --source docs   # import a single source (name from sources.toml)

# Debug retrieval (no LLM call)
uv run python main.py query "how to restart the service" -n 5 --type all

# RAG Q&A (calls the LLM)
uv run python main.py ask "how to troubleshoot order-service 502" -n 5 --type all
uv run python main.py ask "how to troubleshoot order-service 502" --stream   # streaming output (single-pass, no retry)

# Run the gold set (recall + LLM-as-judge faithfulness/relevancy aggregates)
uv run python main.py eval --set eval/goldset.example.json

# Build the knowledge graph (GraphRAG, slow, needs a local LLM; decoupled from vector import)
uv run python main.py graph-build                 # extract entity-relations from all sources
uv run python main.py graph-build --source docs   # build from a single source (name from sources.toml)

# Cross-document reasoning Q&A (multi-step Agent, calls the LLM)
uv run python main.py agent "which services are affected by an order failure" -n 6

# Show vector store status
uv run python main.py status
```

`--type` accepts `all` / `doc`; `-n` sets the number of results returned (for `agent`, `-n` is the per-step retrieval count).

## Smoke testing

> In PowerShell, `$env:X = "true"` only applies to the current terminal session (reset it in a new terminal, or put it in `.env` to make it permanent); after switching `ENABLE_AUTO_MERGE` or `EMBED_PROVIDER`, you must `import --force` to rebuild the index.

```powershell
# 1. Offline logic tests (no model needed)
uv run pytest -q

# 2. Default chain (all switches off; run this first)
uv run python main.py import --force
uv run python main.py status
uv run python main.py health                          # component status JSON
uv run python main.py query "how to restart the service" -n 5    # pure retrieval, no LLM
uv run python main.py ask "how to troubleshoot order-service 502" -n 5
uv run python main.py ask "how to troubleshoot order-service 502" --stream
uv run python main.py eval --set eval/goldset.example.json --format json

# 3. GraphRAG + Agent (build the graph first; slow)
uv run python main.py graph-build --source all
$env:ENABLE_GRAPH = "true";  uv run python main.py ask "which services are affected by an order failure" -n 6
$env:ENABLE_AGENT = "true";  uv run python main.py agent "which services are affected by an order failure" -n 6

# 4. auto-merging (only takes effect after import --force rebuilds the parent-child index)
$env:ENABLE_AUTO_MERGE = "true"
uv run python main.py import --force
uv run python main.py ask "how to troubleshoot order-service 502"

# 5. Native tool calling (depends on whether your chat model truly supports tool_calls; if not, it safely falls back to JSON-decide)
$env:ENABLE_NATIVE_TOOL_CALLING = "true"; $env:ENABLE_AGENT = "true"
uv run python main.py agent "which services are affected by an order failure"
```

**Semantic cache (note: the cache is in-process)**: the cache is a per-process singleton with no disk persistence. Two separate `main.py ask` runs are two processes, so the second one starts with an empty cache and necessarily misses. To observe a hit, both questions must happen **in the same process** — use the long-running API service (set the switch in the same terminal before starting the service):

```powershell
$env:ENABLE_SEMANTIC_CACHE = "true"
uv run python main.py serve
# In another terminal, send the same question twice (only cacheable when doc_type=all and no API-Key):
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"how to troubleshoot order-service 502","top_k":5,"doc_type":"all"}'
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"how to troubleshoot order-service 502","top_k":5,"doc_type":"all"}'
# The second response is noticeably faster, and the corresponding trace shows {"event":"cache","hit":true}
```

## Q&A flow

```
User question
  → received and orchestrated by LangGraph
  → LlamaIndex retrieval (vectorize question → ChromaDB similarity recall → assemble context)
  → LLM generates the answer
  → LangGraph outputs the result
       └─ (if recall is weak or the answer contains "insufficient information" and retries aren't exhausted) widen the search and loop back to retrieval
```

Re-retrieval is governed by `RETRIEVE_SCORE_THRESHOLD` (top recall-score threshold) and `MAX_RETRIEVE_RETRIES` (max retry count).

## GraphRAG and Agent

GraphRAG and vector retrieval are **two decoupled subsystems**, both off by default (`ENABLE_GRAPH`/`ENABLE_AGENT` default `false`; with both off, behavior matches pure vector retrieval).

- **`import` first, then `graph-build`**: the vector store is built by `import`, the knowledge graph by `graph-build` — independent, separately executed. `graph-build` uses a local LLM to extract entity-relations into a graph (ops-domain schema: service / component / fault / root-cause, etc.); it's slow and usually a one-time build, so it's not attached to `import` to avoid slowing imports. Graph artifacts are written to `GRAPH_PERSIST_DIR` (default `./graph_store`, already in `.gitignore`).
- **`ask` auto-routing**: with `ENABLE_GRAPH=true`, `ask` auto-selects by question type between graph retrieval (relations / impact chains / root-cause propagation / cross-document global links) and vector retrieval (single facts / steps / definitions). When routing is uncertain, the LLM errors, or the graph isn't built (`GRAPH_PERSIST_DIR` missing / empty), it always safely falls back to vector retrieval and never crashes.
- **`agent` standalone multi-step entry**: the cross-document reasoning Agent is a separate subcommand using an explicit plan→act→reflect loop (not relying on unreliable native function-calling for local small models), gathering evidence over multiple steps before answering, bounded by `MAX_AGENT_STEPS` (default 4), with vector or graph retrieval tools per step.

| Variable | Default | Description |
|---|---|---|
| `ENABLE_GRAPH` | `false` | when on, `ask` enables question-type routing (graph / vector) |
| `GRAPH_PERSIST_DIR` | `./graph_store` | graph persistence directory (full StorageContext) |
| `GRAPH_RETRIEVE_TOP_K` | `8` | default recall count for graph retrieval |
| `GRAPH_MAX_PATHS_PER_CHUNK` | `10` | max triples extracted per chunk |
| `ENABLE_AGENT` | `false` | cross-document reasoning Agent switch |
| `MAX_AGENT_STEPS` | `4` | max Agent decision steps |

## Cloud LLM

Chat and Embedding can each switch between **local llama.cpp** and a **cloud OpenAI-compatible endpoint**, controlled by `CHAT_PROVIDER` / `EMBED_PROVIDER` (both default `local`). The two providers are independent and can be mixed (e.g. cloud chat + local embed). With no cloud variables configured, behavior is unchanged.

- **Switch to cloud chat**: set `CHAT_PROVIDER=cloud` and configure `CLOUD_CHAT_BASE_URL` / `CLOUD_CHAT_MODEL` / `CLOUD_CHAT_API_KEY`. Works with any OpenAI-compatible vendor: DeepSeek, Tongyi DashScope, Moonshot, Zhipu, OpenAI, etc. (defaults to DeepSeek).
- **Switch to cloud embedding**: set `EMBED_PROVIDER=cloud` and configure `CLOUD_EMBED_BASE_URL` / `CLOUD_EMBED_MODEL` / `CLOUD_EMBED_API_KEY` (defaults to Tongyi DashScope `text-embedding-v3`).

> **Warning**: switching `EMBED_PROVIDER` changes the vector dimension; the old Chroma collection is dimension-incompatible, so you must `uv run python main.py import --force` to reload the vector store before querying. Switching chat only (embedding unchanged) has no such issue.

Minimal example (PowerShell, temporary env vars, current process only; cloud chat only, embedding stays local, no reload needed):

```powershell
$env:CHAT_PROVIDER = "cloud"
$env:CLOUD_CHAT_API_KEY = "sk-your-key"
uv run python main.py ask "how to troubleshoot order-service 502" -n 5 --type all
```

> End-to-end smoke with a real cloud key is an optional manual step (bring your own key); without a key, the local chain works as usual.

## Evaluation and observability

- **Evaluation**: `uv run python main.py eval --set eval/goldset.example.json` iterates the gold set (each item has `question` and `expected_source_substrings`), runs one Q&A per item, and computes three metrics — deterministic `context_recall` (fraction of expected source substrings hit in the retrieved sources), LLM-as-judge `faithfulness` (is the answer supported by context) and `relevancy` (is it on-topic), then prints the aggregate means. The judge uses `get_llm()`, so a stronger cloud model can serve as judge.
- **Trace logging**: with `ENABLE_TRACE=true` (default), every `ask` / `agent` writes key events as JSON Lines to `trace-<id>.jsonl` under `TRACE_DIR` (default `./logs`, gitignored), recording routing decisions, query rewrites before/after, top retrieval score, rerank TopK source+score, answer length, etc. With `ENABLE_TRACE=false`, instrumentation is zero-cost.
- **Replay**: `uv run python scripts/replay.py <trace_id>` reads the corresponding jsonl and prints the full chain, for offline debugging of a given Q&A.
- **Optional retrieval enhancements**: the semantic cache (`app/cache.py`, `ENABLE_SEMANTIC_CACHE`) is wired into `run_ask` (hit/miss written to trace), and the auto-merging parent-child index (`app/automerge.py`, `ENABLE_AUTO_MERGE`) is wired into `import --force` building and the `retrieve_nodes` hot path; both are off by default (opt-in).

## Serving / API

Wraps the RAG engine in an **API-only FastAPI service** (thin layer: auth → call the existing engine → serialize).

```bash
# Start the service (default 127.0.0.1:8000; override with --host/--port or SERVE_HOST/SERVE_PORT)
uv run python main.py serve
uv run python main.py serve --host 0.0.0.0 --port 8000
```

**Auth (API-Key)**: `API_KEYS` is a JSON map `key → {user, allowed_sources}`, e.g.
`API_KEYS={"sk-alice":{"user":"alice","allowed_sources":["运维文档"]}}`.
Requests carry an `X-API-Key` header; on a hit, that user **can only retrieve their `allowed_sources`** (RBAC truly enforced down the retrieval chain).
**Empty `API_KEYS` = open mode (no auth, everything visible), suitable only for local/trusted networks; always configure it when exposing publicly.**

**Endpoints** (curl examples; in auth mode add `-H "X-API-Key: sk-alice"`):

```bash
curl localhost:8000/health
curl -X POST localhost:8000/ask        -H "Content-Type: application/json" -d '{"question":"how to troubleshoot order-service 502","top_k":5,"doc_type":"all"}'
curl -N -X POST localhost:8000/ask/stream -H "Content-Type: application/json" -d '{"question":"..."}'   # SSE streaming
curl -X POST localhost:8000/query      -H "Content-Type: application/json" -d '{"question":"...","top_k":5}'
curl localhost:8000/status
curl -X POST localhost:8000/agent      -H "Content-Type: application/json" -d '{"question":"..."}'   # needs ENABLE_AGENT=true, else 403
curl -X POST localhost:8000/eval       -H "Content-Type: application/json" -d '{"goldset":"eval/goldset.example.json"}'
# Persistent sessions + human-in-the-loop (SQLite checkpointer)
curl -X POST localhost:8000/sessions   -H "Content-Type: application/json" -d '{"question":"...","require_approval":true}'   # returns {status:interrupted, thread_id, pending}
curl -X POST localhost:8000/sessions/<thread_id>/resume -d '{}'   # continue after approval → {status:done, answer}
curl localhost:8000/sessions/<thread_id>
```

**Serving config**: `SERVE_HOST`/`SERVE_PORT`, `API_KEYS`, `REQUEST_TIMEOUT` (soft cap for non-streaming endpoints, 504 on timeout), `CHECKPOINT_DB` (session persistence sqlite, default `./sessions.sqlite`); optional `ENABLE_LANGFUSE` + `LANGFUSE_HOST`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (observability platform, off by default, needs `uv add langfuse`).

> On `REQUEST_TIMEOUT`: blocking calls are offloaded to a thread pool via `anyio.to_thread` to avoid blocking the event loop, but `anyio.fail_after` **cannot forcibly kill** a worker thread already running — what truly prevents an LLM hang is `LLM_TIMEOUT` (httpx timeout). `REQUEST_TIMEOUT` is a soft cap.

## Security

- **Prompt-injection defense (always on)**: retrieved content, when assembling context, is uniformly passed through `sanitize_context` to neutralize common override instructions (CN/EN) and wrapped in `<<DOC>>...<</DOC>>` delimiters; the system prompt explicitly requires the LLM to treat only text inside `<<DOC>>` as material and never execute any instruction within it.
- **Document-level access control (application-layer hook)**: `index.build_access_filters(doc_type, allowed_sources)` builds a metadata filter that adds `source IN allowed` (Chroma has no native RBAC). A multi-user caller can map `user → allowed_sources` and pass it in to scope the documents that user can see.

## Docker

`docker build -t ops-rag .` builds the image (`python:3.14-slim` + uv multi-stage). The default `CMD` is `serve` (binds `0.0.0.0:8000` inside the container):

```bash
docker run -p 8000:8000 --env-file .env ops-rag            # start the API service (default)
docker run --rm --env-file .env ops-rag import --force     # run other subcommands (ENTRYPOINT=main.py)
```

Models are still provided by an external llama.cpp service or cloud OpenAI-compatible endpoint, reachable via environment variables / network.

## Project structure

```
main.py              CLI entry: import / query / ask / graph-build / agent / eval / serve / status
sources.toml         declarative data-source list (filesystem / web)
Dockerfile           python:3.14-slim + uv multi-stage build (default runs serve; models still external)
app/
  config.py          configuration (all from environment variables)
  api/               FastAPI service layer: app (factory + /health), auth (API-Key→Principal),
                     schemas, routes_rag (/ask /query /status /ask/stream /agent /eval),
                     routes_session (sessions + HITL), checkpoint (SqliteSaver), observability (Langfuse adapter, optional)
  sources.py         parse sources.toml → SourceConfig
  connectors/        connector registry: __init__ (protocol + RawDocument + registry), filesystem, web
  loader.py          chunk by format (md → heading chunking, others → sliding window), load via connectors
  rerankers.py       pluggable reranker: api (HTTP) / local (cross-encoder)
  index.py           LlamaIndex + ChromaDB: embedding, retrieval, context assembly, answer generation
  graph.py           LangGraph RAG workflow (with question-type routing integration)
  import_docs.py     chunk → TextNode → write to vector store / build knowledge graph
  graph_store.py     graph persistence directory existence check
  graph_index.py     GraphRAG: PropertyGraphIndex build and graph retrieval
  router.py          retrieval routing: pick graph / vector channel by question type
  agent.py           cross-document reasoning Agent (LangGraph plan→act→reflect loop)
  eval.py            evaluation: context_recall + LLM-as-judge faithfulness/relevancy + run_eval
  manifest.py        content-hash manifest read/write and diff for incremental import
  trace.py           structured JSONL trace logging (new_trace_id / log_event)
  cache.py           in-process semantic cache SemanticCache (optional, ENABLE_SEMANTIC_CACHE, wired into run_ask)
  automerge.py       auto-merging parent-child index (optional, ENABLE_AUTO_MERGE, wired into import --force / retrieve_nodes)
scripts/
  replay.py          replay a chain by trace_id (routing / rewrite / retrieval score / answer)
运维文档/             ops docs source (doc_type=doc)
```

## Adding data sources

Data sources are declared in `sources.toml` in the project root (path overridable via `SOURCES_CONFIG_PATH`). Each `[[sources]]` block's `name` is the value for `import --source`; add/remove sources without touching Python. `type` is dispatched by the connector registry: `filesystem` (local multi-format) and `web` (URL fetching).

```toml
# Local directory source (filesystem)
[[sources]]
name = "docs"
type = "filesystem"
doc_type = "doc"
root = "运维文档"
glob = "**/*.md"          # multi-format: "**/*.{md,pdf,docx}"
exclude = ["README.md"]

# Web source (web)
[[sources]]
name = "oncall-wiki"
type = "web"
doc_type = "doc"
urls = ["https://example.com/runbook"]
```

- Required fields: `name` / `type` / `doc_type`; the rest go in the same block by type (filesystem uses `root`/`glob`/`exclude`, web uses `urls`).
- A chunk's `chunk_id` looks like `{doc_type}_{underscored-relative-path}__{index}`; `--upsert` depends on this id being stable.
- Under a filesystem source root, each top-level subdirectory becomes the `category` metadata. `.md` uses heading chunking; other formats use character sliding-window chunking.
