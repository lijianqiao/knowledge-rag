# 运维知识库 RAG（LangGraph + LlamaIndex + ChromaDB）

把本地 Markdown 运维文档（`运维prompt库/`、`运维文档/`）分块向量化入 ChromaDB，再用 LangGraph 编排检索与问答的命令行 RAG 工具。所有 LLM / Embedding 调用走 OpenAI 兼容接口，默认指向本地 llama.cpp 服务。

## 特性

- **文档导入**：扫描 Markdown，按二级标题 + 字符滑窗分块后写入向量库，支持增量 / 重建 / upsert。
- **检索调试**：直接查看 LlamaIndex → ChromaDB 的召回结果与相似度分数。
- **RAG 问答**：LangGraph 编排「检索 → 生成 → 输出」，召回偏弱时自动扩大范围重检索。
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
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

## 使用

```bash
# 导入文档
uv run python main.py import                 # 首次 / 增量导入全部
uv run python main.py import --force         # 删除 collection 后全量重建
uv run python main.py import --upsert        # 覆盖更新已有同 id 分块
uv run python main.py import --source docs   # 仅导入某一源（all / prompts / docs）

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
app/
  config.py          配置与文档源定义（全部来自环境变量）
  loader.py          Markdown 扫描与分块（纯文本，无外部依赖）
  index.py           LlamaIndex + ChromaDB：Embedding、检索、context 组装、答案生成
  graph.py           LangGraph RAG 工作流
  import_docs.py     分块 → TextNode → 写入向量库
运维prompt库/         prompt 文档源（doc_type=prompt）
运维文档/             运维文档源（doc_type=doc）
```

## 文档源约定

- 新增文档源在 `app/config.py` 的 `DOCUMENT_SOURCES` 中配置：根目录、`doc_type` 标签、跳过文件。
- 分块 `chunk_id` 形如 `{doc_type}_{相对路径下划线化}__{序号}`，`--upsert` 依赖该 id 稳定。
- 文档源根目录下每个一级子目录会作为 `category` 元数据。
