"""Native OpenAI-compatible tool schemas and dispatch for the Agent path."""

from llama_index.core.schema import NodeWithScore

from app.graph_index import graph_retrieve
from app.index import retrieve_nodes


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "Search operational documents using hybrid vector/BM25 retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_search",
                "description": (
                    "Search the knowledge graph for service relationships, dependencies, "
                    "and root-cause paths."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def dispatch_tool(
    name: str,
    query: str,
    top_k: int,
    allowed_sources: list | None = None,
) -> list[NodeWithScore]:
    if name == "vector_search":
        return retrieve_nodes(query, top_k=top_k, allowed_sources=allowed_sources)
    if name == "graph_search":
        return graph_retrieve(query, top_k=top_k)
    raise ValueError(f"Unknown agent tool: {name}")
