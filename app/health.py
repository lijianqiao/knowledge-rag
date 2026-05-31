"""Deep operational health checks for API and CLI."""

from app.graph_store import graph_store_exists
from app.index import get_chroma_collection
from app.models import get_embed_model, get_llm


def _check_chroma() -> dict:
    try:
        return {"ok": True, "count": get_chroma_collection().count()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_graph() -> dict:
    exists = graph_store_exists()
    return {"ok": exists, "exists": exists}


def _check_models() -> dict:
    checks = {}
    try:
        get_embed_model()
        checks["embedding"] = True
    except Exception as exc:
        checks["embedding"] = str(exc)

    try:
        get_llm()
        checks["chat"] = True
    except Exception as exc:
        checks["chat"] = str(exc)

    return {
        "ok": checks.get("embedding") is True and checks.get("chat") is True,
        "checks": checks,
    }


def deep_health() -> dict:
    components = {
        "chroma": _check_chroma(),
        "graph": _check_graph(),
        "models": _check_models(),
    }
    return {
        "ok": all(component.get("ok") for component in components.values()),
        "components": components,
    }
