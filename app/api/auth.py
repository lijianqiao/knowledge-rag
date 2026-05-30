"""API-Key 鉴权：解析 X-API-Key → Principal(user, allowed_sources)。

API_KEYS 为空 = 开放模式（不校验，allowed_sources=None=全部），便于本机/可信网络。
配置后强制校验：缺失/非法 key → 401。
"""

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from app.config import API_KEYS


@dataclass(frozen=True)
class Principal:
    user: str
    allowed_sources: list | None  # None = 不限制


def require_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    if not API_KEYS:  # 开放模式
        return Principal(user="anonymous", allowed_sources=None)
    entry = API_KEYS.get(x_api_key or "")
    if entry is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效或缺失 API Key")
    return Principal(user=entry.get("user", "unknown"), allowed_sources=entry.get("allowed_sources"))
