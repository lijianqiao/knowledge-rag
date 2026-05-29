"""源文件内容 hash 清单：增量导入用。"""

import hashlib
import json
from pathlib import Path


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_manifest(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def diff_manifest(old: dict[str, str], new: dict[str, str]) -> tuple[list[str], list[str]]:
    """返回 (需重导的=新增+变更, 需删除的=消失)。"""
    changed = [k for k, v in new.items() if old.get(k) != v]
    removed = [k for k in old if k not in new]
    return changed, removed
