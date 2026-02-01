from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .artifacts import hash_bytes


@dataclass(frozen=True)
class CacheKey:
    spec_hash: str
    graph_hash: str
    ir_hash: str
    plugin_lock_hash: str
    config_hash: str
    feedback_hash: str = ""

    def to_string(self) -> str:
        raw = "|".join(
            [
                self.spec_hash,
                self.graph_hash,
                self.ir_hash,
                self.plugin_lock_hash,
                self.config_hash,
                self.feedback_hash,
            ]
        ).encode("utf-8")
        return hash_bytes(raw)


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> Optional[Path]:
        path = self.root / key.to_string()
        return path if path.exists() else None

    def put(self, key: CacheKey, data: Dict[str, Any]) -> Path:
        path = self.root / key.to_string()
        path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, sort_keys=True, indent=2)
        (path / "cache.json").write_text(payload, encoding="utf-8")
        return path


def hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hash_bytes(payload)
