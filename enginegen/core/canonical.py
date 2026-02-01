from __future__ import annotations

import json
from typing import Any

ROUND_DIGITS = 12


def canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        rounded = round(value, ROUND_DIGITS)
        if rounded == -0.0:
            rounded = 0.0
        return rounded
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(val) for key, val in value.items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    canonical = canonicalize(value)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
