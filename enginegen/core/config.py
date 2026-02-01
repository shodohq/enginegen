from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_CASE = {
    "synthesizer": {"type": "baseline_rule"},
    "geometry": {"backend": {"type": "simple_stl"}, "export": ["STL"]},
    "pipeline": [],
    "analysis": [],
}


def load_case_config(path: Path) -> Dict[str, Any]:
    data = _load_data(path)
    if not isinstance(data, dict):
        raise ValueError("Case config must be a JSON object")
    return data


def normalize_case_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CASE))
    return _deep_merge(merged, config)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def collect_libs(config: Dict[str, Any]) -> List[str]:
    libs: List[str] = []

    def _collect(section: Dict[str, Any]) -> None:
        if not isinstance(section, dict):
            return
        libs.extend(section.get("libs", []) or [])

    _collect(config.get("synthesizer", {}))
    geometry = config.get("geometry", {})
    if isinstance(geometry, dict):
        _collect(geometry.get("backend", {}))
    for step in config.get("pipeline", []) or []:
        _collect(step)
    for step in config.get("analysis", []) or []:
        _collect(step)
    optimization = config.get("optimization", {})
    if isinstance(optimization, dict):
        _collect(optimization.get("driver", {}))
        _collect(optimization)

    return [lib for lib in libs if lib]


def _load_data(path: Path) -> Any:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if path.suffix == ".toml":
        try:
            import tomllib
        except ImportError:  # pragma: no cover - python <3.11
            import tomli as tomllib  # type: ignore

        with path.open("rb") as handle:
            return tomllib.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
