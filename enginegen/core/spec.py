from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:  # Optional Rust core for canonicalization
    import enginegen_core as _rust_core  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _rust_core = None

from .diagnostics import Diagnostic, Diagnostics
from .canonical import canonical_json_bytes
from .units import normalize_quantity, normalize_quantity_any, parse_quantity

SPEC_VERSION = "1.0.0"


class EngineSpecModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    spec_version: str = Field(default=SPEC_VERSION)
    name: str | None = None
    description: str | None = None
    requirements: Dict[str, Any] | None = None
    constraints: Dict[str, Any] | None = None
    manufacturing: Dict[str, Any] | None = None
    analysis_budget: Dict[str, Any] | None = None
    metadata: Dict[str, Any] | None = None
    extensions: Dict[str, Any] | None = None

    # Legacy fields retained for backward compatibility.
    engine: Dict[str, Any] | None = None
    propellant: Dict[str, Any] | None = None
    performance: Dict[str, Any] | None = None
    geometry: Dict[str, Any] | None = None


def load_spec(path: Path) -> Dict[str, Any]:
    data = _load_data(path)
    if not isinstance(data, dict):
        raise ValueError("Spec must be a JSON object")
    return data


def validate_spec(spec: Dict[str, Any], schema_path: Path) -> Diagnostics:
    diagnostics = Diagnostics()
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    schema["$id"] = schema_path.resolve().as_uri()
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(spec), key=str):
        diagnostics.add(
            Diagnostic(
                code="E-SPEC-SCHEMA",
                message=error.message,
                location="/".join(str(x) for x in error.path),
            )
        )
    return diagnostics


def normalize_spec(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Diagnostics]:
    diagnostics = Diagnostics()
    normalized: Dict[str, Any] = json.loads(json.dumps(spec))

    normalized.setdefault("spec_version", SPEC_VERSION)
    spec_version = normalized.get("spec_version")
    if isinstance(spec_version, str) and _major(spec_version) != _major(SPEC_VERSION):
        diagnostics.add(
            Diagnostic(
                code="E-SPEC-VERSION",
                message=f"Spec version mismatch: expected {SPEC_VERSION}, got {spec_version}",
                location="spec_version",
            )
        )

    def _normalize(path: Tuple[str, ...], kind: str) -> None:
        cursor = normalized
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                return
            cursor = cursor[key]
        field = path[-1]
        if field not in cursor:
            return
        value, diags = normalize_quantity(cursor[field], kind, ".".join(path))
        diagnostics.extend(diags)
        cursor[field] = value

    # performance
    _normalize(("performance", "chamber_pressure"), "pressure")
    _normalize(("performance", "thrust"), "force")

    # geometry
    _normalize(("geometry", "chamber", "length"), "length")
    _normalize(("geometry", "chamber", "radius"), "length")
    _normalize(("geometry", "nozzle", "throat_radius"), "length")
    _normalize(("geometry", "nozzle", "exit_radius"), "length")
    _normalize(("geometry", "nozzle", "length"), "length")

    # manufacturing
    _normalize(("manufacturing", "min_wall_thickness"), "length")
    _normalize(("manufacturing", "min_feature"), "length")
    _normalize(("manufacturing", "max_overhang_angle"), "angle")

    def _normalize_generic(value: Any, path: Tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            if "value" in value and "unit" in value and isinstance(value.get("unit"), str):
                converted, diags = normalize_quantity_any(value, ".".join(path) or "spec")
                diagnostics.extend(diags)
                return converted
            return {key: _normalize_generic(val, path + (str(key),)) for key, val in value.items()}
        if isinstance(value, list):
            return [_normalize_generic(item, path + (str(idx),)) for idx, item in enumerate(value)]
        if isinstance(value, str):
            try:
                raw_value, unit = parse_quantity(value)
            except Exception:
                return value
            if unit == "":
                return value
            converted, diags = normalize_quantity_any(value, ".".join(path) or "spec")
            diagnostics.extend(diags)
            return converted
        return value

    normalized = _normalize_generic(normalized, ())

    try:
        model = EngineSpecModel.model_validate(normalized)
        normalized = model.model_dump(mode="python")
    except ValidationError as exc:
        diagnostics.add(
            Diagnostic(
                code="E-SPEC-PYDANTIC",
                message=str(exc),
                location="spec",
            )
        )

    return normalized, diagnostics


def canonical_json(spec: Dict[str, Any]) -> bytes:
    if _rust_core is not None:
        raw = json.dumps(spec, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return _rust_core.normalize_spec(raw)
    return canonical_json_bytes(spec)


def _major(version: str) -> str:
    return version.split(".")[0] if version else ""


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
