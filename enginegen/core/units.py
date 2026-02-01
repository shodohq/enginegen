from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .diagnostics import Diagnostic


@dataclass(frozen=True)
class Quantity:
    value_si: float
    unit: str


_UNIT_FACTORS: Dict[str, Tuple[str, float]] = {
    # length
    "m": ("length", 1.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    # pressure
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1_000.0),
    "mpa": ("pressure", 1_000_000.0),
    "bar": ("pressure", 100_000.0),
    # force
    "n": ("force", 1.0),
    "kn": ("force", 1_000.0),
    # mass flow
    "kg/s": ("mass_flow", 1.0),
    # temperature
    "k": ("temperature", 1.0),
    "c": ("temperature", 1.0),
    # angle
    "deg": ("angle", 1.0),
    "rad": ("angle", 1.0),
}


def _parse_string(value: str) -> Tuple[float, str]:
    match = re.match(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*([A-Za-z/]+)\s*$", value)
    if not match:
        raise ValueError(f"Invalid quantity string: {value}")
    return float(match.group(1)), match.group(2)


def parse_quantity(value: Any) -> Tuple[float, str]:
    if isinstance(value, (int, float)):
        return float(value), ""
    if isinstance(value, str):
        return _parse_string(value)
    if isinstance(value, dict):
        return float(value.get("value")), str(value.get("unit", ""))
    raise ValueError(f"Unsupported quantity type: {type(value)}")


def normalize_quantity(value: Any, expected_kind: str, field: str) -> Tuple[float, list[Diagnostic]]:
    diags: list[Diagnostic] = []
    try:
        raw_value, unit = parse_quantity(value)
    except Exception as exc:
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-PARSE",
                message=f"Failed to parse unit for {field}: {exc}",
                location=field,
            )
        )
        return 0.0, diags

    if unit == "":
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-MISSING",
                message=f"Unit missing for {field}",
                location=field,
            )
        )
        return 0.0, diags

    key = unit.strip().lower()
    if key not in _UNIT_FACTORS:
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-UNKNOWN",
                message=f"Unknown unit '{unit}' for {field}",
                location=field,
            )
        )
        return 0.0, diags

    kind, factor = _UNIT_FACTORS[key]
    if kind != expected_kind:
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-KIND",
                message=f"Unit kind mismatch for {field}: expected {expected_kind}, got {kind}",
                location=field,
            )
        )
        return 0.0, diags

    if kind == "temperature" and key == "c":
        return raw_value + 273.15, diags

    if kind == "angle" and key == "deg":
        return raw_value * 3.141592653589793 / 180.0, diags

    return raw_value * factor, diags


def normalize_quantity_any(value: Any, field: str) -> Tuple[float, list[Diagnostic]]:
    diags: list[Diagnostic] = []
    try:
        raw_value, unit = parse_quantity(value)
    except Exception as exc:
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-PARSE",
                message=f"Failed to parse unit for {field}: {exc}",
                location=field,
            )
        )
        return 0.0, diags

    if unit == "":
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-MISSING",
                message=f"Unit missing for {field}",
                location=field,
            )
        )
        return 0.0, diags

    key = unit.strip().lower()
    if key not in _UNIT_FACTORS:
        diags.append(
            Diagnostic(
                code="E-SPEC-UNIT-UNKNOWN",
                message=f"Unknown unit '{unit}' for {field}",
                location=field,
            )
        )
        return 0.0, diags

    kind, factor = _UNIT_FACTORS[key]
    if kind == "temperature" and key == "c":
        return raw_value + 273.15, diags
    if kind == "angle" and key == "deg":
        return raw_value * 3.141592653589793 / 180.0, diags

    return raw_value * factor, diags
