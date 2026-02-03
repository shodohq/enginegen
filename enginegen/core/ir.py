from __future__ import annotations

import json
from collections import defaultdict, deque
import re
from pathlib import Path
from typing import Any, Dict, List

import jsonschema

try:  # Optional Rust core for canonicalization
    import enginegen_core as _rust_core  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _rust_core = None

from .diagnostics import Diagnostic, Diagnostics
from .canonical import canonical_json_bytes

CORE_OPS = {
    # Legacy/simple ops used by built-in synth.
    "cylinder",
    "cone",
    "translate",
    "union",
    "annotate",
    "validate",
    # Reduced instruction set (v1).
    "frame.create",
    "plane.create",
    "axis.create",
    "sketch.polyline",
    "sketch.circle",
    "sketch.rect",
    "sketch.spline",
    "sketch.offset",
    "solid.extrude",
    "solid.revolve",
    "solid.sweep",
    "solid.loft",
    "solid.boolean",
    "solid.shell",
    "solid.fillet",
    "solid.chamfer",
    "annotate.group",
    "annotate.port",
    "check.geometry",
    "check.manufacturing",
}

CHECK_OPS = {"validate", "check.geometry", "check.manufacturing"}
ANNOTATE_OPS = {"annotate", "annotate.group", "annotate.port"}

_OP_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")

IR_VERSION = "1.0.0"


def validate_ir(ir: Dict[str, Any]) -> Diagnostics:
    diagnostics = Diagnostics()
    ir_version = ir.get("ir_version")
    if not ir_version:
        diagnostics.add(
            Diagnostic(
                code="E-IR-VERSION",
                message="Missing ir_version",
                location="ir_version",
            )
        )
    elif _major(str(ir_version)) != _major(IR_VERSION):
        diagnostics.add(
            Diagnostic(
                code="E-IR-VERSION",
                message=f"IR version mismatch: expected {IR_VERSION}, got {ir_version}",
                location="ir_version",
            )
        )
    ops = ir.get("ops")
    if ops is None:
        ops = ir.get("nodes")
    if ops is None:
        ops = []

    if not isinstance(ops, list):
        diagnostics.add(
            Diagnostic(
                code="E-IR-OPS",
                message="IR ops must be a list",
                location="ops",
            )
        )
        ops = []

    op_ids = [op.get("id") for op in ops if isinstance(op, dict)]
    if len(set(op_ids)) != len(op_ids):
        diagnostics.add(
            Diagnostic(
                code="E-IR-OP-DUP",
                message="Duplicate IR op id detected",
                location="ops",
            )
        )

    op_map = {op.get("id"): op for op in ops if isinstance(op, dict) and op.get("id")}
    for op in ops:
        if not isinstance(op, dict):
            continue
        op_id = op.get("id")
        if not op_id:
            diagnostics.add(
                Diagnostic(
                    code="E-IR-OP-ID",
                    message="IR op id missing",
                    location="ops",
                )
            )
            continue
        op_name = op.get("op")
        if not isinstance(op_name, str) or not _is_allowed_op(op_name):
            diagnostics.add(
                Diagnostic(
                    code="E-IR-OP",
                    message=f"Unsupported IR op: {op_name}",
                    location=f"ops.{op_id}",
                )
            )
        for ref in op.get("inputs", []) or []:
            if ref not in op_map:
                diagnostics.add(
                    Diagnostic(
                        code="E-IR-REF",
                        message=f"IR op {op_id} references unknown input {ref}",
                        location=f"ops.{op_id}",
                    )
                )

    outputs = ir.get("outputs")
    if not isinstance(outputs, dict):
        diagnostics.add(
            Diagnostic(
                code="E-IR-OUTPUTS",
                message="IR outputs missing or invalid",
                location="outputs",
            )
        )
    else:
        if "main" not in outputs and "solid" not in outputs:
            diagnostics.add(
                Diagnostic(
                    code="E-IR-OUTPUTS",
                    message="IR outputs.main or outputs.solid is required",
                    location="outputs",
                )
            )
        for name, ref in outputs.items():
            if ref not in op_map:
                diagnostics.add(
                    Diagnostic(
                        code="E-IR-OUTPUTS",
                        message=f"IR output {name} references unknown op {ref}",
                        location=f"outputs.{name}",
                    )
                )

    for check in ir.get("checks", []) or []:
        if not isinstance(check, dict):
            continue
        if "type" not in check:
            diagnostics.add(
                Diagnostic(
                    code="E-IR-CHECK",
                    message="IR check missing type",
                    location="checks",
                )
            )

    if _detect_cycle(op_map):
        diagnostics.add(
            Diagnostic(
                code="E-IR-CYCLE",
                message="Cycle detected in IR DAG",
                location="ops",
            )
        )

    return diagnostics


def validate_ir_schema(ir: Dict[str, Any], schema_path: Path) -> Diagnostics:
    diagnostics = Diagnostics()
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    schema["$id"] = schema_path.resolve().as_uri()
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(ir), key=str):
        diagnostics.add(
            Diagnostic(
                code="E-IR-SCHEMA",
                message=error.message,
                location="/".join(str(x) for x in error.path),
            )
        )
    return diagnostics


def extract_checks(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    ops = ir.get("ops")
    if ops is None:
        ops = ir.get("nodes") or []
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        op_name = op.get("op")
        if op_name in CHECK_OPS:
            checks.append(
                {
                    "op": op_name,
                    "params": _op_args(op),
                    "inputs": op.get("inputs", []) or [],
                    "id": op.get("id"),
                }
            )
    for check in ir.get("checks", []) or []:
        if not isinstance(check, dict):
            continue
        if "type" in check:
            checks.append(
                {
                    "op": check.get("type"),
                    "params": check.get("params", {}) or {},
                    "applies_to": check.get("applies_to"),
                    "message": check.get("message"),
                }
            )
        else:
            checks.append(check)
    return checks


def toposort_ir(ir: Dict[str, Any]) -> List[str]:
    ops = ir.get("ops")
    if ops is None:
        ops = ir.get("nodes") or []
    node_map = {node.get("id"): node for node in ops if node.get("id")}
    indegree = {node_id: 0 for node_id in node_map}
    adj = defaultdict(list)

    for node in ops:
        node_id = node.get("id")
        if not node_id:
            continue
        for ref in node.get("inputs", []) or []:
            adj[ref].append(node_id)
            indegree[node_id] += 1

    queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
    order: List[str] = []

    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor in adj.get(node_id, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(node_map):
        raise ValueError("IR DAG has cycles")

    return order


def canonical_json(ir: Dict[str, Any]) -> bytes:
    if _rust_core is not None:
        raw = json.dumps(ir, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return _rust_core.normalize_ir(raw)
    return canonical_json_bytes(ir)


def _detect_cycle(node_map: Dict[str, Any]) -> bool:
    visited = set()
    stack = set()

    def visit(node_id: str) -> bool:
        visited.add(node_id)
        stack.add(node_id)
        node = node_map[node_id]
        for ref in node.get("inputs", []) or []:
            if ref not in node_map:
                continue
            if ref not in visited:
                if visit(ref):
                    return True
            elif ref in stack:
                return True
        stack.remove(node_id)
        return False

    for node_id in node_map:
        if node_id not in visited:
            if visit(node_id):
                return True

    return False


def _major(version: str) -> str:
    return version.split(".")[0] if version else ""


def _is_allowed_op(op: str) -> bool:
    if op in CORE_OPS:
        return True
    if op.startswith("ext."):
        return True
    if _OP_PATTERN.match(op):
        return True
    return False


def _op_args(node: Dict[str, Any]) -> Dict[str, Any]:
    if "args" in node:
        return node.get("args") or {}
    return node.get("params", {}) or {}
