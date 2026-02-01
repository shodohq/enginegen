from __future__ import annotations

from typing import Any, Dict

from enginegen.core.ir import IR_VERSION
from enginegen.core.plugin_api import PluginMeta, SynthesizerPlugin


class BaselineRuleSynth(SynthesizerPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="baseline_rule",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "synthesizer",
                "io": {
                    "inputs": [
                        {"name": "spec", "category": "data", "data_kind": "spec"},
                        {"name": "graph", "category": "data", "data_kind": "graph"},
                    ],
                    "outputs": [
                        {"name": "geometry_ir", "category": "data", "data_kind": "geometry_ir"},
                    ],
                },
                "features": {"ir_version": IR_VERSION},
            },
        )

    def synthesize(
        self,
        spec: Dict[str, Any],
        graph: Dict[str, Any],
        ctx,
        feedback=None,
    ) -> Dict[str, Any]:
        graph_nodes = sorted(
            [node.get("id") for node in graph.get("nodes", []) if node.get("id")]
        )
        graph_edges = len(graph.get("edges", []) or [])

        metadata = spec.get("metadata") or {}
        extensions = spec.get("extensions") or {}
        geom = spec.get("geometry") or metadata.get("geometry") or extensions.get("geometry") or {}
        chamber = geom.get("chamber", {}) or {}
        nozzle = geom.get("nozzle", {}) or {}

        chamber_radius = _as_float(chamber.get("radius"), 0.1)
        chamber_length = _as_float(chamber.get("length"), 0.3)
        throat_radius = _as_float(nozzle.get("throat_radius"), chamber_radius * 0.4)
        exit_radius = _as_float(nozzle.get("exit_radius"), throat_radius * 3.0)
        nozzle_length = _as_float(nozzle.get("length"), (exit_radius - throat_radius) * 3.0)

        ops = [
            {
                "id": "chamber_profile",
                "op": "sketch.circle",
                "inputs": [],
                "args": {"radius": chamber_radius},
            },
            {
                "id": "chamber_solid",
                "op": "solid.extrude",
                "inputs": ["chamber_profile"],
                "args": {"length": chamber_length},
            },
            {
                "id": "throat_profile",
                "op": "sketch.circle",
                "inputs": [],
                "args": {"radius": throat_radius},
            },
            {
                "id": "exit_profile",
                "op": "sketch.circle",
                "inputs": [],
                "args": {"radius": exit_radius},
            },
            {
                "id": "nozzle_solid",
                "op": "solid.loft",
                "inputs": ["throat_profile", "exit_profile"],
                "args": {"length": nozzle_length, "offset": [0.0, 0.0, chamber_length]},
            },
            {
                "id": "engine",
                "op": "solid.boolean",
                "inputs": ["chamber_solid", "nozzle_solid"],
                "args": {"op": "union"},
            },
        ]

        annotations = _graph_annotations(graph, output_id="engine")

        rules = (spec.get("manufacturing") or {}).get("rules", {}) or {}
        mfg_params: Dict[str, Any] = {}
        min_wall = _as_float(rules.get("min_wall_thickness"))
        if min_wall is not None:
            mfg_params["min_wall_thickness"] = min_wall
        min_feature = _as_float(rules.get("min_feature_size"))
        if min_feature is not None:
            mfg_params["min_feature"] = min_feature
        max_overhang = _as_float(rules.get("overhang_limit"))
        if max_overhang is not None:
            mfg_params["max_overhang_angle"] = max_overhang
        if mfg_params:
            annotations["checks"].append({"type": "check.manufacturing", "params": mfg_params})
        annotations["checks"].append({"type": "check.geometry", "params": {"watertight": True}})

        ir: Dict[str, Any] = {
            "ir_version": IR_VERSION,
            "ops": ops,
            "outputs": {"main": "engine"},
            "metadata": {"graph_nodes": graph_nodes, "graph_edges": graph_edges},
        }
        if annotations.get("annotations"):
            ir["annotations"] = annotations["annotations"]
        if annotations.get("checks"):
            ir["checks"] = annotations["checks"]
        return ir


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _graph_annotations(graph: Dict[str, Any], *, output_id: str) -> Dict[str, Any]:
    annotations: Dict[str, Any] = {}
    checks: list[Dict[str, Any]] = []
    ports: list[Dict[str, Any]] = []
    for node in graph.get("nodes", []) or []:
        node_id = node.get("id")
        for port in node.get("ports", []) or []:
            port_id = port.get("id")
            if not node_id or not port_id:
                continue
            name = port.get("name") or f"{node_id}.{port_id}"
            ports.append(
                {
                    "name": name,
                    "kind": port.get("kind", "fluid"),
                    "direction": port.get("direction", "bidir"),
                    "group": "engine",
                    "schema": {"node": node_id, "port": port_id},
                }
            )
    if not ports:
        ports = [
            {"name": "inlet", "kind": "fluid", "direction": "in", "group": "engine"},
            {"name": "outlet", "kind": "fluid", "direction": "out", "group": "engine"},
        ]
    annotations = {
        "groups": [{"name": "engine", "dim": "volume", "refs": [{"op": output_id}]}],
        "ports": ports,
    }
    return {"annotations": annotations, "checks": checks}
