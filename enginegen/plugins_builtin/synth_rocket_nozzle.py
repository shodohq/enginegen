from __future__ import annotations

from math import pi, sqrt, tan
from typing import Any, Dict, Iterable, List, Tuple

from enginegen.core.ir import IR_VERSION
from enginegen.core.plugin_api import PluginMeta, SynthesizerPlugin


class RocketNozzleSynth(SynthesizerPlugin):
    """Generate a hollow bell nozzle using the Fidget implicit geometry dialect."""

    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="rocket_nozzle",
            api_version="1.0.0",
            plugin_version="0.2.0",
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
                "features": {
                    "ir_version": IR_VERSION,
                    "dialect": "enginegen.implicit.fidget.v1",
                    "profile": "rao_cubic",
                    "hollow": True,
                },
            },
        )

    def synthesize(
        self,
        spec: Dict[str, Any],
        graph: Dict[str, Any],
        ctx,
        feedback=None,
    ) -> Dict[str, Any]:
        metadata = spec.get("metadata") or {}
        extensions = spec.get("extensions") or {}
        geom = spec.get("geometry") or metadata.get("geometry") or extensions.get("geometry") or {}
        chamber = geom.get("chamber", {}) or {}
        nozzle = geom.get("nozzle", {}) or {}

        chamber_radius = _as_float(chamber.get("radius"), 0.12)
        chamber_length = _as_float(chamber.get("length"), 0.25)

        throat_radius = _as_float(nozzle.get("throat_radius"), chamber_radius * 0.35)
        exit_radius = _as_float(nozzle.get("exit_radius"))
        expansion_ratio = _as_float(nozzle.get("expansion_ratio"))
        if exit_radius is None and expansion_ratio:
            exit_radius = throat_radius * sqrt(expansion_ratio)
        if exit_radius is None:
            exit_radius = throat_radius * 4.5

        chamber_radius = _ensure_min(chamber_radius, throat_radius * 1.05)
        exit_radius = _ensure_min(exit_radius, throat_radius * 1.05)

        converging_length = _as_float(nozzle.get("converging_length"))
        throat_length = _as_float(nozzle.get("throat_length"))
        bell_length = _as_float(nozzle.get("bell_length"))
        total_nozzle_length = _as_float(nozzle.get("length"))

        if total_nozzle_length is not None:
            if converging_length is None:
                converging_length = total_nozzle_length * 0.25
            if throat_length is None:
                throat_length = total_nozzle_length * 0.08
            if bell_length is None:
                bell_length = total_nozzle_length - converging_length - throat_length

        converging_length = _ensure_positive(
            converging_length, (chamber_radius - throat_radius) * 1.25
        )
        throat_length = _ensure_positive(throat_length, throat_radius * 0.6)
        bell_length = _ensure_positive(bell_length, (exit_radius - throat_radius) * 3.2)

        wall_thickness = _wall_thickness(nozzle, spec)
        inner_extension = max(wall_thickness * 2.0, 0.005)

        bell_samples = _as_int(nozzle.get("bell_samples"), 10)
        bell_samples = max(4, bell_samples)

        throat_angle = _angle_to_rad(
            nozzle.get("bell_throat_angle_deg"), default_deg=30.0
        )
        exit_angle = _angle_to_rad(nozzle.get("bell_exit_angle_deg"), default_deg=15.0)

        z_chamber_end = chamber_length
        z_converge_end = z_chamber_end + converging_length
        z_throat_end = z_converge_end + throat_length
        total_length = z_throat_end + bell_length

        bell_profile = _bell_profile(
            throat_radius, exit_radius, bell_length, throat_angle, exit_angle, bell_samples
        )
        bell_z = [z_throat_end + x for x, _ in bell_profile]
        bell_r = [r for _, r in bell_profile]

        inner_ops: List[Dict[str, Any]] = []
        outer_ops: List[Dict[str, Any]] = []

        inner_segments: List[str] = []
        outer_segments: List[str] = []

        # Shared profiles
        inner_chamber_profile = _circle_op(inner_ops, "inner_chamber_profile", chamber_radius)
        inner_throat_profile = _circle_op(inner_ops, "inner_throat_profile", throat_radius)
        inner_exit_profile = _circle_op(inner_ops, "inner_exit_profile", exit_radius)

        outer_chamber_profile = _circle_op(
            outer_ops, "outer_chamber_profile", chamber_radius + wall_thickness
        )
        outer_throat_profile = _circle_op(
            outer_ops, "outer_throat_profile", throat_radius + wall_thickness
        )
        outer_exit_profile = _circle_op(
            outer_ops, "outer_exit_profile", exit_radius + wall_thickness
        )

        # Chamber + converging + throat (inner)
        inner_segments.append(
            _extrude_op(
                inner_ops,
                "inner_chamber",
                inner_chamber_profile,
                0.0,
                z_chamber_end,
            )
        )
        inner_segments.append(
            _loft_op(
                inner_ops,
                "inner_converging",
                inner_chamber_profile,
                inner_throat_profile,
                z_chamber_end,
                z_converge_end,
            )
        )
        inner_segments.append(
            _extrude_op(
                inner_ops,
                "inner_throat",
                inner_throat_profile,
                z_converge_end,
                z_throat_end,
            )
        )

        # Chamber + converging + throat (outer)
        outer_segments.append(
            _extrude_op(
                outer_ops,
                "outer_chamber",
                outer_chamber_profile,
                0.0,
                z_chamber_end,
            )
        )
        outer_segments.append(
            _loft_op(
                outer_ops,
                "outer_converging",
                outer_chamber_profile,
                outer_throat_profile,
                z_chamber_end,
                z_converge_end,
            )
        )
        outer_segments.append(
            _extrude_op(
                outer_ops,
                "outer_throat",
                outer_throat_profile,
                z_converge_end,
                z_throat_end,
            )
        )

        # Bell segments
        inner_bell_profiles = _sample_profiles(inner_ops, "inner_bell", bell_r)
        outer_bell_profiles = _sample_profiles(
            outer_ops, "outer_bell", [r + wall_thickness for r in bell_r]
        )
        for idx in range(len(bell_z) - 1):
            inner_segments.append(
                _loft_op(
                    inner_ops,
                    f"inner_bell_{idx}",
                    inner_bell_profiles[idx],
                    inner_bell_profiles[idx + 1],
                    bell_z[idx],
                    bell_z[idx + 1],
                )
            )
            outer_segments.append(
                _loft_op(
                    outer_ops,
                    f"outer_bell_{idx}",
                    outer_bell_profiles[idx],
                    outer_bell_profiles[idx + 1],
                    bell_z[idx],
                    bell_z[idx + 1],
                )
            )

        # Inner extensions to open ends
        inner_segments.append(
            _extrude_op(
                inner_ops,
                "inner_inlet_ext",
                inner_chamber_profile,
                -inner_extension,
                0.0,
            )
        )
        inner_segments.append(
            _extrude_op(
                inner_ops,
                "inner_exit_ext",
                inner_exit_profile,
                total_length,
                total_length + inner_extension,
            )
        )

        inner_union = _union_op(inner_ops, "inner_union", inner_segments)
        outer_union = _union_op(outer_ops, "outer_union", outer_segments)

        ops = [
            *inner_ops,
            *outer_ops,
            {
                "id": "nozzle_shell",
                "op": "fidget.csg.difference",
                "inputs": [outer_union, inner_union],
            },
        ]

        outer_radius = max(chamber_radius, exit_radius) + wall_thickness
        domain_bbox = _domain_bbox(outer_radius, total_length, inner_extension)

        annotations = _graph_annotations(graph, output_id="nozzle_shell")
        checks = _manufacturing_checks(spec)
        annotations["checks"].extend(checks)
        annotations["checks"].append({"type": "check.geometry", "params": {"watertight": True}})

        ir: Dict[str, Any] = {
            "ir_version": IR_VERSION,
            "dialect": "enginegen.implicit.fidget.v1",
            "units": {"length": "m"},
            "ops": ops,
            "outputs": {"main": "nozzle_shell"},
            "metadata": {
                "domain_bbox": domain_bbox,
                "segment_lengths": {
                    "chamber": chamber_length,
                    "converging": converging_length,
                    "throat": throat_length,
                    "bell": bell_length,
                },
                "wall_thickness": wall_thickness,
                "bell_samples": bell_samples,
            },
        }
        if annotations.get("annotations"):
            ir["annotations"] = annotations["annotations"]
        if annotations.get("checks"):
            ir["checks"] = annotations["checks"]
        return ir


def _circle_op(ops: List[Dict[str, Any]], op_id: str, radius: float) -> str:
    ops.append(
        {
            "id": op_id,
            "op": "fidget.prim.circle",
            "args": {"center": [0.0, 0.0], "radius": float(radius)},
        }
    )
    return op_id


def _extrude_op(
    ops: List[Dict[str, Any]], op_id: str, profile_id: str, lower: float, upper: float
) -> str:
    ops.append(
        {
            "id": op_id,
            "op": "fidget.gen.extrude_z",
            "inputs": [profile_id],
            "args": {"lower": float(lower), "upper": float(upper)},
        }
    )
    return op_id


def _loft_op(
    ops: List[Dict[str, Any]],
    op_id: str,
    profile_a: str,
    profile_b: str,
    lower: float,
    upper: float,
) -> str:
    ops.append(
        {
            "id": op_id,
            "op": "fidget.gen.loft_z",
            "inputs": [profile_a, profile_b],
            "args": {"lower": float(lower), "upper": float(upper)},
        }
    )
    return op_id


def _union_op(ops: List[Dict[str, Any]], op_id: str, inputs: Iterable[str]) -> str:
    op_inputs = [item for item in inputs if item]
    if not op_inputs:
        raise RuntimeError("union requires at least one input")
    ops.append({"id": op_id, "op": "fidget.csg.union", "inputs": op_inputs})
    return op_id


def _sample_profiles(
    ops: List[Dict[str, Any]], prefix: str, radii: List[float]
) -> List[str]:
    profiles: List[str] = []
    for idx, radius in enumerate(radii):
        profiles.append(_circle_op(ops, f"{prefix}_profile_{idx}", radius))
    return profiles


def _bell_profile(
    throat_radius: float,
    exit_radius: float,
    length: float,
    throat_angle: float,
    exit_angle: float,
    samples: int,
) -> List[Tuple[float, float]]:
    if length <= 1e-6:
        return [(0.0, throat_radius), (length, exit_radius)]

    c = tan(throat_angle)
    d = throat_radius
    target_exit = exit_radius
    tan_exit = tan(exit_angle)

    a, b = _solve_cubic_coeffs(length, d, c, target_exit, tan_exit)

    profile: List[Tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        x = t * length
        r = a * x**3 + b * x**2 + c * x + d
        r = _clamp(r, throat_radius, exit_radius)
        profile.append((x, r))
    profile[0] = (0.0, throat_radius)
    profile[-1] = (length, exit_radius)
    return profile


def _solve_cubic_coeffs(
    length: float,
    r0: float,
    slope0: float,
    r1: float,
    slope1: float,
) -> Tuple[float, float]:
    l2 = length * length
    l3 = l2 * length
    rhs1 = r1 - slope0 * length - r0
    rhs2 = slope1 - slope0
    denom = l3 * (2.0 * length) - (3.0 * l2) * (l2)
    if abs(denom) <= 1e-12:
        return 0.0, (r1 - r0 - slope0 * length) / max(l2, 1e-6)
    a = (rhs1 * (2.0 * length) - rhs2 * l2) / denom
    b = (rhs2 * l3 - rhs1 * (3.0 * l2)) / denom
    return a, b


def _domain_bbox(radius: float, length: float, extension: float) -> List[List[float]]:
    margin = max(radius * 0.15, 0.01)
    z_min = -extension - margin
    z_max = length + extension + margin
    r = radius + margin
    return [[-r, -r, z_min], [r, r, z_max]]


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _angle_to_rad(value: Any, *, default_deg: float) -> float:
    raw = _as_float(value)
    if raw is None or raw <= 0:
        raw = float(default_deg)
    # If the value looks like degrees, convert to radians.
    if raw > (pi + 0.1):
        return raw * pi / 180.0
    return raw


def _ensure_positive(value: float | None, fallback: float) -> float:
    if value is None or value <= 0.0:
        return max(fallback, 1e-6)
    return value


def _ensure_min(value: float, minimum: float) -> float:
    if value <= minimum:
        return minimum
    return value


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _wall_thickness(nozzle: Dict[str, Any], spec: Dict[str, Any]) -> float:
    thickness = _as_float(nozzle.get("wall_thickness"))
    if thickness is not None and thickness > 0:
        return thickness
    rules = (spec.get("manufacturing") or {}).get("rules", {}) or {}
    thickness = _as_float(rules.get("min_wall_thickness"))
    if thickness is not None and thickness > 0:
        return max(thickness, 0.002)
    return 0.005


def _manufacturing_checks(spec: Dict[str, Any]) -> list[Dict[str, Any]]:
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
        return [{"type": "check.manufacturing", "params": mfg_params}]
    return []


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
