from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2, cos, sin, pi, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from enginegen.core.ir import extract_checks, toposort_ir
from enginegen.core.plugin_api import ArtifactBundle, ArtifactRef, GeometryBackendPlugin, PluginMeta

SUPPORTED_OPS = {
    "cylinder",
    "cone",
    "translate",
    "union",
    "solid.boolean",
    "frame.create",
    "plane.create",
    "axis.create",
    "sketch.circle",
    "solid.extrude",
    "solid.loft",
    "annotate",
    "annotate.group",
    "annotate.port",
    "validate",
    "check.geometry",
    "check.manufacturing",
}


@dataclass(frozen=True)
class Primitive:
    kind: str
    params: Dict[str, float]
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Profile:
    kind: str
    params: Dict[str, float]


Value = Union[Primitive, Profile]


class SimpleStlBackend(GeometryBackendPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="simple_stl",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "geometry_backend",
                "io": {
                    "inputs": [
                        {"name": "geometry_ir", "category": "data", "data_kind": "geometry_ir"},
                    ],
                    "outputs": [
                        {"name": "cad.stl", "category": "artifact", "artifact_kind": "cad.stl"},
                        {"name": "cad.step", "category": "artifact", "artifact_kind": "cad.step"},
                        {"name": "validation.json", "category": "artifact", "artifact_kind": "validation.json"},
                    ],
                },
                "features": {
                    "export_formats": ["STL", "STEP"],
                    "ir_version": "1.0.0",
                    "ir_ops": sorted(SUPPORTED_OPS),
                },
            },
        )

    def compile(
        self, ir: Dict[str, Any], ctx, *, export_formats: Optional[List[str]] = None
    ) -> ArtifactBundle:
        primitives = _build_primitives(ir)
        facets: List[
            Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
        ] = []
        for prim in primitives:
            tx, ty, tz = prim.translate
            if prim.kind == "cylinder":
                radius = prim.params["radius"]
                length = prim.params["length"]
                facets.extend(
                    _cylinder_facets(
                        radius,
                        length,
                        origin=(tx, ty, tz),
                        segments=32,
                        cap_start=True,
                        cap_end=False,
                    )
                )
            elif prim.kind == "cone":
                facets.extend(
                    _cone_facets(
                        prim.params["radius_start"],
                        prim.params["radius_end"],
                        prim.params["length"],
                        origin=(tx, ty, tz),
                        segments=32,
                        cap_start=False,
                        cap_end=True,
                    )
                )
            else:
                raise RuntimeError(f"Unsupported primitive: {prim.kind}")

        artifacts = ArtifactBundle()
        requested = export_formats or ["STL", "STEP"]
        requested_norm = {str(fmt).upper() for fmt in requested}
        if "STL" in requested_norm:
            stl_path = ctx.run_dir / "artifacts" / "engine.stl"
            _write_ascii_stl(stl_path, facets)
            artifacts.add(
                ArtifactRef(
                    kind="cad.stl",
                    path=stl_path,
                    producer=self.meta().name,
                    metadata={"format": "STL"},
                )
            )

        if "STEP" in requested_norm:
            step_path = ctx.run_dir / "artifacts" / "engine.step"
            _write_step(step_path, facets)
            artifacts.add(
                ArtifactRef(
                    kind="cad.step",
                    path=step_path,
                    producer=self.meta().name,
                    metadata={"format": "STEP"},
                )
            )

        if not artifacts.items:
            raise RuntimeError("No export formats selected for geometry backend")

        return artifacts

    def validate(self, ir: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        checks = extract_checks(ir)
        if not checks:
            return artifacts

        primitives = _build_primitives(ir)
        min_radius = _min_radius(primitives)
        cone_angles = _cone_angles(primitives)

        stl_path = None
        for ref in artifacts.by_kind("cad.stl"):
            stl_path = ref.path
            break

        results: List[Dict[str, Any]] = []
        for check in checks:
            op = check.get("op")
            params_in = check.get("params", {}) or {}
            if op in {"validate", "check.manufacturing"}:
                min_wall = params_in.get("min_wall_thickness")
                if min_wall is not None:
                    status = "PASS" if float(min_wall) <= min_radius else "FAIL"
                    results.append(
                        _check_result(
                            "min_wall_thickness",
                            status,
                            value=float(min_wall),
                            limit=min_radius,
                            target="engine",
                            hint="Reduce min_wall_thickness or increase radii",
                        )
                    )
                min_feature = params_in.get("min_feature")
                if min_feature is None:
                    min_feature = params_in.get("min_feature_size")
                if min_feature is not None:
                    status = "PASS" if float(min_feature) <= min_radius else "FAIL"
                    results.append(
                        _check_result(
                            "min_feature",
                            status,
                            value=float(min_feature),
                            limit=min_radius,
                            target="engine",
                            hint="Reduce min_feature or increase radii",
                        )
                    )
                max_overhang = params_in.get("max_overhang_angle")
                if max_overhang is None:
                    max_overhang = params_in.get("overhang_limit")
                if max_overhang is not None and cone_angles:
                    max_angle = max(cone_angles)
                    status = "PASS" if max_angle <= float(max_overhang) else "FAIL"
                    results.append(
                        _check_result(
                            "max_overhang_angle",
                            status,
                            value=max_angle,
                            limit=float(max_overhang),
                            target="nozzle",
                            hint="Reduce expansion ratio or increase nozzle length",
                        )
                    )
            if op in {"check.geometry", "validate"} and stl_path is not None:
                watertight = _check_watertight(stl_path)
                results.append(
                    _check_result(
                        "watertight",
                        "PASS" if watertight else "FAIL",
                        value=watertight,
                        limit=True,
                        target="stl",
                        hint="Inspect STL facets and ensure closed surface",
                    )
                )

        results_path = ctx.run_dir / "artifacts" / "validation.json"
        payload = {"checks": results}
        results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.add(
            ArtifactRef(
                kind="validation.json",
                path=results_path,
                producer=self.meta().name,
                metadata={"count": len(results)},
            )
        )
        return artifacts


def _node_args(node: Dict[str, Any]) -> Dict[str, Any]:
    if "args" in node:
        return node.get("args") or {}
    return node.get("params", {}) or {}


def _build_primitives(ir: Dict[str, Any]) -> List[Primitive]:
    nodes = ir.get("ops") or ir.get("nodes") or []
    if not nodes:
        raise RuntimeError("IR contains no ops")

    node_map = {node.get("id"): node for node in nodes}
    order = toposort_ir(ir)
    values: Dict[str, List[Value]] = {}

    for node_id in order:
        node = node_map.get(node_id)
        if node is None:
            continue
        op = node.get("op")
        if op not in SUPPORTED_OPS:
            raise RuntimeError(f"Unsupported IR op for simple backend: {op}")
        if op == "cylinder":
            values[node_id] = [_primitive_cylinder(node)]
        elif op == "cone":
            values[node_id] = [_primitive_cone(node)]
        elif op == "sketch.circle":
            values[node_id] = [_profile_circle(node)]
        elif op == "solid.extrude":
            base = _gather_inputs(values, node)
            profile = _extract_profile(base, op="solid.extrude")
            values[node_id] = [_extrude_profile(profile, node)]
        elif op == "solid.loft":
            base = _gather_inputs(values, node)
            profiles = _extract_profiles(base, op="solid.loft")
            values[node_id] = [_loft_profiles(profiles, node)]
        elif op == "translate":
            base = _gather_inputs(values, node)
            dx, dy, dz = _translate_vector(node)
            values[node_id] = [_apply_translate_value(item, dx, dy, dz) for item in base]
        elif op in {"union", "solid.boolean"}:
            if op == "solid.boolean":
                mode = str((_node_args(node) or {}).get("op", "")).lower()
                if mode and mode != "union":
                    raise RuntimeError(f"simple backend only supports union, got {mode}")
            values[node_id] = _gather_primitives(values, node, op)
        else:
            values[node_id] = _gather_inputs(values, node)

    output_ids = _resolve_output_nodes(ir, nodes)
    primitives: List[Primitive] = []
    for node_id in output_ids:
        for item in values.get(node_id, []):
            if isinstance(item, Primitive):
                primitives.append(item)

    if not primitives:
        raise RuntimeError("IR did not produce any geometry primitives")
    return primitives


def _resolve_output_nodes(ir: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[str]:
    outputs = ir.get("outputs")
    if isinstance(outputs, dict):
        output_ids = []
        if outputs.get("main"):
            output_ids.append(outputs.get("main"))
        for key, value in outputs.items():
            if key == "main":
                continue
            if value:
                output_ids.append(value)
        output_ids = [node_id for node_id in output_ids if node_id]
        if output_ids:
            return output_ids
    referenced = set()
    for node in nodes:
        for ref in node.get("inputs", []) or []:
            referenced.add(ref)
    outputs = [node.get("id") for node in nodes if node.get("id") not in referenced]
    outputs = [node_id for node_id in outputs if node_id]
    if outputs:
        return outputs
    if nodes:
        last = nodes[-1].get("id")
        return [last] if last else []
    return []


def _gather_inputs(values: Dict[str, List[Value]], node: Dict[str, Any]) -> List[Value]:
    items: List[Value] = []
    for ref in node.get("inputs", []) or []:
        if ref not in values:
            raise RuntimeError(f"IR references missing node {ref}")
        items.extend(values[ref])
    return items


def _gather_primitives(
    values: Dict[str, List[Value]], node: Dict[str, Any], op: str
) -> List[Primitive]:
    primitives: List[Primitive] = []
    for ref in node.get("inputs", []) or []:
        if ref not in values:
            raise RuntimeError(f"IR references missing node {ref}")
        for item in values[ref]:
            if isinstance(item, Primitive):
                primitives.append(item)
            else:
                raise RuntimeError(f"{op} expects solid inputs, got profile {item.kind}")
    return primitives


def _extract_profile(values: List[Value], *, op: str) -> Profile:
    profiles = [item for item in values if isinstance(item, Profile)]
    if len(profiles) != 1:
        raise RuntimeError(f"{op} expects exactly one profile input")
    return profiles[0]


def _extract_profiles(values: List[Value], *, op: str) -> List[Profile]:
    profiles = [item for item in values if isinstance(item, Profile)]
    if len(profiles) < 2:
        raise RuntimeError(f"{op} expects at least two profile inputs")
    return profiles[:2]


def _primitive_cylinder(node: Dict[str, Any]) -> Primitive:
    params = _node_args(node)
    if "radius" not in params or "length" not in params:
        raise RuntimeError("cylinder requires params.radius and params.length")
    return Primitive(
        kind="cylinder",
        params={"radius": float(params["radius"]), "length": float(params["length"])},
    )


def _primitive_cone(node: Dict[str, Any]) -> Primitive:
    params = _node_args(node)
    if "radius_start" not in params or "radius_end" not in params or "length" not in params:
        raise RuntimeError("cone requires params.radius_start, params.radius_end, params.length")
    return Primitive(
        kind="cone",
        params={
            "radius_start": float(params["radius_start"]),
            "radius_end": float(params["radius_end"]),
            "length": float(params["length"]),
        },
    )


def _profile_circle(node: Dict[str, Any]) -> Profile:
    params = _node_args(node)
    if "radius" not in params:
        raise RuntimeError("sketch.circle requires params.radius")
    return Profile(kind="circle", params={"radius": float(params["radius"])})


def _offset_vector(params: Dict[str, Any]) -> Tuple[float, float, float]:
    raw = params.get("offset")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return (0.0, 0.0, 0.0)


def _extrude_profile(profile: Profile, node: Dict[str, Any]) -> Primitive:
    params = _node_args(node)
    if "length" not in params:
        raise RuntimeError("solid.extrude requires params.length")
    length = float(params["length"])
    if profile.kind != "circle":
        raise RuntimeError(f"solid.extrude supports circle profiles only, got {profile.kind}")
    offset = _offset_vector(params)
    return Primitive(
        kind="cylinder",
        params={"radius": profile.params["radius"], "length": length},
        translate=offset,
    )


def _loft_profiles(profiles: List[Profile], node: Dict[str, Any]) -> Primitive:
    params = _node_args(node)
    if "length" not in params:
        raise RuntimeError("solid.loft requires params.length")
    length = float(params["length"])
    if any(profile.kind != "circle" for profile in profiles[:2]):
        raise RuntimeError("solid.loft supports circle profiles only")
    offset = _offset_vector(params)
    return Primitive(
        kind="cone",
        params={
            "radius_start": profiles[0].params["radius"],
            "radius_end": profiles[1].params["radius"],
            "length": length,
        },
        translate=offset,
    )


def _translate_vector(node: Dict[str, Any]) -> Tuple[float, float, float]:
    params = _node_args(node)
    return (
        float(params.get("x", 0.0)),
        float(params.get("y", 0.0)),
        float(params.get("z", 0.0)),
    )

def _apply_translate_value(item: Value, dx: float, dy: float, dz: float) -> Value:
    if isinstance(item, Profile):
        raise RuntimeError("translate cannot be applied to sketch profiles")
    tx, ty, tz = item.translate
    return Primitive(
        kind=item.kind,
        params=item.params,
        translate=(tx + dx, ty + dy, tz + dz),
    )


def _min_radius(primitives: List[Primitive]) -> float:
    radii: List[float] = []
    for prim in primitives:
        if prim.kind == "cylinder":
            radii.append(prim.params["radius"])
        elif prim.kind == "cone":
            radii.append(min(prim.params["radius_start"], prim.params["radius_end"]))
    return min(radii) if radii else 0.0


def _cone_angles(primitives: List[Primitive]) -> List[float]:
    angles: List[float] = []
    for prim in primitives:
        if prim.kind != "cone":
            continue
        length = prim.params["length"]
        if length <= 0:
            continue
        angle = atan2(
            prim.params["radius_end"] - prim.params["radius_start"],
            length,
        )
        angles.append(angle)
    return angles


def _cylinder_facets(
    radius: float,
    length: float,
    *,
    origin: Tuple[float, float, float],
    segments: int,
    cap_start: bool,
    cap_end: bool,
):
    facets = []
    ox, oy, oz = origin
    step = 2 * pi / segments
    for i in range(segments):
        theta0 = i * step
        theta1 = (i + 1) * step
        x0, y0 = radius * cos(theta0) + ox, radius * sin(theta0) + oy
        x1, y1 = radius * cos(theta1) + ox, radius * sin(theta1) + oy
        z0 = oz
        z1 = oz + length

        facets.append(((x0, y0, z0), (x1, y1, z0), (x1, y1, z1)))
        facets.append(((x0, y0, z0), (x1, y1, z1), (x0, y0, z1)))

        if cap_start:
            facets.append(((ox, oy, z0), (x1, y1, z0), (x0, y0, z0)))
        if cap_end:
            facets.append(((ox, oy, z1), (x0, y0, z1), (x1, y1, z1)))
    return facets


def _cone_facets(
    r0: float,
    r1: float,
    length: float,
    *,
    origin: Tuple[float, float, float],
    segments: int,
    cap_start: bool,
    cap_end: bool,
):
    facets = []
    ox, oy, oz = origin
    step = 2 * pi / segments
    for i in range(segments):
        theta0 = i * step
        theta1 = (i + 1) * step
        x00, y00 = r0 * cos(theta0) + ox, r0 * sin(theta0) + oy
        x01, y01 = r0 * cos(theta1) + ox, r0 * sin(theta1) + oy
        x10, y10 = r1 * cos(theta0) + ox, r1 * sin(theta0) + oy
        x11, y11 = r1 * cos(theta1) + ox, r1 * sin(theta1) + oy
        z0 = oz
        z1 = oz + length

        facets.append(((x00, y00, z0), (x01, y01, z0), (x11, y11, z1)))
        facets.append(((x00, y00, z0), (x11, y11, z1), (x10, y10, z1)))

        if cap_start:
            facets.append(((ox, oy, z0), (x01, y01, z0), (x00, y00, z0)))
        if cap_end:
            facets.append(((ox, oy, z1), (x10, y10, z1), (x11, y11, z1)))
    return facets


def _write_ascii_stl(path: Path, facets):
    lines = ["solid engine"]
    for v1, v2, v3 in facets:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        lines.append(f"      vertex {v1[0]} {v1[1]} {v1[2]}")
        lines.append(f"      vertex {v2[0]} {v2[1]} {v2[2]}")
        lines.append(f"      vertex {v3[0]} {v3[1]} {v3[2]}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid engine")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: float) -> str:
    rounded = round(value, 6)
    if rounded == -0.0:
        rounded = 0.0
    text = f"{rounded:.6f}"
    return text


def _write_step(path: Path, facets):
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('EngineGen STEP export'),'2;1');",
        "FILE_NAME('engine.step','',('enginegen'),('enginegen'),'enginegen','enginegen','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));",
        "ENDSEC;",
        "DATA;",
    ]

    entities: List[str] = []

    def add(entity: str) -> int:
        entities.append(entity)
        return len(entities)

    app_ctx = add("APPLICATION_CONTEXT('configuration controlled 3d design of mechanical parts')")
    app_proto = add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2003,#{app_ctx})"
    )
    prod_ctx = add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    product = add("PRODUCT('engine','engine','',(#%d))" % prod_ctx)
    formation = add(f"PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#{product},.MADE.)")
    design_ctx = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    prod_def = add(f"PRODUCT_DEFINITION('design','',#{formation},#{design_ctx})")
    prod_shape = add(f"PRODUCT_DEFINITION_SHAPE('','',#{prod_def})")

    len_unit = add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.))")
    ang_unit = add("(PLANE_ANGLE_UNIT() NAMED_UNIT(*) SI_UNIT($,.RADIAN.))")
    sol_unit = add("(SOLID_ANGLE_UNIT() NAMED_UNIT(*) SI_UNIT($,.STERADIAN.))")
    uncertainty = add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{len_unit},'distance_accuracy_value','')"
    )
    context = add(
        "GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f" GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))"
        f" GLOBAL_UNIT_ASSIGNED_CONTEXT((#{len_unit},#{ang_unit},#{sol_unit}))"
        " REPRESENTATION_CONTEXT('','')"
    )

    origin = add("CARTESIAN_POINT('',(0.0,0.0,0.0))")
    axis = add("DIRECTION('',(0.0,0.0,1.0))")
    ref_dir = add("DIRECTION('',(1.0,0.0,0.0))")
    axis2 = add(f"AXIS2_PLACEMENT_3D('',#{origin},#{axis},#{ref_dir})")

    face_ids: List[int] = []

    for v1, v2, v3 in facets:
        p1 = add(
            f"CARTESIAN_POINT('',({_fmt(v1[0])},{_fmt(v1[1])},{_fmt(v1[2])}))"
        )
        p2 = add(
            f"CARTESIAN_POINT('',({_fmt(v2[0])},{_fmt(v2[1])},{_fmt(v2[2])}))"
        )
        p3 = add(
            f"CARTESIAN_POINT('',({_fmt(v3[0])},{_fmt(v3[1])},{_fmt(v3[2])}))"
        )
        nx, ny, nz = _triangle_normal(v1, v2, v3)
        n_dir = add(f"DIRECTION('',({_fmt(nx)},{_fmt(ny)},{_fmt(nz)}))")
        rx, ry, rz = _reference_dir(nx, ny, nz)
        r_dir = add(f"DIRECTION('',({_fmt(rx)},{_fmt(ry)},{_fmt(rz)}))")
        plane_pos = add(f"AXIS2_PLACEMENT_3D('',#{p1},#{n_dir},#{r_dir})")
        plane = add(f"PLANE('',#{plane_pos})")
        loop = add(f"POLY_LOOP('',(#{p1},#{p2},#{p3}))")
        bound = add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        face = add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)")
        face_ids.append(face)

    shell = add(f"CLOSED_SHELL('',({','.join(f'#{fid}' for fid in face_ids)}))")
    solid = add(f"MANIFOLD_SOLID_BREP('',#{shell})")
    brep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{solid},#{axis2}),#{context})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{prod_shape},#{brep})")

    for idx, entity in enumerate(entities, start=1):
        lines.append(f"#{idx} = {entity};")

    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")

    path.write_text("\n".join(lines), encoding="utf-8")


def _triangle_normal(v1, v2, v3) -> Tuple[float, float, float]:
    ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
    bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    length = sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def _reference_dir(nx: float, ny: float, nz: float) -> Tuple[float, float, float]:
    if abs(nz) < 0.9:
        up = (0.0, 0.0, 1.0)
    else:
        up = (0.0, 1.0, 0.0)
    rx = ny * up[2] - nz * up[1]
    ry = nz * up[0] - nx * up[2]
    rz = nx * up[1] - ny * up[0]
    length = sqrt(rx * rx + ry * ry + rz * rz)
    if length == 0.0:
        return (1.0, 0.0, 0.0)
    return (rx / length, ry / length, rz / length)


def _check_watertight(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "facet normal" in text and text.strip().endswith("endsolid engine")


def _check_result(rule: str, status: str, *, value: Any, limit: Any, target: str, hint: str) -> Dict[str, Any]:
    return {
        "rule": rule,
        "status": status,
        "value": value,
        "limit": limit,
        "target": target,
        "hint": hint,
    }
