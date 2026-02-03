from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple

from enginegen.core.ir import extract_checks
from enginegen.core.diagnostics import Diagnostic, EngineGenError
from enginegen.core.plugin_api import ArtifactBundle, ArtifactRef, GeometryBackendPlugin, PluginMeta

SUPPORTED_OPS = sorted(
    {
        # Prefixed dialect ops
        "fidget.prim.sphere",
        "fidget.prim.box",
        "fidget.prim.circle",
        "fidget.prim.circle2d",
        "fidget.prim.rectangle",
        "fidget.prim.rect2d",
        "fidget.prim.rect",
        "fidget.csg.union",
        "fidget.csg.intersection",
        "fidget.csg.difference",
        "fidget.csg.inverse",
        "fidget.csg.blend",
        "fidget.csg.blend_quadratic",
        "fidget.ops.blend",
        "fidget.xform.move",
        "fidget.xform.scale",
        "fidget.xform.scale_uniform",
        "fidget.xform.rotate_x",
        "fidget.xform.rotate_y",
        "fidget.xform.rotate_z",
        "fidget.xform.reflect_x",
        "fidget.xform.reflect_y",
        "fidget.xform.reflect_z",
        "fidget.gen.extrude_z",
        "fidget.gen.loft_z",
        "fidget.gen.revolve_y",
        # Unprefixed / compatibility ops
        "prim.sphere",
        "prim.box",
        "prim.circle",
        "prim.circle2d",
        "prim.rectangle",
        "prim.rect2d",
        "prim.rect",
        "csg.union",
        "csg.intersection",
        "csg.difference",
        "csg.inverse",
        "csg.blend",
        "csg.blend_quadratic",
        "ops.blend",
        "xform.move",
        "xform.scale",
        "xform.scale_uniform",
        "xform.rotate_x",
        "xform.rotate_y",
        "xform.rotate_z",
        "xform.reflect_x",
        "xform.reflect_y",
        "xform.reflect_z",
        "solid.extrude_z",
        "solid.loft_z",
        "solid.revolve_y",
        "gen.extrude_z",
        "gen.loft_z",
        "gen.revolve_y",
    }
)

DEFAULT_MESH = {
    "depth": 6,
    "max_tris": 2_000_000,
    "parallel": True,
    "deterministic_order": True,
}

DEFAULT_CACHE = {
    "bytecode": False,
}

DEFAULT_CONFIG = {
    "evaluation_engine": "auto",
    "mesh": DEFAULT_MESH,
    "exports": {"stl": True, "obj": False, "ply": False},
    "cache": DEFAULT_CACHE,
}


class FidgetGeometryBackend(GeometryBackendPlugin):
    def __init__(self) -> None:
        self._last_config: Optional[Dict[str, Any]] = None

    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="builtin.geometry.fidget",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "geometry_backend",
                "io": {
                    "inputs": [
                        {
                            "name": "geometry_ir",
                            "category": "data",
                            "data_kind": "geometry_ir",
                        },
                        {
                            "name": "config",
                            "category": "data",
                            "data_kind": "config",
                            "optional": True,
                        },
                    ],
                    "outputs": [
                        {
                            "name": "mesh_stl",
                            "category": "artifact",
                            "artifact_kind": "geometry.mesh.stl",
                        },
                        {
                            "name": "mesh_obj",
                            "category": "artifact",
                            "artifact_kind": "geometry.mesh.obj",
                            "optional": True,
                        },
                        {
                            "name": "mesh_ply",
                            "category": "artifact",
                            "artifact_kind": "geometry.mesh.ply",
                            "optional": True,
                        },
                        {
                            "name": "geometry_checks",
                            "category": "artifact",
                            "artifact_kind": "checks.geometry",
                        },
                        {
                            "name": "geometry_diagnostics",
                            "category": "artifact",
                            "artifact_kind": "diagnostics.geometry",
                            "optional": True,
                        },
                        {
                            "name": "geometry_annotations",
                            "category": "artifact",
                            "artifact_kind": "geometry.annotations",
                            "optional": True,
                        },
                        {
                            "name": "license_notice",
                            "category": "artifact",
                            "artifact_kind": "license.notice",
                            "optional": True,
                        },
                        {
                            "name": "mesh.triangles",
                            "category": "metric",
                            "metric": "mesh.triangles",
                        },
                        {
                            "name": "mesh.seconds",
                            "category": "metric",
                            "metric": "mesh.seconds",
                        },
                    ],
                },
                "supports": {
                    "deterministic": True,
                    "cacheable": True,
                    "parallel_safe": False,
                    "restartable": True,
                    "streaming_logs": True,
                },
                "features": {
                    "export_formats": ["STL", "OBJ", "PLY"],
                    "ir_version": "1.0.0",
                    "ir_ops": SUPPORTED_OPS,
                    "evaluation_engines": ["vm", "jit", "auto"],
                    "meshing": ["manifold_dual_contouring"],
                },
                    "config_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "title": "builtin.geometry.fidget config",
                        "type": "object",
                        "additionalProperties": False,
                    "properties": {
                        "evaluation_engine": {
                            "type": "string",
                            "enum": ["auto", "jit", "vm"],
                            "default": "auto",
                        },
                        "domain_bbox": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {"type": "number"},
                            },
                            "description": "[[xmin,ymin,zmin],[xmax,ymax,zmax]]",
                        },
                        "mesh": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "depth": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 16,
                                    "default": 6,
                                },
                                "max_tris": {
                                    "type": "integer",
                                    "minimum": 1000,
                                    "default": 2000000,
                                },
                                "parallel": {"type": "boolean", "default": True},
                                "deterministic_order": {
                                    "type": "boolean",
                                    "default": True,
                                },
                            },
                        },
                        "exports": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "stl": {"type": "boolean", "default": True},
                                "obj": {"type": "boolean", "default": False},
                                "ply": {"type": "boolean", "default": False},
                            },
                        },
                        "cache": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "bytecode": {"type": "boolean", "default": False}
                            },
                        },
                    },
                    "required": ["domain_bbox"],
                },
                "notes": "Fidget-based implicit geometry backend. JIT requires AVX2 (x86_64) or NEON (aarch64).",
            },
        )

    def compile(
        self,
        ir: Dict[str, Any],
        ctx,
        *,
        export_formats: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ArtifactBundle:
        requested = export_formats or ["STL"]
        requested_norm = {str(fmt).upper() for fmt in requested}
        unsupported = requested_norm.difference({"STL", "OBJ", "PLY"})
        if unsupported:
            raise EngineGenError(
                Diagnostic(
                    code="E-EXPORT-FORMAT",
                    message=f"Unsupported export formats: {', '.join(sorted(unsupported))}",
                    hints=["Supported formats: STL, OBJ, PLY"],
                )
            )

        cfg = _normalize_config(config, ir)
        if "STL" in requested_norm:
            cfg.setdefault("exports", {})["stl"] = True
        if "OBJ" in requested_norm:
            cfg.setdefault("exports", {})["obj"] = True
        if "PLY" in requested_norm:
            cfg.setdefault("exports", {})["ply"] = True
        self._last_config = cfg

        enginegen_core = _load_enginegen_core()

        ir_json = json.dumps(ir, separators=(",", ":"), ensure_ascii=True)
        cfg_json = json.dumps(cfg, separators=(",", ":"), ensure_ascii=True)

        try:
            result_bytes = enginegen_core.fidget_compile_and_mesh(
                ir_json, cfg_json, str(ctx.run_dir / "artifacts")
            )
            result = json.loads(result_bytes)
        except Exception as exc:  # pragma: no cover - native extension only
            _raise_structured_error(exc)
            raise

        artifacts = ArtifactBundle()
        stl_path = Path(result["stl_path"])
        if cfg.get("exports", {}).get("stl", True):
            artifacts.add(
                ArtifactRef(
                    kind="geometry.mesh.stl",
                    path=stl_path,
                    producer=self.meta().name,
                    metadata={
                        "format": "STL",
                        "engine": result.get("engine"),
                        "domain_bbox": cfg.get("domain_bbox"),
                        "mesh": cfg.get("mesh"),
                        "evaluation_engine": cfg.get("evaluation_engine"),
                    },
                )
            )
        obj_path = result.get("obj_path")
        if obj_path:
            artifacts.add(
                ArtifactRef(
                    kind="geometry.mesh.obj",
                    path=Path(obj_path),
                    producer=self.meta().name,
                    metadata={"format": "OBJ"},
                )
            )
        ply_path = result.get("ply_path")
        if ply_path:
            artifacts.add(
                ArtifactRef(
                    kind="geometry.mesh.ply",
                    path=Path(ply_path),
                    producer=self.meta().name,
                    metadata={"format": "PLY"},
                )
            )

        if result.get("diagnostics"):
            diag_path = ctx.run_dir / "artifacts" / "geometry.diagnostics.json"
            diag_path.write_text(
                json.dumps({"diagnostics": result["diagnostics"]}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifacts.add(
                ArtifactRef(
                    kind="diagnostics.geometry",
                    path=diag_path,
                    producer=self.meta().name,
                    metadata={"count": len(result["diagnostics"])},
                )
            )

        annotations = ir.get("annotations")
        if annotations:
            ann_path = ctx.run_dir / "artifacts" / "geometry.annotations.json"
            ann_path.write_text(
                json.dumps(annotations, indent=2, sort_keys=True), encoding="utf-8"
            )
            artifacts.add(
                ArtifactRef(
                    kind="geometry.annotations",
                    path=ann_path,
                    producer=self.meta().name,
                    metadata={"groups": len(annotations.get("groups", []) or [])},
                )
            )

        notice_src = Path(__file__).resolve().parent / "licenses" / "fidget_notice.txt"
        if notice_src.exists():
            notice_path = ctx.run_dir / "artifacts" / "NOTICE.fidget.txt"
            notice_path.write_text(notice_src.read_text(encoding="utf-8"), encoding="utf-8")
            artifacts.add(
                ArtifactRef(
                    kind="license.notice",
                    path=notice_path,
                    producer=self.meta().name,
                    metadata={"dependency": "fidget", "license": "MPL-2.0"},
                )
            )

        metrics = result.get("metrics") or {}
        artifacts.metrics = metrics
        return artifacts

    def validate(self, ir: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        checks = extract_checks(ir)
        if not checks:
            return artifacts

        stl_ref = None
        for ref in artifacts.by_kind("geometry.mesh.stl"):
            stl_ref = ref
            break
        if stl_ref is None:
            return artifacts

        bbox = None
        depth = None
        if stl_ref.metadata:
            bbox = stl_ref.metadata.get("domain_bbox")
            mesh_meta = stl_ref.metadata.get("mesh") or {}
            depth = mesh_meta.get("depth")

        pitch = _voxel_pitch(bbox, depth)

        results: List[Dict[str, Any]] = []
        watertight_failed = False
        tri_count: Optional[int] = None
        tris: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]] = None
        for check in checks:
            op = check.get("op")
            params = check.get("params", {}) or {}
            if op in {"validate", "check.geometry"}:
                watertight, tri_count = _check_watertight(stl_ref.path)
                if not watertight:
                    watertight_failed = True
                results.append(
                    _check_result(
                        "watertight",
                        "PASS" if watertight else "FAIL",
                        value=watertight,
                        limit=True,
                        target="mesh",
                        hint="Inspect mesh for open boundaries",
                    )
                )
            if op in {"validate", "check.manufacturing"}:
                min_wall = params.get("min_wall_thickness")
                if min_wall is not None:
                    if tris is None:
                        tris = _load_stl_triangles(stl_ref.path)
                    sampled = _estimate_min_thickness(tris, bbox, depth, tri_count)
                    if sampled is not None:
                        results.append(
                            _check_result(
                                "min_wall_thickness_sampled",
                                "PASS" if sampled >= float(min_wall) else "WARN",
                                value=sampled,
                                limit=float(min_wall),
                                target="mesh",
                                hint="Increase thickness or mesh resolution",
                            )
                        )
                    results.append(
                        _resolution_check(
                            "min_wall_thickness",
                            float(min_wall),
                            pitch,
                            "Increase mesh depth or relax min_wall_thickness",
                        )
                    )
                min_feature = params.get("min_feature")
                if min_feature is None:
                    min_feature = params.get("min_feature_size")
                if min_feature is not None:
                    results.append(
                        _resolution_check(
                            "min_feature",
                            float(min_feature),
                            pitch,
                            "Increase mesh depth or relax min_feature",
                        )
                    )

        if results:
            summary = _summarize_checks(results)
            metrics = artifacts.metrics or {}
            metrics.update(
                {
                    "checks.geometry.count": summary["count"],
                    "checks.geometry.pass": summary["pass"],
                    "checks.geometry.warn": summary["warn"],
                    "checks.geometry.fail": summary["fail"],
                }
            )
            artifacts.metrics = metrics
            notes = artifacts.notes or {}
            notes["checks.geometry"] = {"summary": summary, "results": results}
            artifacts.notes = notes
            out_path = ctx.run_dir / "artifacts" / "checks.geometry.json"
            out_path.write_text(
                json.dumps({"checks": results}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifacts.add(
                ArtifactRef(
                    kind="checks.geometry",
                    path=out_path,
                    producer=self.meta().name,
                    metadata={"count": len(results)},
                )
            )
        if watertight_failed:
            raise EngineGenError(
                Diagnostic(
                    code="E-WATERTIGHT",
                    message="Mesh is not watertight",
                    hints=["Inspect mesh for open boundaries or increase mesh depth"],
                    location="checks.watertight",
                )
            )
        return artifacts


def _load_enginegen_core():
    try:
        import enginegen_core  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "enginegen_core native extension is required for builtin.geometry.fidget"
        ) from exc
    return enginegen_core


def _raise_structured_error(exc: Exception) -> None:
    message = str(exc)
    try:
        payload = json.loads(message)
    except Exception:
        return
    if not isinstance(payload, dict) or "code" not in payload or "message" not in payload:
        return
    hint = payload.get("hint")
    op_id = payload.get("op_id")
    diag = Diagnostic(
        code=str(payload.get("code")),
        message=str(payload.get("message")),
        hints=[str(hint)] if hint else [],
        location=str(op_id) if op_id else None,
        data={"op_id": op_id} if op_id else None,
    )
    raise EngineGenError(diag) from exc


def _normalize_config(config: Optional[Dict[str, Any]], ir: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg["mesh"] = dict(DEFAULT_MESH)
    cfg["exports"] = {"stl": True, "obj": False, "ply": False}
    cfg["cache"] = dict(DEFAULT_CACHE)
    if config:
        for key, value in config.items():
            if key in {"type", "libs"}:
                continue
            cfg[key] = value
    mesh = cfg.get("mesh") or {}
    merged_mesh = dict(DEFAULT_MESH)
    merged_mesh.update(mesh)
    cfg["mesh"] = merged_mesh
    exports = cfg.get("exports") or {}
    merged_exports = {"stl": True, "obj": False, "ply": False}
    merged_exports.update(exports)
    cfg["exports"] = merged_exports
    cache = cfg.get("cache") or {}
    merged_cache = dict(DEFAULT_CACHE)
    merged_cache.update(cache)
    cfg["cache"] = merged_cache

    if "domain_bbox" not in cfg:
        cfg["domain_bbox"] = _domain_bbox_from_ir(ir)
    if "domain_bbox" not in cfg or cfg["domain_bbox"] is None:
        raise EngineGenError(
            Diagnostic(
                code="E-DOMAIN-BBOX",
                message="domain_bbox is required for builtin.geometry.fidget",
                hints=["Provide domain_bbox in config or IR metadata"],
            )
        )
    _validate_bbox(cfg["domain_bbox"])
    return cfg


def _domain_bbox_from_ir(ir: Dict[str, Any]) -> Optional[Any]:
    domain = ir.get("domain")
    bbox = _extract_bbox(domain)
    if bbox is not None:
        return bbox
    metadata = ir.get("metadata") or {}
    if isinstance(metadata, dict):
        if "domain_bbox" in metadata:
            return _coerce_bbox(metadata.get("domain_bbox"))
        bbox = _extract_bbox(metadata.get("domain"))
        if bbox is not None:
            return bbox
    extensions = ir.get("extensions") or {}
    if isinstance(extensions, dict):
        if "domain_bbox" in extensions:
            return _coerce_bbox(extensions.get("domain_bbox"))
    return None


def _extract_bbox(value: Any) -> Optional[Any]:
    if not isinstance(value, dict):
        return None
    if "domain_bbox" in value:
        return _coerce_bbox(value.get("domain_bbox"))
    if "aabb" in value:
        return _coerce_bbox(value.get("aabb"))
    if "bbox" in value:
        return _coerce_bbox(value.get("bbox"))
    return None


def _coerce_bbox(value: Any) -> Optional[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if "min" in value and "max" in value:
            return [value.get("min"), value.get("max")]
    return None


def _validate_bbox(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise EngineGenError(
            Diagnostic(
                code="E-DOMAIN-BBOX",
                message="domain_bbox must be [[xmin,ymin,zmin],[xmax,ymax,zmax]]",
            )
        )
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            raise EngineGenError(
                Diagnostic(
                    code="E-DOMAIN-BBOX",
                    message="domain_bbox must be [[xmin,ymin,zmin],[xmax,ymax,zmax]]",
                )
            )
        for coord in item:
            if not isinstance(coord, (int, float)):
                raise EngineGenError(
                    Diagnostic(
                        code="E-DOMAIN-BBOX",
                        message="domain_bbox values must be numbers",
                    )
                )
    if value[0][0] >= value[1][0] or value[0][1] >= value[1][1] or value[0][2] >= value[1][2]:
        raise EngineGenError(
            Diagnostic(
                code="E-DOMAIN-BBOX",
                message="domain_bbox min must be < max on all axes",
            )
        )


def _voxel_pitch(bbox: Any, depth: Any) -> Optional[float]:
    if bbox is None or depth is None:
        return None
    try:
        dx = float(bbox[1][0]) - float(bbox[0][0])
        dy = float(bbox[1][1]) - float(bbox[0][1])
        dz = float(bbox[1][2]) - float(bbox[0][2])
        extent = max(abs(dx), abs(dy), abs(dz))
        return extent / (2 ** int(depth))
    except Exception:
        return None


def _resolution_check(rule: str, limit: float, pitch: Optional[float], hint: str) -> Dict[str, Any]:
    if pitch is None:
        return _check_result(
            rule,
            "WARN",
            value=None,
            limit=limit,
            target="mesh",
            hint="Resolution unknown; cannot verify against limit",
        )
    status = "PASS" if pitch <= limit else "WARN"
    return _check_result(
        rule,
        status,
        value=pitch,
        limit=limit,
        target="mesh",
        hint=hint,
    )


def _load_stl_triangles(
    path: Path,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    data = path.read_bytes()
    if len(data) < 84:
        return []
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected != len(data):
        return []
    tris: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []
    offset = 84
    for _ in range(tri_count):
        vals = struct.unpack_from("<12fH", data, offset)
        v1 = (float(vals[3]), float(vals[4]), float(vals[5]))
        v2 = (float(vals[6]), float(vals[7]), float(vals[8]))
        v3 = (float(vals[9]), float(vals[10]), float(vals[11]))
        tris.append((v1, v2, v3))
        offset += 50
    return tris


def _estimate_min_thickness(
    tris: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]],
    bbox: Any,
    depth: Any,
    tri_count: Optional[int],
) -> Optional[float]:
    if not tris or bbox is None:
        return None
    if tri_count is not None and tri_count > 200_000:
        return None
    try:
        bmin = [float(bbox[0][0]), float(bbox[0][1]), float(bbox[0][2])]
        bmax = [float(bbox[1][0]), float(bbox[1][1]), float(bbox[1][2])]
    except Exception:
        return None
    extent = [bmax[i] - bmin[i] for i in range(3)]
    if any(e <= 0 for e in extent):
        return None
    samples = 6
    try:
        if depth is not None:
            samples = max(3, min(10, 2 ** min(int(depth), 3)))
    except Exception:
        samples = 6
    min_thickness = None
    eps = max(extent) * 1e-6
    axes = [0, 1, 2]
    for axis in axes:
        a0 = bmin[axis] - eps
        adir = [0.0, 0.0, 0.0]
        adir[axis] = 1.0
        other = [i for i in axes if i != axis]
        for i in range(samples):
            for j in range(samples):
                p = [0.0, 0.0, 0.0]
                p[axis] = a0
                p[other[0]] = bmin[other[0]] + (i + 0.5) / samples * extent[other[0]]
                p[other[1]] = bmin[other[1]] + (j + 0.5) / samples * extent[other[1]]
                hits: List[float] = []
                for v1, v2, v3 in tris:
                    t = _ray_intersect_triangle(p, adir, v1, v2, v3)
                    if t is not None:
                        hits.append(t)
                if not hits:
                    continue
                hits.sort()
                filtered: List[float] = []
                for t in hits:
                    if not filtered or abs(t - filtered[-1]) > eps:
                        filtered.append(t)
                for k in range(0, len(filtered) - 1, 2):
                    thickness = filtered[k + 1] - filtered[k]
                    if thickness <= 0:
                        continue
                    if min_thickness is None or thickness < min_thickness:
                        min_thickness = thickness
    return min_thickness


def _ray_intersect_triangle(
    origin: List[float],
    direction: List[float],
    v0: Tuple[float, float, float],
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float],
) -> Optional[float]:
    eps = 1e-9

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    edge1 = sub(v1, v0)
    edge2 = sub(v2, v0)
    h = cross(direction, edge2)
    a = dot(edge1, h)
    if -eps < a < eps:
        return None
    f = 1.0 / a
    s = sub(tuple(origin), v0)
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = cross(s, edge1)
    v = f * dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * dot(edge2, q)
    if t > eps:
        return t
    return None


def _check_watertight(path: Path) -> Tuple[bool, Optional[int]]:
    data = path.read_bytes()
    if len(data) < 84:
        return False, None
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected != len(data):
        # Fallback to ASCII heuristic
        text = path.read_text(encoding="utf-8", errors="ignore")
        return "facet normal" in text and text.strip().lower().endswith("endsolid"), None

    vert_index: Dict[Tuple[float, float, float], int] = {}
    edges: Dict[Tuple[int, int], int] = {}
    offset = 84
    for _ in range(tri_count):
        vals = struct.unpack_from("<12fH", data, offset)
        v1 = (vals[3], vals[4], vals[5])
        v2 = (vals[6], vals[7], vals[8])
        v3 = (vals[9], vals[10], vals[11])
        i1 = _vertex_id(vert_index, v1)
        i2 = _vertex_id(vert_index, v2)
        i3 = _vertex_id(vert_index, v3)
        for a, b in ((i1, i2), (i2, i3), (i3, i1)):
            edge = (a, b) if a < b else (b, a)
            edges[edge] = edges.get(edge, 0) + 1
        offset += 50
    return all(count == 2 for count in edges.values()), tri_count


def _vertex_id(cache: Dict[Tuple[float, float, float], int], vert: Tuple[float, float, float]) -> int:
    key = (round(vert[0], 6), round(vert[1], 6), round(vert[2], 6))
    if key in cache:
        return cache[key]
    idx = len(cache)
    cache[key] = idx
    return idx


def _check_result(rule: str, status: str, *, value: Any, limit: Any, target: str, hint: str) -> Dict[str, Any]:
    return {
        "rule": rule,
        "status": status,
        "value": value,
        "limit": limit,
        "target": target,
        "hint": hint,
    }


def _summarize_checks(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for result in results:
        status = result.get("status")
        if status in counts:
            counts[status] += 1
    return {
        "count": len(results),
        "pass": counts["PASS"],
        "warn": counts["WARN"],
        "fail": counts["FAIL"],
    }
