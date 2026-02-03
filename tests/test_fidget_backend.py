import json
from pathlib import Path

import pytest

from enginegen.core.plugin_api import HostContext
from enginegen.plugins_builtin.geometry_fidget import FidgetGeometryBackend


def _load_ir(name: str):
    path = Path(__file__).resolve().parent.parent / "examples" / "ir" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _ctx(tmp_path: Path, run_id: str) -> HostContext:
    run_dir = tmp_path / run_id
    artifacts = run_dir / "artifacts"
    cache = run_dir / "cache"
    logs = run_dir / "logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return HostContext(
        run_id=run_id,
        run_dir=run_dir,
        artifact_dir=artifacts,
        cache_dir=cache,
        logs_dir=logs,
    )


def _require_enginegen_core():
    try:
        import enginegen_core  # noqa: F401
    except Exception:
        pytest.skip("enginegen_core native extension not available")


def test_fidget_depth_monotonic(tmp_path):
    _require_enginegen_core()
    ir = _load_ir("fidget_sphere_box.json")
    backend = FidgetGeometryBackend()

    cfg_low = {
        "evaluation_engine": "vm",
        "mesh": {"depth": 3, "parallel": False, "deterministic_order": True},
        "exports": {"stl": True},
    }
    cfg_high = {
        "evaluation_engine": "vm",
        "mesh": {"depth": 4, "parallel": False, "deterministic_order": True},
        "exports": {"stl": True},
    }

    bundle_low = backend.compile(ir, _ctx(tmp_path, "low"), config=cfg_low)
    bundle_high = backend.compile(ir, _ctx(tmp_path, "high"), config=cfg_high)

    low = bundle_low.metrics.get("mesh.triangles") if bundle_low.metrics else None
    high = bundle_high.metrics.get("mesh.triangles") if bundle_high.metrics else None
    assert low is not None and high is not None
    assert high >= low


def test_fidget_domain_bbox_from_metadata(tmp_path):
    _require_enginegen_core()
    ir = _load_ir("fidget_extrude.json")
    backend = FidgetGeometryBackend()
    cfg = {
        "evaluation_engine": "vm",
        "mesh": {"depth": 2, "parallel": False, "deterministic_order": True},
        "exports": {"stl": True},
    }
    bundle = backend.compile(ir, _ctx(tmp_path, "metadata"), config=cfg)
    assert bundle.by_kind("geometry.mesh.stl")


def test_fidget_domain_bbox_from_domain(tmp_path):
    _require_enginegen_core()
    ir = _load_ir("fidget_sphere_box.json")
    ir.pop("metadata", None)
    ir["domain"] = {"aabb": {"min": [-0.2, -0.2, -0.2], "max": [0.2, 0.2, 0.2]}}
    backend = FidgetGeometryBackend()
    cfg = {
        "evaluation_engine": "vm",
        "mesh": {"depth": 2, "parallel": False, "deterministic_order": True},
        "exports": {"stl": True},
    }
    bundle = backend.compile(ir, _ctx(tmp_path, "domain"), config=cfg)
    assert bundle.by_kind("geometry.mesh.stl")
