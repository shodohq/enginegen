from __future__ import annotations

import json
from pathlib import Path
import tempfile

from enginegen.core.graph import load_graph
from enginegen.core.ir import validate_ir, validate_ir_schema
from enginegen.core.pipeline import run_pipeline
from enginegen.core.plugin_api import API_VERSION
from enginegen.core.plugin_loader import PluginRegistry
from enginegen.core.spec import load_spec, normalize_spec, validate_spec


def test_ir_schema_and_validation():
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "examples" / "spec.yaml")
    graph = load_graph(root / "examples" / "graph.json")

    registry = PluginRegistry()
    registry.discover()

    synth = registry.get("synthesizer", "baseline_rule")
    normalized_spec, _ = normalize_spec(spec)
    ir = synth.synthesize(normalized_spec, graph, ctx=None)

    schema_diag = validate_ir_schema(ir, root / "schemas" / "ir.schema.json")
    assert not schema_diag.has_errors()
    ir_diag = validate_ir(ir)
    assert not ir_diag.has_errors()


def test_ir_rejects_unknown_op():
    ir = {
        "ir_version": "1.0.0",
        "ops": [{"id": "n1", "op": "unknown"}],
        "outputs": {"main": "n1"},
    }
    diag = validate_ir(ir)
    assert any(d.code == "E-IR-OP" for d in diag.items)


def test_ir_rejects_missing_reference():
    ir = {
        "ir_version": "1.0.0",
        "ops": [
            {"id": "n1", "op": "sketch.circle", "inputs": ["missing"], "args": {"radius": 1.0}}
        ],
        "outputs": {"main": "n1"},
    }
    diag = validate_ir(ir)
    assert any(d.code == "E-IR-REF" for d in diag.items)


def test_ir_rejects_cycles():
    ir = {
        "ir_version": "1.0.0",
        "ops": [
            {"id": "a", "op": "sketch.circle", "inputs": ["b"], "args": {"radius": 1.0}},
            {"id": "b", "op": "sketch.circle", "inputs": ["a"], "args": {"radius": 1.0}},
        ],
        "outputs": {"main": "a"},
    }
    diag = validate_ir(ir)
    assert any(d.code == "E-IR-CYCLE" for d in diag.items)


def test_manifest_contains_provenance_and_external_runs():
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "examples" / "spec.yaml")
    graph = load_graph(root / "examples" / "graph.json")

    registry = PluginRegistry()
    registry.discover()

    cfg = {
        "synthesizer": {"type": "baseline_rule"},
        "geometry": {"backend": {"type": "simple_stl"}, "export": ["STL"]},
        "pipeline": [{"type": "noop"}],
        "analysis": [{"type": "scalar_metrics"}],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_pipeline(cfg, spec, graph, registry, Path(tmpdir), root / "schemas")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        environment = manifest.get("environment", {})
        assert environment.get("os") is not None
        diagnostics = manifest.get("diagnostics", {})
        external_runs = diagnostics.get("external_runs", [])
        assert external_runs, "external_runs should be recorded"
        assert all(run.get("sandboxed") for run in external_runs)


def test_plugin_api_contract():
    registry = PluginRegistry()
    registry.discover()
    expected = [
        ("synthesizer", "baseline_rule"),
        ("geometry_backend", "simple_stl"),
        ("adapter", "noop"),
        ("analysis", "scalar_metrics"),
    ]
    for kind, name in expected:
        plugin = registry.get(kind, name)
        meta = plugin.meta()
        assert meta.name == name
        assert meta.api_version.split(".")[0] == API_VERSION.split(".")[0]
        assert meta.plugin_version


def test_spec_schema_rejects_missing_version():
    root = Path(__file__).resolve().parents[1]
    spec = {
        "name": "invalid",
        "requirements": {"targets": {}},
        "constraints": {},
        "manufacturing": {"process": "SLM"},
    }
    diag = validate_spec(spec, root / "schemas" / "engine_spec.schema.json")
    assert diag.has_errors()
