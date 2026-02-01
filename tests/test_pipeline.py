from __future__ import annotations

from pathlib import Path
import tempfile

from enginegen.core.config import normalize_case_config
from enginegen.core.graph import load_graph, validate_graph
from enginegen.core.pipeline import run_pipeline
from enginegen.core.plugin_loader import PluginRegistry
from enginegen.core.spec import load_spec, validate_spec


def test_pipeline_runs():
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "examples" / "spec.yaml")
    graph = load_graph(root / "examples" / "graph.json")

    registry = PluginRegistry()
    registry.discover()

    schema_dir = root / "schemas"

    cfg = normalize_case_config(
        {
            "synthesizer": {"type": "baseline_rule"},
            "geometry": {"backend": {"type": "simple_stl"}, "export": ["STL"]},
            "pipeline": [{"type": "noop"}],
            "analysis": [{"type": "scalar_metrics"}],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = run_pipeline(cfg, spec, graph, registry, Path(tmpdir), schema_dir)
        assert (run_dir / "manifest.json").exists()


def test_spec_and_graph_valid():
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "examples" / "spec.yaml")
    graph = load_graph(root / "examples" / "graph.json")
    spec_diag = validate_spec(spec, root / "schemas" / "engine_spec.schema.json")
    graph_diag = validate_graph(graph)
    assert not spec_diag.has_errors()
    assert not graph_diag.has_errors()
