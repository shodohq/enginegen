# Architecture

## Layered Model

```
[A] Spec/Schema
[B] SystemGraph
[C] Synthesis Plugins
[D] Geometry Backend Plugins
[E] Analysis/Adapter Plugins
[F] Feedback/Optimization
```

- Spec/Schema defines the vocabulary and constraints.
- SystemGraph models dependencies via ports and edges.
- Synthesis produces Geometry IR and annotations/checks.
- Geometry Backend compiles IR to CAD/mesh artifacts.
- Analysis/Adapter consumes artifacts and produces metrics or results.

## Repo Boundaries

- `enginegen/core/` - Contracts, validation, pipeline, artifact store, plugin loader
- `schemas/` - JSON schemas (Spec, Graph, IR, Manifest, Capabilities)
- `enginegen/plugins_builtin/` - Built-in baseline plugins
- `rust/enginegen_core/` - Optional Rust core (canonicalization + fidget)
- `rust/enginegen_geom_fidget/` - Fidget-based implicit geometry backend
- `rust/enginegen_plugin_abi/` - Native plugin ABI header

## Contracts and Versions

- `spec_version`: `schemas/engine_spec.schema.json`
- `graph_version`: `schemas/graph.schema.json`
- `ir_version`: `schemas/ir.schema.json`
- `capabilities_version`: `schemas/capabilities.schema.json`
- `manifest_version`: `schemas/manifest.schema.json`
- `api_version`: `enginegen/core/plugin_api.py`

## Built-in Plugins

- Synthesizer: `baseline_rule`
- Geometry: `simple_stl`, `builtin.geometry.fidget`
- Adapter: `noop`
- Analysis: `scalar_metrics`

These are registered via entry points in `pyproject.toml` and as a fallback
in `enginegen/core/plugin_loader.py`.
