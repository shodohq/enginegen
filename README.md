# EngineGen

EngineGen is a modular engine generation pipeline that turns requirements into
manufacturable geometry, runs analysis, and feeds results back into the generator.
It treats design as an algorithmic generator (CEM-like), not a shape-generation AI.

## Highlights

- Contract-first schemas: EngineSpec, SystemGraph, Geometry IR, Artifact Manifest
- OpenFOAM-style swapping: select plugins via `type` and optional `libs`
- Geometry IR decouples synthesis logic from geometry kernels
- Provenance and caching: per-run manifests, hashes, and logs
- Python-first with optional Rust core and native plugin ABI

## Architecture

```
Spec/Schema -> SystemGraph -> Synthesis -> Geometry Backend -> Analysis/Optimization -> Feedback
```

## Quick Start

```bash
python -m enginegen.cli run -c examples/case.yaml
```

Outputs are written to `runs/<run_id>/` with `spec.normalized.json`, `graph.json`,
`ir.json`, `config.yaml`, `artifacts/`, `logs/`, and `manifest.json`.

## Examples

- `examples/spec.yaml` - sample EngineSpec
- `examples/graph.json` - sample SystemGraph
- `examples/case.yaml` - case config
- `examples/ir/` - Geometry IR examples (implicit dialect)

## Built-in Plugins

- `baseline_rule` (synthesizer)
- `simple_stl` (geometry backend)
- `builtin.geometry.fidget` (implicit geometry backend)
- `noop` (adapter)
- `scalar_metrics` (analysis)

Note: `builtin.geometry.fidget` requires the native `enginegen_core` extension (Rust).

## Documentation

- `docs/README.md` - documentation index
- `docs/overview.md` - concept and pipeline intent
- `docs/architecture.md` - architecture and module boundaries
- `docs/spec.md` - EngineSpec schema
- `docs/system-graph.md` - SystemGraph schema
- `docs/geometry-ir.md` - Geometry IR structure and dialects
- `docs/plugins.md` - plugin API and discovery
- `docs/pipeline.md` - case config, execution, artifacts
- `docs/native-plugins.md` - native plugin ABI
- `docs/roadmap.md` - phased roadmap

## Project Status

This repository provides a minimal but end-to-end pipeline and stable contracts. Domain-specific
engineering logic belongs in custom synthesis/analysis plugins.
