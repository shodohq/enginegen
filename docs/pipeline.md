# Pipeline Execution

## CLI

- `enginegen validate -c <case.yaml>`
- `enginegen build -c <case.yaml>`
- `enginegen run -c <case.yaml>`
- `enginegen report -r runs/<run_id>`

## Case Config

The CLI accepts a case config file. You can either put everything under
`engineCase` (as in `examples/case.yaml`) or use a flat top-level layout.

Minimal example:

```yaml
spec: spec.yaml
graph: graph.json
engineCase:
  synthesizer:
    type: baseline_rule
  geometry:
    backend:
      type: simple_stl
    export: ["STL"]
  pipeline:
    - { type: noop }
  analysis:
    - { type: scalar_metrics }
```

## Execution Flow

1. Validate Spec and Graph schemas
2. Normalize Spec (units, defaults)
3. Run synthesizer -> Geometry IR
4. Compile IR -> geometry artifacts
5. Run adapters (pipeline) -> external tools
6. Run analysis plugins -> metrics
7. Write artifacts + manifest

## Output Layout

Each run is written to `runs/<run_id>/` with:

- `spec.normalized.json`
- `graph.json`
- `ir.json`
- `config.yaml`
- `artifacts/`
- `logs/`
- `manifest.json`

The manifest schema is `schemas/manifest.schema.json`.

## Provenance and Caching

- Input hashes (Spec, Graph, IR, config, plugin lock) are recorded in the manifest
- Geometry results are cached in `cache/` using a stable `CacheKey`
- Plugin versions and capabilities are recorded per run

## Sandboxing and Audit

External tools run through `enginegen/core/external.py` and can be sandboxed
using `enginegen/core/sandbox_wrapper.py` with environment variables. Audit and
access control can be enabled via case config (`audit`, `access_control`).
