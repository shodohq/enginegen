# EngineGen

Minimal implementation of the engine generation pipeline described in the spec and design documents.

## Quick start

```bash
python -m enginegen.cli run -c examples/case.yaml
```

Outputs are written to `runs/<run_id>/` with `spec.normalized.json`, `graph.json`, `ir.json`, `config.yaml`, `artifacts/`, `logs/`, and `manifest.json`.

## CLI

- `enginegen validate -c <case.yaml>`
- `enginegen build -c <case.yaml>`
- `enginegen run -c <case.yaml>`
- `enginegen report -r runs/<run_id>`
