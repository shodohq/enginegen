# Geometry IR

Geometry IR is a reduced instruction set used to decouple synthesis logic from geometry
kernels. It is validated against `schemas/ir.schema.json`.

## Structure

- `ir_version`: schema version
- `ops`: DAG of operations (`id`, `op`, `inputs`, `args`)
- `outputs`: named outputs (`main` or `solid` required)
- `annotations`: face/volume groups and port definitions
- `checks`: geometry/manufacturing validation requests
- `dialect`: optional dialect marker for backend-specific op sets

## Core Ops (v1)

The core set is intentionally small. See `enginegen/core/ir.py` for the allow-list.
Common ops include:

- `sketch.circle`, `sketch.rect`, `sketch.polyline`
- `solid.extrude`, `solid.revolve`, `solid.loft`, `solid.boolean`
- `annotate.group`, `annotate.port`
- `check.geometry`, `check.manufacturing`

## Dialects

The repository includes a dialect schema for an implicit geometry backend:

- `enginegen.implicit.fidget.v1` in `schemas/dialects/enginegen.implicit.fidget.v1.schema.json`

Example:

```json
{
  "ir_version": "1.0.0",
  "dialect": "enginegen.implicit.fidget.v1",
  "ops": [
    {"id": "c", "op": "fidget.prim.circle", "args": {"center": [0.0, 0.0], "radius": 0.1}},
    {"id": "rev", "op": "fidget.gen.revolve_y", "inputs": ["c"], "args": {"offset": 0.0}}
  ],
  "outputs": {"main": "rev"}
}
```

## Validation

`enginegen/core/ir.py` validates:

- supported op names
- missing references
- output integrity
- DAG cycles

Backends can interpret `checks` to enforce manufacturing rules or geometry quality.
