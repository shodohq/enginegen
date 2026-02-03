# SystemGraph

SystemGraph models the engine as a directed graph of components connected by typed ports.
It is validated against `schemas/graph.schema.json`.

## Core Concepts

- Node: functional block (e.g., chamber, nozzle)
- Port: interface on a node with `kind` and `direction`
- Edge: connection between node ports (must respect port kind)

## Example

```json
{
  "graph_version": "1.0.0",
  "nodes": [
    {"id": "chamber", "kind": "chamber", "ports": [{"id": "out", "kind": "fluid", "direction": "out"}]},
    {"id": "nozzle", "kind": "nozzle", "ports": [{"id": "in", "kind": "fluid", "direction": "in"}]}
  ],
  "edges": [
    {"from": {"node": "chamber", "port": "out"}, "to": {"node": "nozzle", "port": "in"}, "kind": "fluid"}
  ]
}
```

## Validation

`enginegen/core/graph.py` performs:

- cycle detection
- missing port references
- port kind mismatch
- required port checks
