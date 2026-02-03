# EngineSpec

EngineSpec is the primary input contract for requirements, constraints, and manufacturing
rules. It is validated against `schemas/engine_spec.schema.json`.

## Required Fields

- `spec_version`
- `name`
- `requirements`
- `constraints`
- `manufacturing`

## Requirements

`requirements.targets` is a map keyed by metric name. Each target defines a goal
(minimize, maximize, match, meet_or_exceed, meet_or_below) and a value or range.

Example:

```yaml
spec_version: "1.0.0"
name: demo
requirements:
  targets:
    thrust:
      goal: meet_or_exceed
      target: { value: 100, unit: kN }
constraints: {}
manufacturing:
  process: SLM
```

## Constraints

Constraints allow packaging, interface ports, and named limits:

- `constraints.interfaces` - named ports (fluid, thermal, structural, ...)
- `constraints.limits` - named limits with severity and ranges
- `constraints.design_rules` - free-form rule container

## Manufacturing

`manufacturing.rules` captures min feature size, min wall thickness, overhang limits, and
other process-specific constraints. These are surfaced as checks in Geometry IR.

## Extensions

Use `metadata` and `extensions` for plugin-specific fields without breaking the core schema.
