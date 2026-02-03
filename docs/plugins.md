# Plugins

EngineGen supports runtime-swappable plugins inspired by OpenFOAM's `type`/`libs` pattern.
Plugins are selected by string key in the case config, and optionally discovered via
entry points or native libraries.

## Plugin Kinds

- `synthesizer`: Spec + Graph -> Geometry IR
- `geometry_backend`: IR -> CAD/mesh artifacts
- `adapter`: external tool integration (mesh/solver/etc)
- `analysis`: artifacts -> metrics or derived artifacts
- `optimization`: optional driver layer

## Plugin API

See `enginegen/core/plugin_api.py` for the interface definitions.
Each plugin must return `PluginMeta` with:

- `name`
- `api_version`
- `plugin_version`
- `capabilities` (validated against `schemas/capabilities.schema.json`)

## Discovery

Python entry points are supported via `enginegen/core/plugin_loader.py`:

- `enginegen.synthesizer`
- `enginegen.geometry_backend`
- `enginegen.adapter`
- `enginegen.analysis`
- `enginegen.optimization`

Example entry point:

```toml
[project.entry-points."enginegen.synthesizer"]
baseline_rule = "my_pkg.synth:BaselineSynth"
```

`libs` in the case config can include Python packages or native shared libraries (`.so`,
`.dylib`, `.dll`).

## Case Config Example

```yaml
engineCase:
  synthesizer: { type: rocket_nozzle }
  geometry:
    backend: { type: builtin.geometry.fidget }
    export: ["STL"]
  pipeline:
    - type: openfoam_cfd
      mode: mock
      case_template: examples/openfoam/case_template
  analysis:
    - { type: openfoam_metrics }
  optimization:
    type: nozzle_tuner
```

## Built-in Plugins

- `baseline_rule` (synthesizer)
- `rocket_nozzle` (synthesizer, outputs Fidget dialect IR)
- `simple_stl` (geometry_backend)
- `builtin.geometry.fidget` (geometry_backend, implicit)
- `noop` (adapter)
- `openfoam_cfd` (adapter)
- `scalar_metrics` (analysis)
- `openfoam_metrics` (analysis)
- `nozzle_tuner` (optimization)

`builtin.geometry.fidget` depends on the native `enginegen_core` extension.
