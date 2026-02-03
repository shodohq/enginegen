# Native Plugins (C ABI)

EngineGen can load native plugins via a small C ABI defined in
`rust/enginegen_plugin_abi/include/enginegen_plugin.h` and the Python
wrapper in `enginegen/core/native_plugin.py`.

## Required Symbols

A native shared library must export:

- `enginegen_plugin_descriptor()` -> UTF-8 JSON string
- `enginegen_plugin_call(method, input, input_len, output, output_len)` -> int
- `enginegen_plugin_free(ptr)`

`enginegen_plugin_descriptor()` returns JSON like:

```json
{
  "name": "plugin_name",
  "kind": "synthesizer|geometry_backend|adapter|analysis|optimization",
  "api_version": "1.0.0",
  "plugin_version": "0.1.0",
  "capabilities": {"capabilities_version": "1.0.0", "plugin_kind": "...", "io": {...}}
}
```

## Methods

`enginegen_plugin_call` dispatches by method name and JSON payload:

- `synthesize`: `{spec, graph, ctx, feedback?}` -> `{ir}`
- `compile`: `{ir, ctx, export_formats?}` -> `{artifacts}`
- `validate`: `{ir, artifacts, ctx}` -> `{artifacts}`
- `run`: `{spec, artifacts, ctx}` -> `{artifacts}`
- `evaluate`: `{spec, artifacts, ctx}` -> `{artifacts}`
- `optimize`: `{spec, artifacts, ctx}` -> `{artifacts}`

The wrapper converts payloads and returns an `ArtifactBundle` consistent
with the Python plugin API.
