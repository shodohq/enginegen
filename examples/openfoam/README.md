# OpenFOAM Example

This folder contains a minimal case template scaffold used by the `openfoam_cfd` adapter.
It is intentionally incomplete and is meant to be customized for your solver and meshing
workflow (e.g. `blockMesh`, `snappyHexMesh`, `simpleFoam`).

Suggested flow:
- Place the STL produced by EngineGen in `constant/triSurface/nozzle.stl`.
- Customize the dictionaries under `system/` and `constant/`.
- Set `mode: run` and `commands:` in `examples/case.yaml`.

The default example runs in `mode: mock` so it does not require OpenFOAM.
