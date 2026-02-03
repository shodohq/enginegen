# Overview

EngineGen is a modular pipeline for generating manufacturable rocket engine geometry from
requirements, then running analysis and feeding results back into the generator.
It treats design as an algorithmic generator (CEM-like) rather than a shape-generation AI.

The core idea is to keep the contracts stable and swap implementations at runtime, in the same
spirit as OpenFOAM's dictionary-driven selection (`type` and optional `libs`).

## Pipeline

```
Spec/Schema
  -> SystemGraph
  -> Synthesis (generator rules + parameters)
  -> Geometry Backend (compile IR -> CAD/mesh)
  -> Analysis / Optimization (adapters + evaluators)
  -> Feedback
```

## Key Principles

- Contract-first: EngineSpec, SystemGraph, GeometryIR, and Artifact Manifest are versioned
  schemas before any specific solver or kernel choice.
- Pluggable by string key: select implementations with `type` and optionally load
  additional libraries via `libs`.
- Geometry decoupling: Geometry IR sits between synthesis logic and geometry kernels.
- Provenance: every run produces a manifest with hashes, plugin versions, and logs.

## Scope Note

EngineGen focuses on the pipeline and interfaces. Domain-specific engine sizing or
performance algorithms live inside user-provided synthesis plugins.
