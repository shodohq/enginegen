# Roadmap (Draft)

This is a phased plan aligned with the contract-first, plugin-based architecture.

## Phase 0: Contracts

- Freeze EngineSpec, SystemGraph, Geometry IR schemas
- Define plugin API compatibility policy
- Establish artifact manifest format

## Phase 1: End-to-End Geometry

- Spec -> IR -> geometry export (STL/STEP/mesh)
- Minimal run pipeline with reproducible artifacts

## Phase 2: Low-Fidelity Analysis

- Add at least one analysis or adapter plugin
- Record metrics and feedback into the manifest

## Phase 3: High-Fidelity Adapters

- External solver adapters with sandboxed execution
- Stable I/O contracts for replacement or upgrades

## Phase 4: Optimization / Surrogates

- Optional optimization driver plugin
- Surrogate integration based on recorded metrics

## Phase 5: Manufacturing Rules as First-Class Inputs

- Encode manufacturing constraints in Spec and Geometry IR checks
- Enforce validation in geometry backends
