from __future__ import annotations

import inspect
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from .artifacts import ArtifactStore, collect_environment, hash_bytes, safe_run_id
from .cache import CacheKey, CacheStore, hash_config
from .diagnostics import Diagnostic, Diagnostics
from .ir import canonical_json as ir_canonical_json
from .ir import validate_ir, validate_ir_schema
from .logging import get_event_logger, get_logger
from .plugin_api import ArtifactBundle, ArtifactRef, HostContext, PluginMeta
from .plugin_loader import PluginRegistry
from .spec import canonical_json as spec_canonical_json
from .spec import normalize_spec, validate_spec
from .graph import validate_graph, validate_graph_schema
from .version import __version__
from .audit import load_access_controller, load_audit_logger


def run_pipeline(
    case_config: Dict[str, Any],
    spec: Dict[str, Any],
    graph: Dict[str, Any],
    registry: PluginRegistry,
    root: Path,
    schema_dir: Path,
    run_id: Optional[str] = None,
) -> Path:
    run_dir, _ = execute_pipeline(
        case_config,
        spec,
        graph,
        registry,
        root,
        schema_dir,
        run_id=run_id,
    )
    return run_dir


def execute_pipeline(
    case_config: Dict[str, Any],
    spec: Dict[str, Any],
    graph: Dict[str, Any],
    registry: PluginRegistry,
    root: Path,
    schema_dir: Path,
    run_id: Optional[str] = None,
) -> Tuple[Path, ArtifactBundle]:
    store = ArtifactStore(root)
    run_id = run_id or safe_run_id()
    run_dir = store.create_run(run_id)
    logs_dir = run_dir / "logs"
    logger = get_logger("pipeline", logs_dir)
    event_logger = get_event_logger(logs_dir)
    audit_logger = load_audit_logger(case_config.get("audit"), logs_dir)
    access_controller = load_access_controller(case_config.get("access_control"))

    artifacts = ArtifactBundle()
    ctx: Optional[HostContext] = None
    inputs_snapshot: Dict[str, Any] = {}
    inputs_meta: Dict[str, Any] = {}
    plugin_lock: List[Dict[str, Any]] = []
    normalized_spec: Optional[Dict[str, Any]] = None
    ir: Optional[Dict[str, Any]] = None
    input_hashes: Dict[str, str] = {}
    cache_key: Optional[CacheKey] = None
    cache_hit = False
    budget: Dict[str, Any] = {}
    budget_state: Dict[str, Any] = {
        "start": time.monotonic(),
        "hf_runs": 0,
        "lf_runs": 0,
        "total_runs": 0,
        "skipped": [],
    }
    error_info: Optional[Dict[str, Any]] = None
    success = False
    step_metrics: List[Dict[str, Any]] = []
    pipeline_start = time.perf_counter()
    feedback_hash: str = ""

    _log_event(event_logger, "pipeline.start", run_id=run_id)
    _audit_event(audit_logger, "pipeline.start", run_id=run_id)

    inputs_snapshot["config.yaml"] = _dump_config(case_config)
    inputs_snapshot["config.resolved.json"] = case_config
    store.write_inputs(run_dir, inputs_snapshot)

    try:
        diagnostics = Diagnostics()
        diagnostics.extend(validate_spec(spec, schema_dir / "engine_spec.schema.json"))
        diagnostics.extend(validate_graph_schema(graph, schema_dir / "graph.schema.json"))
        diagnostics.extend(validate_graph(graph))

        normalized_spec, spec_diags = normalize_spec(spec)
        diagnostics.extend(spec_diags)
        inputs_snapshot["spec.normalized.json"] = normalized_spec
        inputs_snapshot["graph.json"] = graph
        store.write_inputs(run_dir, inputs_snapshot)

        if diagnostics.has_errors():
            _write_diagnostics(run_dir, diagnostics)
            diagnostics.raise_for_errors()

        export_formats = _normalize_export_formats(case_config.get("geometry", {}).get("export"))
        seed = _resolve_seed(case_config, normalized_spec)
        nondeterminism = _normalize_nondeterminism(case_config.get("nondeterminism"))
        inputs_meta.update(
            {
                "seed": seed,
                "nondeterminism": nondeterminism,
            }
        )

        ctx = HostContext(
            run_id=run_id,
            run_dir=run_dir,
            artifact_dir=run_dir / "artifacts",
            cache_dir=run_dir / "cache",
            logs_dir=logs_dir,
            export_formats=export_formats,
            seed=seed,
            nondeterminism=nondeterminism,
            audit=audit_logger,
            access=access_controller,
        )

        budget = _parse_budget(normalized_spec)
        budget_state = {
            "start": time.monotonic(),
            "hf_runs": 0,
            "lf_runs": 0,
            "total_runs": 0,
            "skipped": [],
        }

        feedback_bundle, feedback_meta, feedback_hash = _load_feedback_bundle(case_config.get("feedback"))
        if feedback_meta:
            inputs_meta["feedback"] = feedback_meta

        synth_cfg = case_config.get("synthesizer", {})
        synth = registry.get("synthesizer", synth_cfg["type"])
        _ensure_capabilities(synth.meta(), schema_dir, run_dir)
        _ensure_plugin_kind(synth.meta(), "synthesizer")
        logger.info("Synthesizer: %s", synth_cfg["type"])
        plugin_lock.append(_plugin_lock_entry("synthesizer", synth.meta()))
        inputs_snapshot["plugins.lock.json"] = plugin_lock
        store.write_inputs(run_dir, inputs_snapshot)

        _access_check(
            access_controller,
            "synthesizer",
            {"plugin": synth.meta().name, "run_id": run_id},
            audit_logger,
        )
        _log_event(
            event_logger,
            "plugin.start",
            run_id=run_id,
            kind="synthesizer",
            plugin=synth.meta().name,
            step="synthesize",
            inputs=["spec.normalized.json", "graph.json"],
        )
        _audit_event(
            audit_logger,
            "plugin.start",
            run_id=run_id,
            kind="synthesizer",
            plugin=synth.meta().name,
            step="synthesize",
        )
        step_start = time.perf_counter()
        try:
            ir = synth.synthesize(normalized_spec, graph, ctx, feedback_bundle)
        except Exception as exc:
            _log_event(
                event_logger,
                "plugin.error",
                run_id=run_id,
                kind="synthesizer",
                plugin=synth.meta().name,
                step="synthesize",
                elapsed_s=time.perf_counter() - step_start,
                error=str(exc),
            )
            _audit_event(
                audit_logger,
                "plugin.error",
                run_id=run_id,
                kind="synthesizer",
                plugin=synth.meta().name,
                step="synthesize",
                error=str(exc),
            )
            raise
        _record_step(step_metrics, "synthesize", synth.meta().name, step_start)
        inputs_snapshot["ir.json"] = ir
        store.write_inputs(run_dir, inputs_snapshot)
        _log_event(
            event_logger,
            "plugin.end",
            run_id=run_id,
            kind="synthesizer",
            plugin=synth.meta().name,
            step="synthesize",
            elapsed_s=time.perf_counter() - step_start,
            outputs=["ir.json"],
        )
        _audit_event(
            audit_logger,
            "plugin.end",
            run_id=run_id,
            kind="synthesizer",
            plugin=synth.meta().name,
            step="synthesize",
        )

        ir_diags = validate_ir_schema(ir, schema_dir / "ir.schema.json")
        ir_diags.extend(validate_ir(ir))
        dialect = ir.get("dialect")
        if isinstance(dialect, str) and dialect:
            dialect_path = schema_dir / "dialects" / f"{dialect}.schema.json"
            if dialect_path.exists():
                ir_diags.extend(validate_ir_schema(ir, dialect_path))
        if ir_diags.has_errors():
            _write_diagnostics(run_dir, ir_diags)
            ir_diags.raise_for_errors()

        geom_cfg = case_config.get("geometry", {}).get("backend", {})
        geom = registry.get("geometry_backend", geom_cfg["type"])
        _ensure_capabilities(geom.meta(), schema_dir, run_dir)
        _ensure_plugin_kind(geom.meta(), "geometry_backend")
        _enforce_export_support(geom.meta(), export_formats)
        logger.info("Geometry backend: %s", geom_cfg["type"])
        _check_ir_compat(geom.meta(), ir.get("ir_version"))
        _check_ir_ops(geom.meta(), ir)
        plugin_lock.append(_plugin_lock_entry("geometry_backend", geom.meta()))
        inputs_snapshot["plugins.lock.json"] = plugin_lock
        store.write_inputs(run_dir, inputs_snapshot)

        spec_hash = _hash_spec(normalized_spec)
        graph_hash = _hash_graph(graph)
        ir_hash = _hash_ir(ir)
        config_hash = _hash_config(case_config)
        plugin_lock_hash = _hash_plugin_lock(plugin_lock)

        input_hashes = {
            "spec_hash": spec_hash,
            "graph_hash": graph_hash,
            "ir_hash": ir_hash,
            "config_hash": config_hash,
            "plugins_lock_hash": plugin_lock_hash,
        }

        cache_store = CacheStore(root / "cache")
        cache_key = CacheKey(
            spec_hash=spec_hash,
            graph_hash=graph_hash,
            ir_hash=ir_hash,
            plugin_lock_hash=plugin_lock_hash,
            config_hash=config_hash,
            feedback_hash=feedback_hash,
        )
        _access_check(
            access_controller,
            "geometry_backend",
            {"plugin": geom.meta().name, "run_id": run_id},
            audit_logger,
        )
        _log_event(
            event_logger,
            "plugin.start",
            run_id=run_id,
            kind="geometry_backend",
            plugin=geom.meta().name,
            step="geometry.compile",
            inputs=["ir.json"],
            export_formats=export_formats,
        )
        _audit_event(
            audit_logger,
            "plugin.start",
            run_id=run_id,
            kind="geometry_backend",
            plugin=geom.meta().name,
            step="geometry.compile",
        )

        cached_bundle = _load_cached_bundle(cache_store, cache_key, ctx)
        if cached_bundle is not None:
            logger.info("Geometry cache hit: %s", cache_key.to_string())
            artifacts = cached_bundle
            cache_hit = True
            _record_step(step_metrics, "geometry.cache_hit", geom.meta().name, time.perf_counter())
            _log_event(
                event_logger,
                "geometry.cache_hit",
                run_id=run_id,
                plugin=geom.meta().name,
                cache_key=cache_key.to_string(),
                outputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "geometry.cache_hit",
                run_id=run_id,
                plugin=geom.meta().name,
                cache_key=cache_key.to_string(),
            )
            _log_event(
                event_logger,
                "plugin.end",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.compile",
                cache_hit=True,
                outputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "plugin.end",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.compile",
                cache_hit=True,
            )
        else:
            step_start = time.perf_counter()
            try:
                artifacts = _call_geometry_compile(geom, ir, ctx, export_formats, geom_cfg)
            except Exception as exc:
                _log_event(
                    event_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="geometry_backend",
                    plugin=geom.meta().name,
                    step="geometry.compile",
                    elapsed_s=time.perf_counter() - step_start,
                    error=str(exc),
                )
                _audit_event(
                    audit_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="geometry_backend",
                    plugin=geom.meta().name,
                    step="geometry.compile",
                    error=str(exc),
                )
                raise
            _record_step(step_metrics, "geometry.compile", geom.meta().name, step_start)
            _log_event(
                event_logger,
                "plugin.end",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.compile",
                elapsed_s=time.perf_counter() - step_start,
                outputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "plugin.end",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.compile",
            )

            _log_event(
                event_logger,
                "plugin.start",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.validate",
                inputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "plugin.start",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.validate",
            )
            step_start = time.perf_counter()
            try:
                artifacts = geom.validate(ir, artifacts, ctx)
            except Exception as exc:
                _log_event(
                    event_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="geometry_backend",
                    plugin=geom.meta().name,
                    step="geometry.validate",
                    elapsed_s=time.perf_counter() - step_start,
                    error=str(exc),
                )
                _audit_event(
                    audit_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="geometry_backend",
                    plugin=geom.meta().name,
                    step="geometry.validate",
                    error=str(exc),
                )
                raise
            _record_step(step_metrics, "geometry.validate", geom.meta().name, step_start)
            _log_event(
                event_logger,
                "plugin.end",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.validate",
                elapsed_s=time.perf_counter() - step_start,
                outputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "plugin.end",
                run_id=run_id,
                kind="geometry_backend",
                plugin=geom.meta().name,
                step="geometry.validate",
            )
            _store_cache_bundle(cache_store, cache_key, artifacts)

        for step_cfg in case_config.get("pipeline", []) or []:
            if not _budget_allows(step_cfg, budget, budget_state, logger, default_fidelity="high"):
                continue
            adapter = registry.get("adapter", step_cfg["type"])
            _ensure_capabilities(adapter.meta(), schema_dir, run_dir)
            _ensure_plugin_kind(adapter.meta(), "adapter")
            required_inputs = _require_plugin_inputs(adapter.meta(), artifacts, "adapter")
            logger.info("Adapter: %s", step_cfg["type"])
            plugin_lock.append(_plugin_lock_entry("adapter", adapter.meta()))
            inputs_snapshot["plugins.lock.json"] = plugin_lock
            store.write_inputs(run_dir, inputs_snapshot)
            before_runs = len(artifacts.external_runs)
            before_logs = len(artifacts.by_kind("log.external.run"))
            before_stdout = len(artifacts.by_kind("log.external.stdout"))
            before_stderr = len(artifacts.by_kind("log.external.stderr"))
            _access_check(
                access_controller,
                "adapter",
                {"plugin": adapter.meta().name, "run_id": run_id},
                audit_logger,
            )
            _log_event(
                event_logger,
                "plugin.start",
                run_id=run_id,
                kind="adapter",
                plugin=adapter.meta().name,
                step="adapter.run",
                inputs=_artifact_refs(artifacts, required_inputs),
            )
            _audit_event(
                audit_logger,
                "plugin.start",
                run_id=run_id,
                kind="adapter",
                plugin=adapter.meta().name,
                step="adapter.run",
            )
            step_start = time.perf_counter()
            try:
                artifacts = adapter.run(normalized_spec, artifacts, ctx)
            except Exception as exc:
                _log_event(
                    event_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="adapter",
                    plugin=adapter.meta().name,
                    step="adapter.run",
                    elapsed_s=time.perf_counter() - step_start,
                    error=str(exc),
                )
                _audit_event(
                    audit_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="adapter",
                    plugin=adapter.meta().name,
                    step="adapter.run",
                    error=str(exc),
                )
                raise
            _record_step(step_metrics, "adapter.run", adapter.meta().name, step_start)
            if len(artifacts.external_runs) == before_runs and len(
                artifacts.by_kind("log.external.run")
            ) == before_logs:
                raise RuntimeError(
                    f"Adapter {adapter.meta().name} did not record external run logs"
                )
            if len(artifacts.by_kind("log.external.stdout")) == before_stdout:
                raise RuntimeError(
                    f"Adapter {adapter.meta().name} did not record external stdout logs"
                )
            if len(artifacts.by_kind("log.external.stderr")) == before_stderr:
                raise RuntimeError(
                    f"Adapter {adapter.meta().name} did not record external stderr logs"
                )
            stdout_refs = {str(ref.path) for ref in artifacts.by_kind("log.external.stdout")}
            stderr_refs = {str(ref.path) for ref in artifacts.by_kind("log.external.stderr")}
            for run in artifacts.external_runs[before_runs:]:
                if str(run.stdout) not in stdout_refs:
                    raise RuntimeError(
                        f"Adapter {adapter.meta().name} missing stdout artifact for {run.name}"
                    )
                if str(run.stderr) not in stderr_refs:
                    raise RuntimeError(
                        f"Adapter {adapter.meta().name} missing stderr artifact for {run.name}"
                    )
                if not run.sandboxed:
                    raise RuntimeError(
                        f"Adapter {adapter.meta().name} executed external run without sandbox"
                    )
                if not run.tool_version:
                    raise RuntimeError(
                        f"Adapter {adapter.meta().name} did not record tool_version for {run.name}"
                    )
            _log_event(
                event_logger,
                "plugin.end",
                run_id=run_id,
                kind="adapter",
                plugin=adapter.meta().name,
                step="adapter.run",
                elapsed_s=time.perf_counter() - step_start,
                outputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "plugin.end",
                run_id=run_id,
                kind="adapter",
                plugin=adapter.meta().name,
                step="adapter.run",
            )
            _increment_budget(step_cfg, budget_state, default_fidelity="high")
            if _budget_exhausted(budget, budget_state):
                logger.warning("Analysis budget exhausted; stopping pipeline steps")
                break

        for step_cfg in case_config.get("analysis", []) or []:
            if not _budget_allows(step_cfg, budget, budget_state, logger, default_fidelity="low"):
                continue
            analysis = registry.get("analysis", step_cfg["type"])
            _ensure_capabilities(analysis.meta(), schema_dir, run_dir)
            _ensure_plugin_kind(analysis.meta(), "analysis")
            required_inputs = _require_plugin_inputs(analysis.meta(), artifacts, "analysis")
            logger.info("Analysis: %s", step_cfg["type"])
            plugin_lock.append(_plugin_lock_entry("analysis", analysis.meta()))
            inputs_snapshot["plugins.lock.json"] = plugin_lock
            store.write_inputs(run_dir, inputs_snapshot)
            _access_check(
                access_controller,
                "analysis",
                {"plugin": analysis.meta().name, "run_id": run_id},
                audit_logger,
            )
            _log_event(
                event_logger,
                "plugin.start",
                run_id=run_id,
                kind="analysis",
                plugin=analysis.meta().name,
                step="analysis.evaluate",
                inputs=_artifact_refs(artifacts, required_inputs),
            )
            _audit_event(
                audit_logger,
                "plugin.start",
                run_id=run_id,
                kind="analysis",
                plugin=analysis.meta().name,
                step="analysis.evaluate",
            )
            step_start = time.perf_counter()
            try:
                artifacts = analysis.evaluate(normalized_spec, artifacts, ctx)
            except Exception as exc:
                _log_event(
                    event_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="analysis",
                    plugin=analysis.meta().name,
                    step="analysis.evaluate",
                    elapsed_s=time.perf_counter() - step_start,
                    error=str(exc),
                )
                _audit_event(
                    audit_logger,
                    "plugin.error",
                    run_id=run_id,
                    kind="analysis",
                    plugin=analysis.meta().name,
                    step="analysis.evaluate",
                    error=str(exc),
                )
                raise
            _record_step(step_metrics, "analysis.evaluate", analysis.meta().name, step_start)
            _log_event(
                event_logger,
                "plugin.end",
                run_id=run_id,
                kind="analysis",
                plugin=analysis.meta().name,
                step="analysis.evaluate",
                elapsed_s=time.perf_counter() - step_start,
                outputs=_artifact_refs(artifacts),
            )
            _audit_event(
                audit_logger,
                "plugin.end",
                run_id=run_id,
                kind="analysis",
                plugin=analysis.meta().name,
                step="analysis.evaluate",
            )
            _increment_budget(step_cfg, budget_state, default_fidelity="low")
            if _budget_exhausted(budget, budget_state):
                logger.warning("Analysis budget exhausted; stopping analysis steps")
                break

        opt_cfg = _select_optimization(case_config)
        if opt_cfg is not None:
            if _budget_allows(opt_cfg, budget, budget_state, logger, default_fidelity="high"):
                optimizer = registry.get("optimization", opt_cfg["type"])
                _ensure_capabilities(optimizer.meta(), schema_dir, run_dir)
                _ensure_plugin_kind(optimizer.meta(), "optimization")
                required_inputs = _require_plugin_inputs(optimizer.meta(), artifacts, "optimization")
                logger.info("Optimization: %s", opt_cfg["type"])
                plugin_lock.append(_plugin_lock_entry("optimization", optimizer.meta()))
                inputs_snapshot["plugins.lock.json"] = plugin_lock
                store.write_inputs(run_dir, inputs_snapshot)
                _access_check(
                    access_controller,
                    "optimization",
                    {"plugin": optimizer.meta().name, "run_id": run_id},
                    audit_logger,
                )
                _log_event(
                    event_logger,
                    "plugin.start",
                    run_id=run_id,
                    kind="optimization",
                    plugin=optimizer.meta().name,
                    step="optimization.optimize",
                    inputs=_artifact_refs(artifacts, required_inputs),
                )
                _audit_event(
                    audit_logger,
                    "plugin.start",
                    run_id=run_id,
                    kind="optimization",
                    plugin=optimizer.meta().name,
                    step="optimization.optimize",
                )
                step_start = time.perf_counter()
                try:
                    artifacts = optimizer.optimize(normalized_spec, artifacts, ctx)
                except Exception as exc:
                    _log_event(
                        event_logger,
                        "plugin.error",
                        run_id=run_id,
                        kind="optimization",
                        plugin=optimizer.meta().name,
                        step="optimization.optimize",
                        elapsed_s=time.perf_counter() - step_start,
                        error=str(exc),
                    )
                    _audit_event(
                        audit_logger,
                        "plugin.error",
                        run_id=run_id,
                        kind="optimization",
                        plugin=optimizer.meta().name,
                        step="optimization.optimize",
                        error=str(exc),
                    )
                    raise
                _record_step(step_metrics, "optimization.optimize", optimizer.meta().name, step_start)
                _log_event(
                    event_logger,
                    "plugin.end",
                    run_id=run_id,
                    kind="optimization",
                    plugin=optimizer.meta().name,
                    step="optimization.optimize",
                    elapsed_s=time.perf_counter() - step_start,
                    outputs=_artifact_refs(artifacts),
                )
                _audit_event(
                    audit_logger,
                    "plugin.end",
                    run_id=run_id,
                    kind="optimization",
                    plugin=optimizer.meta().name,
                    step="optimization.optimize",
                )
                _increment_budget(opt_cfg, budget_state, default_fidelity="high")
            if _budget_exhausted(budget, budget_state):
                logger.warning("Analysis budget exhausted; stopping optimization")

        success = True
    except Exception as exc:
        error_info = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            status = "success" if success else "failed"
            _log_event(
                event_logger,
                "pipeline.end",
                run_id=run_id,
                status=status,
                elapsed_s=time.perf_counter() - pipeline_start,
                error=error_info,
            )
            _audit_event(
                audit_logger,
                "pipeline.end",
                run_id=run_id,
                status=status,
                error=error_info,
            )

            store.add_artifacts(run_dir, artifacts)
            if inputs_snapshot:
                store.write_inputs(run_dir, inputs_snapshot)
            _add_input_refs(run_dir, artifacts)

            if not input_hashes:
                spec_hash = _hash_spec(normalized_spec) if normalized_spec is not None else hash_bytes(b"")
                graph_hash = _hash_graph(graph) if graph is not None else hash_bytes(b"")
                ir_hash = _hash_ir(ir) if ir is not None else hash_bytes(b"")
                config_hash = _hash_config(case_config)
                plugin_lock_hash = _hash_plugin_lock(plugin_lock)
                input_hashes = {
                    "spec_hash": spec_hash,
                    "graph_hash": graph_hash,
                    "ir_hash": ir_hash,
                    "config_hash": config_hash,
                    "plugins_lock_hash": plugin_lock_hash,
                }
            else:
                input_hashes["plugins_lock_hash"] = _hash_plugin_lock(plugin_lock)

            environment = collect_environment()
            if hasattr(audit_logger, "path"):
                environment["audit_log"] = str(getattr(audit_logger, "path"))

            diagnostics: Dict[str, Any] = {
                "external_runs": [run.to_dict() for run in artifacts.external_runs],
                "budget": budget,
                "skipped_steps": budget_state.get("skipped", []),
                "status": status,
                "steps": step_metrics,
                "inputs_meta": inputs_meta,
            }
            if ctx is not None:
                diagnostics["determinism"] = {
                    "seed": ctx.seed,
                    "nondeterminism": ctx.nondeterminism,
                }
            if artifacts.notes:
                diagnostics["artifact_notes"] = artifacts.notes
            if error_info:
                diagnostics["error"] = error_info
            if cache_key is not None:
                diagnostics["cache"] = {"hit": cache_hit, "key": cache_key.to_string()}

            metrics = {"total_elapsed_s": time.perf_counter() - pipeline_start, **_resource_snapshot()}
            if artifacts.metrics:
                metrics.update(artifacts.metrics)

            host = {
                "name": "enginegen",
                "version": __version__,
                "platform": platform.platform(),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            }

            manifest = store.build_manifest(
                run_id=run_id,
                status=status,
                host=host,
                inputs=input_hashes,
                plugins=plugin_lock,
                bundle=artifacts,
                environment=environment,
                diagnostics=diagnostics,
                metrics=metrics,
            )
            store.write_manifest(run_dir, manifest)
        except Exception:
            if success:
                raise

    return run_dir, artifacts


def _log_event(event_logger, event: str, **data: Any) -> None:
    if event_logger is None:
        return
    payload = {"event": event}
    payload.update(data)
    try:
        event_logger.record(payload)
    except Exception:
        return


def _audit_event(audit_logger, event: str, **data: Any) -> None:
    if audit_logger is None:
        return
    payload = {"event": event}
    payload.update(data)
    try:
        audit_logger.record(payload)
    except Exception:
        return


def _access_check(access_controller, action: str, context: Dict[str, Any], audit_logger) -> None:
    if access_controller is None:
        return
    try:
        access_controller.check(action, context)
    except Exception as exc:
        _audit_event(
            audit_logger,
            "access.denied",
            action=action,
            context=context,
            error=str(exc),
        )
        raise


def _artifact_refs(bundle: ArtifactBundle, kinds: Optional[List[str]] = None) -> List[Dict[str, str]]:
    items = bundle.items
    if kinds is not None:
        items = [ref for ref in items if ref.kind in kinds]
    return [{"kind": ref.kind, "path": str(ref.path)} for ref in items]


def _normalize_export_formats(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(fmt).upper() for fmt in value]
    return [str(value).upper()]


def _resolve_seed(case_config: Dict[str, Any], spec: Dict[str, Any]) -> Optional[int]:
    seed = case_config.get("seed")
    if seed is None:
        seed = case_config.get("random_seed")
    if seed is None:
        seed = spec.get("seed")
    if seed is None:
        seed = spec.get("random_seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except Exception as exc:
        raise ValueError(f"seed must be an integer: {seed}") from exc


def _normalize_nondeterminism(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _call_geometry_compile(
    geom,
    ir: Dict[str, Any],
    ctx: HostContext,
    export_formats: Optional[List[str]],
    config: Optional[Dict[str, Any]] = None,
) -> ArtifactBundle:
    try:
        sig = inspect.signature(geom.compile)
    except (TypeError, ValueError):
        return geom.compile(ir, ctx)
    kwargs: Dict[str, Any] = {}
    if "export_formats" in sig.parameters:
        kwargs["export_formats"] = export_formats
    if "config" in sig.parameters:
        kwargs["config"] = config
    if kwargs:
        return geom.compile(ir, ctx, **kwargs)
    return geom.compile(ir, ctx)


def _hash_spec(spec: Dict[str, Any]) -> str:
    return hash_bytes(spec_canonical_json(spec))


def _hash_graph(graph: Dict[str, Any]) -> str:
    payload = json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hash_bytes(payload)


def _hash_ir(ir: Dict[str, Any]) -> str:
    return hash_bytes(ir_canonical_json(ir))


def _hash_config(config: Dict[str, Any]) -> str:
    return hash_config(config)


def _hash_plugin_lock(plugin_lock: List[Dict[str, Any]]) -> str:
    payload = json.dumps(plugin_lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hash_bytes(payload)


def _write_diagnostics(run_dir: Path, diagnostics: Diagnostics) -> None:
    path = run_dir / "diagnostics.json"
    path.write_text(json.dumps(diagnostics.to_list(), indent=2), encoding="utf-8")


def _add_input_refs(run_dir: Path, artifacts: ArtifactBundle) -> None:
    mapping = {
        "spec.normalized.json": "input.spec",
        "graph.json": "input.graph",
        "ir.json": "input.ir",
        "config.yaml": "input.config",
        "config.resolved.json": "input.config.resolved",
        "plugins.lock.json": "input.plugins.lock",
    }
    existing = {ref.path.resolve() for ref in artifacts.items}
    for filename, kind in mapping.items():
        path = (run_dir / filename).resolve()
        if not path.exists():
            continue
        if path in existing:
            continue
        artifacts.add(
            ArtifactRef(
                kind=kind,
                path=path,
                producer="enginegen",
            )
        )


def _dump_config(config: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(config, sort_keys=True)
    except Exception:
        return json.dumps(config, indent=2, sort_keys=True)


def _major(version: Optional[str]) -> str:
    if not version:
        return ""
    return str(version).split(".")[0]


def _capability_feature(meta: PluginMeta, key: str) -> Any:
    cap = meta.capabilities or {}
    features = cap.get("features")
    if isinstance(features, dict) and key in features:
        return features[key]
    return cap.get(key)


def _capability_inputs(meta: PluginMeta) -> Optional[List[Any]]:
    cap = meta.capabilities or {}
    io = cap.get("io")
    if isinstance(io, dict):
        inputs = io.get("inputs")
        if isinstance(inputs, list):
            return inputs
    legacy_inputs = cap.get("inputs")
    if isinstance(legacy_inputs, list):
        return legacy_inputs
    return None


def _ensure_plugin_kind(meta: PluginMeta, expected: str) -> None:
    declared = meta.capabilities.get("plugin_kind")
    if declared and declared != expected:
        raise RuntimeError(
            f"Plugin kind mismatch for {meta.name}: expected {expected}, got {declared}"
        )


def _check_ir_compat(meta: PluginMeta, ir_version: Optional[str]) -> None:
    cap = _capability_feature(meta, "ir_version")
    if not cap or not ir_version:
        return
    if _major(cap) != _major(ir_version):
        raise RuntimeError(
            f"IR version mismatch: backend {meta.name} supports {cap}, got {ir_version}"
        )


def _check_ir_ops(meta: PluginMeta, ir: Dict[str, Any]) -> None:
    supported = _capability_feature(meta, "ir_ops")
    if not supported:
        return
    if not isinstance(supported, list):
        raise RuntimeError("geometry_backend capabilities.features.ir_ops must be a list")
    supported_set = {str(op) for op in supported}
    allow_ext = "ext.*" in supported_set
    used_ops = {
        str(node.get("op"))
        for node in (ir.get("ops") or ir.get("nodes") or [])
        if isinstance(node, dict) and node.get("op") is not None
    }
    unsupported = [
        op for op in sorted(used_ops) if op not in supported_set and not (allow_ext and op.startswith("ext."))
    ]
    if unsupported:
        raise RuntimeError(
            f"Geometry backend {meta.name} does not support IR ops: {', '.join(unsupported)}"
        )


def _ensure_capabilities(meta: PluginMeta, schema_dir: Path, run_dir: Path) -> None:
    schema_path = schema_dir / "capabilities.schema.json"
    diagnostics = _validate_capabilities(meta, schema_path)
    if diagnostics.has_errors():
        _write_diagnostics(run_dir, diagnostics)
        diagnostics.raise_for_errors()


def _validate_capabilities(meta: PluginMeta, schema_path: Path) -> Diagnostics:
    diagnostics = Diagnostics()
    if not schema_path.exists():
        return diagnostics
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        diagnostics.add(
            Diagnostic(
                code="E-CAP-SCHEMA-LOAD",
                message=f"Failed to load capabilities schema: {exc}",
                location=str(schema_path),
            )
        )
        return diagnostics
    schema["$id"] = schema_path.resolve().as_uri()
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(meta.capabilities), key=str):
        location = "/".join(str(x) for x in error.path)
        diagnostics.add(
            Diagnostic(
                code="E-PLUGIN-CAPABILITIES",
                message=f"{meta.name}: {error.message}",
                location=f"capabilities/{location}" if location else "capabilities",
            )
        )
    return diagnostics


def _require_capability(meta: PluginMeta, key: str, kind: str) -> Any:
    if key not in meta.capabilities:
        raise RuntimeError(f"{kind} plugin {meta.name} missing capability '{key}'")
    return meta.capabilities[key]


def _require_plugin_inputs(meta: PluginMeta, artifacts: ArtifactBundle, kind: str) -> List[str]:
    required = _capability_inputs(meta)
    if required is None:
        return []
    missing: List[str] = []
    required_kinds: List[str] = []
    for item in required:
        if isinstance(item, str):
            required_kinds.append(item)
            if not artifacts.by_kind(item):
                missing.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("optional"):
            continue
        category = item.get("category")
        if category == "artifact":
            kind_name = item.get("artifact_kind") or item.get("name")
            if kind_name:
                required_kinds.append(str(kind_name))
            if kind_name and not artifacts.by_kind(kind_name):
                missing.append(str(kind_name))
        elif category == "metric":
            metric_name = item.get("metric") or item.get("name")
            metrics = artifacts.metrics or {}
            if metric_name and metric_name not in metrics:
                missing.append(str(metric_name))
    if missing:
        raise RuntimeError(
            f"{kind} plugin {meta.name} missing required inputs: {', '.join(missing)}"
        )
    return required_kinds


def _enforce_export_support(meta: PluginMeta, requested) -> None:
    if not requested:
        return
    supported = _capability_feature(meta, "export_formats")
    if supported is None:
        return
    if not isinstance(supported, list):
        raise RuntimeError("geometry_backend capabilities.features.export_formats must be a list")
    supported_norm = [str(fmt).upper() for fmt in supported]
    missing = [fmt for fmt in requested if str(fmt).upper() not in supported_norm]
    if missing:
        raise RuntimeError(
            f"Geometry backend {meta.name} does not support export formats: {', '.join(missing)}"
        )


def _plugin_lock_entry(kind: str, meta: PluginMeta) -> Dict[str, Any]:
    capabilities_hash = _hash_capabilities(meta.capabilities)
    return {
        "plugin_kind": kind,
        "name": meta.name,
        "api_version": meta.api_version,
        "plugin_version": meta.plugin_version,
        "capabilities": meta.capabilities,
        "capabilities_hash": capabilities_hash,
    }


def _hash_capabilities(capabilities: Dict[str, Any]) -> str:
    payload = json.dumps(capabilities, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hash_bytes(payload)


def _parse_budget(spec: Dict[str, Any]) -> Dict[str, Any]:
    budget = spec.get("analysis_budget", {}) or {}
    max_hf = budget.get("hf_runs") if budget.get("hf_runs") is not None else budget.get("high_fidelity_runs")
    max_lf = budget.get("lf_runs")
    max_wall = budget.get("time_limit_sec") if budget.get("time_limit_sec") is not None else budget.get("max_walltime_s")
    max_total = budget.get("max_total_runs")
    min_priority = budget.get("min_priority")
    return {
        "hf_runs": int(max_hf) if max_hf is not None else None,
        "lf_runs": int(max_lf) if max_lf is not None else None,
        "time_limit_sec": int(max_wall) if max_wall is not None else None,
        "max_total_runs": int(max_total) if max_total is not None else None,
        "min_priority": int(min_priority) if min_priority is not None else None,
    }


def _budget_elapsed(state: Dict[str, Any]) -> float:
    return time.monotonic() - state["start"]


def _budget_allows(
    step_cfg: Dict[str, Any],
    budget: Dict[str, Any],
    state: Dict[str, Any],
    logger,
    *,
    default_fidelity: str,
) -> bool:
    min_priority = budget.get("min_priority")
    priority = step_cfg.get("priority", 0)
    try:
        priority_value = int(priority)
    except Exception:
        priority_value = 0
    if min_priority is not None and priority_value < min_priority:
        _record_skip(state, step_cfg, "priority below min_priority")
        logger.warning("Skipping step due to priority threshold")
        return False
    max_total = budget.get("max_total_runs")
    if max_total is not None and state.get("total_runs", 0) >= max_total:
        _record_skip(state, step_cfg, "max_total_runs exceeded")
        logger.warning("Skipping step due to total runs budget")
        return False
    max_wall = budget.get("time_limit_sec")
    if max_wall is not None and _budget_elapsed(state) >= max_wall:
        _record_skip(state, step_cfg, "time_limit_sec exceeded")
        logger.warning("Skipping step due to walltime budget")
        return False
    fidelity = step_cfg.get("fidelity", default_fidelity)
    max_hf = budget.get("hf_runs")
    if fidelity != "low" and max_hf is not None and state["hf_runs"] >= max_hf:
        _record_skip(state, step_cfg, "hf_runs exceeded")
        logger.warning("Skipping step due to hf_runs budget")
        return False
    max_lf = budget.get("lf_runs")
    if fidelity == "low" and max_lf is not None and state.get("lf_runs", 0) >= max_lf:
        _record_skip(state, step_cfg, "lf_runs exceeded")
        logger.warning("Skipping step due to lf_runs budget")
        return False
    return True


def _increment_budget(step_cfg: Dict[str, Any], state: Dict[str, Any], *, default_fidelity: str) -> None:
    fidelity = step_cfg.get("fidelity", default_fidelity)
    if fidelity != "low":
        state["hf_runs"] += 1
    else:
        state["lf_runs"] = state.get("lf_runs", 0) + 1
    state["total_runs"] = state.get("total_runs", 0) + 1


def _budget_exhausted(budget: Dict[str, Any], state: Dict[str, Any]) -> bool:
    max_total = budget.get("max_total_runs")
    if max_total is not None and state.get("total_runs", 0) >= max_total:
        return True
    max_wall = budget.get("time_limit_sec")
    if max_wall is not None and _budget_elapsed(state) >= max_wall:
        return True
    max_hf = budget.get("hf_runs")
    if max_hf is not None and state["hf_runs"] >= max_hf:
        return True
    return False


def _record_skip(state: Dict[str, Any], step_cfg: Dict[str, Any], reason: str) -> None:
    state["skipped"].append({"type": step_cfg.get("type"), "reason": reason})


def _record_step(metrics: List[Dict[str, Any]], step: str, plugin: str, start: float) -> None:
    elapsed = time.perf_counter() - start
    payload = {"step": step, "plugin": plugin, "elapsed_s": elapsed}
    payload.update(_resource_snapshot())
    metrics.append(payload)


def _resource_snapshot() -> Dict[str, Any]:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"rss_kb": usage.ru_maxrss}
    except Exception:
        return {}


def _select_optimization(case_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    opt_cfg = case_config.get("optimization")
    if not isinstance(opt_cfg, dict):
        return None
    if "driver" in opt_cfg and isinstance(opt_cfg["driver"], dict):
        opt_cfg = opt_cfg["driver"]
    if not isinstance(opt_cfg, dict):
        return None
    if not opt_cfg.get("type"):
        return None
    if str(opt_cfg.get("type")).lower() == "none":
        return None
    return opt_cfg


def _load_feedback_bundle(
    feedback_cfg: Any,
) -> Tuple[Optional[ArtifactBundle], Optional[Dict[str, Any]], str]:
    if not feedback_cfg:
        return None, None, ""
    manifest_path: Optional[Path] = None
    if isinstance(feedback_cfg, str):
        manifest_path = Path(feedback_cfg)
    elif isinstance(feedback_cfg, dict):
        if feedback_cfg.get("manifest"):
            manifest_path = Path(feedback_cfg["manifest"])
        elif feedback_cfg.get("run_dir"):
            manifest_path = Path(feedback_cfg["run_dir"]) / "manifest.json"
        elif feedback_cfg.get("path"):
            manifest_path = Path(feedback_cfg["path"])
    if manifest_path is None:
        return None, None, ""
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feedback manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = ArtifactBundle()
    items = manifest.get("artifacts") or manifest.get("items") or []
    for item in items:
        uri = Path(item.get("path") or item.get("uri") or "")
        if not uri.is_absolute():
            uri = (manifest_path.parent / uri).resolve()
        producer = item.get("producer")
        if isinstance(producer, dict):
            producer = producer.get("name")
        bundle.add(
            ArtifactRef(
                kind=item.get("kind", "unknown"),
                path=uri,
                content_hash=item.get("content_hash"),
                producer=producer,
                metadata=item.get("metadata", {}),
            )
        )
    bundle.notes = {"manifest": manifest}
    meta = {"manifest": str(manifest_path)}
    feedback_hash = hash_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )
    meta["hash"] = feedback_hash
    return bundle, meta, feedback_hash


def _is_cacheable(ref: ArtifactRef) -> bool:
    if ref.kind.startswith("log."):
        return False
    if ref.kind.startswith(("cad.", "mesh.", "geometry.", "validation", "checks.")):
        return True
    return False


def _load_cached_bundle(
    cache_store: CacheStore, cache_key: CacheKey, ctx: HostContext
) -> Optional[ArtifactBundle]:
    cache_dir = cache_store.get(cache_key)
    if cache_dir is None:
        return None
    meta_path = cache_dir / "cache.json"
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    bundle = ArtifactBundle()
    for item in data.get("items", []) or []:
        src = cache_dir / "artifacts" / item["file"]
        dest = ctx.artifact_dir / item["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        bundle.add(
            ArtifactRef(
                kind=item["kind"],
                path=dest,
                content_hash=item.get("content_hash"),
                producer=item.get("producer"),
                metadata=item.get("metadata", {}),
            )
        )
    return bundle


def _store_cache_bundle(cache_store: CacheStore, cache_key: CacheKey, bundle: ArtifactBundle) -> None:
    cache_dir = cache_store.root / cache_key.to_string()
    artifacts_dir = cache_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for ref in bundle.items:
        if not _is_cacheable(ref):
            continue
        dest = artifacts_dir / ref.path.name
        shutil.copy2(ref.path, dest)
        items.append(
            {
                "kind": ref.kind,
                "file": ref.path.name,
                "content_hash": ref.content_hash,
                "producer": ref.producer,
                "metadata": ref.metadata,
            }
        )
    cache_payload = {"items": items, "created_at": time.time()}
    (cache_dir / "cache.json").write_text(
        json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
