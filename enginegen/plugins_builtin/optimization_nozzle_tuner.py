from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict

from enginegen.core.plugin_api import ArtifactBundle, ArtifactRef, OptimizationPlugin, PluginMeta


class NozzleTuner(OptimizationPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="nozzle_tuner",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "optimization",
                "io": {
                    "inputs": [
                        {
                            "name": "openfoam.results",
                            "category": "artifact",
                            "artifact_kind": "cfd.openfoam.results",
                            "optional": True,
                        },
                        {
                            "name": "thrust_n",
                            "category": "metric",
                            "metric": "thrust_n",
                            "optional": True,
                        },
                    ],
                    "outputs": [
                        {
                            "name": "spec.tuned.json",
                            "category": "artifact",
                            "artifact_kind": "spec.tuned.json",
                        },
                        {
                            "name": "tuning.summary",
                            "category": "artifact",
                            "artifact_kind": "tuning.summary.json",
                        },
                        {
                            "name": "case.tuned.yaml",
                            "category": "artifact",
                            "artifact_kind": "case.tuned.yaml",
                            "optional": True,
                        },
                    ],
                },
                "features": {"tuning": ["exit_radius", "bell_length"]},
            },
        )

    def optimize(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        metrics = dict(artifacts.metrics or {})
        metrics.update(_metrics_from_results(artifacts))

        target_thrust = _target_value(spec, "thrust")
        measured_thrust = _as_float(metrics.get("thrust_n"))

        spec_out = json.loads(json.dumps(spec))
        geom = _get_geometry(spec_out)
        nozzle = geom.setdefault("nozzle", {})

        exit_radius = _as_float(nozzle.get("exit_radius")) or _fallback_exit_radius(nozzle)
        bell_length = _as_float(nozzle.get("bell_length")) or None

        ratio = None
        if target_thrust and measured_thrust and measured_thrust > 0:
            ratio = max(0.7, min(1.3, target_thrust / measured_thrust))

        updated = False
        if ratio is not None:
            exit_radius_new = exit_radius * sqrt(ratio)
            nozzle["exit_radius"] = exit_radius_new
            if bell_length:
                bell_length_new = bell_length * max(0.85, min(1.15, ratio))
                nozzle["bell_length"] = bell_length_new
            updated = True

        summary = {
            "status": "updated" if updated else "skipped",
            "target_thrust_n": target_thrust,
            "measured_thrust_n": measured_thrust,
            "ratio": ratio,
            "before": {"exit_radius": exit_radius, "bell_length": bell_length},
            "after": {
                "exit_radius": nozzle.get("exit_radius", exit_radius),
                "bell_length": nozzle.get("bell_length", bell_length),
            },
        }

        tuned_path = ctx.run_dir / "artifacts" / "spec.tuned.json"
        tuned_path.write_text(json.dumps(spec_out, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.add(
            ArtifactRef(
                kind="spec.tuned.json",
                path=tuned_path,
                producer=self.meta().name,
                metadata={"updated": updated},
            )
        )

        summary_path = ctx.run_dir / "artifacts" / "tuning.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.add(
            ArtifactRef(
                kind="tuning.summary.json",
                path=summary_path,
                producer=self.meta().name,
            )
        )

        case_path = ctx.run_dir / "artifacts" / "case.tuned.yaml"
        case_config = _load_case_config(ctx)
        if case_config:
            case_payload = {
                "spec": "spec.tuned.json",
                "graph": "../graph.json",
                "engineCase": case_config,
            }
            _write_yaml(case_path, case_payload)
            artifacts.add(
                ArtifactRef(
                    kind="case.tuned.yaml",
                    path=case_path,
                    producer=self.meta().name,
                )
            )
        return artifacts


def _metrics_from_results(artifacts: ArtifactBundle) -> Dict[str, Any]:
    refs = artifacts.by_kind("cfd.openfoam.results")
    if not refs:
        return {}
    try:
        payload = json.loads(refs[0].path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


def _load_case_config(ctx) -> Dict[str, Any]:
    config_path = ctx.run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _target_value(spec: Dict[str, Any], key: str) -> float | None:
    targets = ((spec.get("requirements") or {}).get("targets") or {})
    target = targets.get(key, {})
    value = target.get("target") if isinstance(target, dict) else None
    if isinstance(value, dict) and "value" in value:
        return _as_float(value.get("value"))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get_geometry(spec: Dict[str, Any]) -> Dict[str, Any]:
    geom = (
        spec.get("geometry")
        or (spec.get("metadata") or {}).get("geometry")
        or (spec.get("extensions") or {}).get("geometry")
        or {}
    )
    if "metadata" in spec:
        spec["metadata"]["geometry"] = geom
    else:
        spec.setdefault("metadata", {})["geometry"] = geom
    return geom


def _fallback_exit_radius(nozzle: Dict[str, Any]) -> float:
    throat = _as_float(nozzle.get("throat_radius")) or 0.04
    return throat * 4.5


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None
