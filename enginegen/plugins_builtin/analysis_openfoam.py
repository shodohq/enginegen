from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from enginegen.core.plugin_api import AnalysisPlugin, ArtifactBundle, ArtifactRef, PluginMeta


class OpenFoamMetrics(AnalysisPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="openfoam_metrics",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "analysis",
                "io": {
                    "inputs": [
                        {
                            "name": "openfoam.metrics",
                            "category": "artifact",
                            "artifact_kind": "cfd.openfoam.metrics",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "openfoam.results",
                            "category": "artifact",
                            "artifact_kind": "cfd.openfoam.results",
                        },
                        {"name": "thrust_n", "category": "metric", "metric": "thrust_n"},
                        {
                            "name": "chamber_pressure_pa",
                            "category": "metric",
                            "metric": "chamber_pressure_pa",
                        },
                        {
                            "name": "exit_pressure_pa",
                            "category": "metric",
                            "metric": "exit_pressure_pa",
                        },
                        {
                            "name": "mass_flow_kg_s",
                            "category": "metric",
                            "metric": "mass_flow_kg_s",
                        },
                        {"name": "isp_s", "category": "metric", "metric": "isp_s"},
                        {
                            "name": "expansion_ratio",
                            "category": "metric",
                            "metric": "expansion_ratio",
                        },
                    ],
                },
                "features": {
                    "metrics": [
                        "thrust_n",
                        "chamber_pressure_pa",
                        "exit_pressure_pa",
                        "mass_flow_kg_s",
                        "isp_s",
                        "expansion_ratio",
                    ]
                },
            },
        )

    def evaluate(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        metrics_path = _find_metrics_path(artifacts)
        data: Dict[str, Any] = {}
        if metrics_path:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))

        metrics = data.get("metrics") if isinstance(data, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}

        if metrics:
            if artifacts.metrics is None:
                artifacts.metrics = {}
            artifacts.metrics.update(metrics)

        results_path = ctx.run_dir / "artifacts" / "openfoam.results.json"
        results = {
            "mode": data.get("mode") if isinstance(data, dict) else None,
            "metrics": metrics,
            "notes": data.get("notes") if isinstance(data, dict) else {},
        }
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

        artifacts.add(
            ArtifactRef(
                kind="cfd.openfoam.results",
                path=results_path,
                producer=self.meta().name,
                metadata={"metrics": list(metrics.keys())},
            )
        )
        return artifacts


def _find_metrics_path(artifacts: ArtifactBundle) -> Path | None:
    refs = artifacts.by_kind("cfd.openfoam.metrics")
    if not refs:
        return None
    return refs[0].path
