from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from enginegen.core.plugin_api import AnalysisPlugin, ArtifactBundle, ArtifactRef, PluginMeta


class ScalarMetrics(AnalysisPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="scalar_metrics",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "analysis",
                "io": {
                    "inputs": [
                        {"name": "cad.stl", "category": "artifact", "artifact_kind": "cad.stl"},
                    ],
                    "outputs": [
                        {"name": "results.json", "category": "artifact", "artifact_kind": "results.json"},
                        {"name": "bbox_min_x", "category": "metric", "metric": "bbox_min_x"},
                        {"name": "bbox_min_y", "category": "metric", "metric": "bbox_min_y"},
                        {"name": "bbox_min_z", "category": "metric", "metric": "bbox_min_z"},
                        {"name": "bbox_max_x", "category": "metric", "metric": "bbox_max_x"},
                        {"name": "bbox_max_y", "category": "metric", "metric": "bbox_max_y"},
                        {"name": "bbox_max_z", "category": "metric", "metric": "bbox_max_z"},
                        {"name": "bbox_dx", "category": "metric", "metric": "bbox_dx"},
                        {"name": "bbox_dy", "category": "metric", "metric": "bbox_dy"},
                        {"name": "bbox_dz", "category": "metric", "metric": "bbox_dz"},
                        {"name": "bbox_volume", "category": "metric", "metric": "bbox_volume"},
                    ],
                },
                "features": {
                    "metrics": [
                        "bbox_min_x",
                        "bbox_min_y",
                        "bbox_min_z",
                        "bbox_max_x",
                        "bbox_max_y",
                        "bbox_max_z",
                        "bbox_dx",
                        "bbox_dy",
                        "bbox_dz",
                        "bbox_volume",
                    ],
                },
            },
        )

    def evaluate(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        bbox = None
        for ref in artifacts.by_kind("cad.stl"):
            bbox = _bbox_from_stl(ref.path)
            break

        metrics: Dict[str, Any] = {}
        if bbox is not None:
            mins = bbox["min"]
            maxs = bbox["max"]
            dx = maxs[0] - mins[0]
            dy = maxs[1] - mins[1]
            dz = maxs[2] - mins[2]
            metrics = {
                "bbox_min_x": mins[0],
                "bbox_min_y": mins[1],
                "bbox_min_z": mins[2],
                "bbox_max_x": maxs[0],
                "bbox_max_y": maxs[1],
                "bbox_max_z": maxs[2],
                "bbox_dx": dx,
                "bbox_dy": dy,
                "bbox_dz": dz,
                "bbox_volume": dx * dy * dz,
            }

        result = {"metrics": metrics}

        if metrics:
            if artifacts.metrics is None:
                artifacts.metrics = {}
            artifacts.metrics.update(metrics)

        results_path = ctx.run_dir / "artifacts" / "results.json"
        results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

        artifacts.add(
            ArtifactRef(
                kind="results.json",
                path=results_path,
                producer=self.meta().name,
                metadata={"metrics": list(result["metrics"].keys())},
            )
        )
        return artifacts


def _bbox_from_stl(path: Path):
    mins = [None, None, None]
    maxs = [None, None, None]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("vertex"):
            continue
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
        for i in range(3):
            mins[i] = coords[i] if mins[i] is None else min(mins[i], coords[i])
            maxs[i] = coords[i] if maxs[i] is None else max(maxs[i], coords[i])
    if mins[0] is None:
        return None
    return {"min": mins, "max": maxs}
