from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .core.config import collect_libs, normalize_case_config
from .core.pipeline import execute_pipeline
from .core.plugin_api import ArtifactBundle
from .core.plugin_loader import PluginRegistry


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    artifacts: ArtifactBundle


def compile(
    spec: Dict[str, Any],
    graph: Dict[str, Any],
    case_config: Dict[str, Any],
    *,
    root: Union[Path, str, None] = None,
    schema_dir: Union[Path, str, None] = None,
    run_id: Optional[str] = None,
) -> RunResult:
    cfg = normalize_case_config(case_config)
    cfg = _strip_pipeline(cfg)
    return _execute(cfg, spec, graph, root=root, schema_dir=schema_dir, run_id=run_id)


def build(
    spec: Dict[str, Any],
    graph: Dict[str, Any],
    case_config: Dict[str, Any],
    *,
    root: Union[Path, str, None] = None,
    schema_dir: Union[Path, str, None] = None,
    run_id: Optional[str] = None,
) -> RunResult:
    return compile(spec, graph, case_config, root=root, schema_dir=schema_dir, run_id=run_id)


def run(
    spec: Dict[str, Any],
    graph: Dict[str, Any],
    case_config: Dict[str, Any],
    *,
    root: Union[Path, str, None] = None,
    schema_dir: Union[Path, str, None] = None,
    run_id: Optional[str] = None,
) -> RunResult:
    cfg = normalize_case_config(case_config)
    return _execute(cfg, spec, graph, root=root, schema_dir=schema_dir, run_id=run_id)


def _execute(
    case_config: Dict[str, Any],
    spec: Dict[str, Any],
    graph: Dict[str, Any],
    *,
    root: Union[Path, str, None],
    schema_dir: Union[Path, str, None],
    run_id: Optional[str],
) -> RunResult:
    root_path = Path(root) if root is not None else Path.cwd()
    schema_path = Path(schema_dir) if schema_dir is not None else _default_schema_dir()
    registry = PluginRegistry()
    registry.discover(collect_libs(case_config))
    run_dir, artifacts = execute_pipeline(
        case_config,
        spec,
        graph,
        registry,
        root_path,
        schema_path,
        run_id=run_id,
    )
    return RunResult(run_dir=run_dir, artifacts=artifacts)


def _strip_pipeline(case_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json_clone(case_config)
    cfg["pipeline"] = []
    cfg["analysis"] = []
    cfg.pop("optimization", None)
    return cfg


def _default_schema_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "schemas"


def json_clone(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload))
