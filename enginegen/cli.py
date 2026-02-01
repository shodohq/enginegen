from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.config import collect_libs, load_case_config, normalize_case_config
from .core.pipeline import run_pipeline
from .core.plugin_loader import PluginRegistry
from .core.spec import load_spec, normalize_spec, validate_spec
from .core.graph import load_graph, validate_graph, validate_graph_schema
from .core.diagnostics import Diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(prog="enginegen")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("-c", "--config", required=True)
    validate.add_argument("--spec", required=False)
    validate.add_argument("--graph", required=False)

    build = sub.add_parser("build")
    build.add_argument("-c", "--config", required=True)
    build.add_argument("--spec", required=False)
    build.add_argument("--graph", required=False)

    run = sub.add_parser("run")
    run.add_argument("-c", "--config", required=True)
    run.add_argument("--spec", required=False)
    run.add_argument("--graph", required=False)

    report = sub.add_parser("report")
    report.add_argument("-r", "--run", required=True)

    args = parser.parse_args()

    if args.command == "report":
        _report(Path(args.run))
        return

    config_path = Path(args.config)
    config_data = load_case_config(config_path)
    case_config, spec_path, graph_path = _split_case_config(
        config_data, args.spec, args.graph, config_path.parent
    )

    spec = load_spec(spec_path)
    graph = load_graph(graph_path)

    schema_dir = Path(__file__).resolve().parent.parent / "schemas"

    if args.command == "validate":
        diagnostics = Diagnostics()
        diagnostics.extend(validate_spec(spec, schema_dir / "engine_spec.schema.json"))
        diagnostics.extend(validate_graph_schema(graph, schema_dir / "graph.schema.json"))
        diagnostics.extend(validate_graph(graph))
        _, spec_diags = normalize_spec(spec)
        diagnostics.extend(spec_diags)
        if diagnostics.has_errors():
            print(json.dumps(diagnostics.to_list(), indent=2, sort_keys=True))
            raise SystemExit(1)
        print(json.dumps({"status": "ok"}))
        return

    registry = PluginRegistry()
    registry.discover(collect_libs(case_config))

    if args.command == "build":
        run_pipeline(_strip_pipeline(case_config), spec, graph, registry, Path.cwd(), schema_dir)
        return

    if args.command == "run":
        run_pipeline(case_config, spec, graph, registry, Path.cwd(), schema_dir)
        return


def _split_case_config(config_data, spec_arg, graph_arg, base_dir: Path):
    if "engineCase" in config_data:
        case_config = config_data["engineCase"]
    else:
        case_config = config_data

    spec_path = spec_arg or case_config.get("spec") or case_config.get("spec_path")
    graph_path = graph_arg or case_config.get("graph") or case_config.get("graph_path")

    if not spec_path:
        raise ValueError("Spec path is required (config spec= or --spec)")
    if not graph_path:
        raise ValueError("Graph path is required (config graph= or --graph)")

    case_config = normalize_case_config(case_config)

    spec_path = Path(spec_path)
    graph_path = Path(graph_path)

    if not spec_path.is_absolute():
        spec_path = (base_dir / spec_path).resolve()
    if not graph_path.is_absolute():
        graph_path = (base_dir / graph_path).resolve()

    return case_config, spec_path, graph_path


def _strip_pipeline(case_config):
    cfg = json.loads(json.dumps(case_config))
    cfg["pipeline"] = []
    cfg["analysis"] = []
    cfg.pop("optimization", None)
    return cfg


def _report(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
