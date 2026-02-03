from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from enginegen.core.external import SandboxConfig, run_external
from enginegen.core.plugin_api import AdapterPlugin, ArtifactBundle, ArtifactRef, PluginMeta


class OpenFoamAdapter(AdapterPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="openfoam_cfd",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "adapter",
                "io": {
                    "inputs": [
                        {
                            "name": "geometry.mesh.stl",
                            "category": "artifact",
                            "artifact_kind": "geometry.mesh.stl",
                            "optional": True,
                        },
                        {
                            "name": "cad.stl",
                            "category": "artifact",
                            "artifact_kind": "cad.stl",
                            "optional": True,
                        },
                    ],
                    "outputs": [
                        {
                            "name": "openfoam.case",
                            "category": "artifact",
                            "artifact_kind": "cfd.openfoam.case",
                        },
                        {
                            "name": "openfoam.metrics",
                            "category": "artifact",
                            "artifact_kind": "cfd.openfoam.metrics",
                        },
                    ],
                },
                "features": {
                    "modes": ["mock", "run"],
                    "notes": "Generates an OpenFOAM case and runs commands (or mock).",
                },
            },
        )

    def run(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        cfg = _load_step_config(ctx, self.meta().name)
        mode = str(cfg.get("mode", "mock")).lower()
        case_dir = ctx.run_dir / "openfoam_case"
        case_dir.mkdir(parents=True, exist_ok=True)

        case_template = cfg.get("case_template")
        if case_template:
            _copy_case_template(case_dir, case_template)
        else:
            _ensure_stub_case(case_dir)

        mesh_path = _find_mesh(artifacts)
        if mesh_path is None:
            raise RuntimeError("openfoam_cfd requires an STL mesh artifact")
        tri_surface = case_dir / "constant" / "triSurface"
        tri_surface.mkdir(parents=True, exist_ok=True)
        stl_name = str(cfg.get("stl_name") or "nozzle.stl")
        stl_target = tri_surface / stl_name
        if mesh_path.resolve() != stl_target.resolve():
            shutil.copyfile(mesh_path, stl_target)

        if mode == "run":
            commands = _normalize_commands(cfg.get("commands"))
            if not commands:
                raise RuntimeError(
                    "openfoam_cfd mode=run requires commands (e.g. blockMesh, simpleFoam)"
                )
            shell = cfg.get("shell")
            env = cfg.get("env") or {}
            sandbox_cfg = _sandbox_from_config(cfg, ctx, mode)
            for idx, cmd in enumerate(commands):
                run_cmd = _wrap_command(cmd, shell)
                result = run_external(
                    run_cmd,
                    cwd=case_dir,
                    logs_dir=ctx.logs_dir,
                    name=f"openfoam_{idx}",
                    env=env,
                    sandbox=sandbox_cfg,
                )
                self.record_external_run(
                    result,
                    ctx,
                    artifacts,
                    name=f"openfoam_{idx}",
                    tool_version=str(cfg.get("solver") or "openfoam"),
                )
            metrics = _load_openfoam_metrics(case_dir)
        else:
            result = run_external(
                [sys.executable, "-c", "print('openfoam mock')"],
                cwd=ctx.run_dir,
                logs_dir=ctx.logs_dir,
                name="openfoam_mock",
                sandbox=_sandbox_from_config(cfg, ctx, mode),
            )
            self.record_external_run(
                result, ctx, artifacts, name="openfoam_mock", tool_version="mock"
            )
            metrics = _mock_metrics(spec)

        metrics_path = ctx.run_dir / "artifacts" / "openfoam.metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        artifacts.add(
            ArtifactRef(
                kind="cfd.openfoam.metrics",
                path=metrics_path,
                producer=self.meta().name,
                metadata={"mode": metrics.get("mode")},
            )
        )
        artifacts.add(
            ArtifactRef(
                kind="cfd.openfoam.case",
                path=case_dir,
                producer=self.meta().name,
                metadata={"stl": stl_name, "mode": mode},
            )
        )
        return artifacts


def _load_step_config(ctx, plugin_name: str) -> Dict[str, Any]:
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
    if not isinstance(data, dict):
        return {}
    for step in data.get("pipeline", []) or []:
        if isinstance(step, dict) and step.get("type") == plugin_name:
            return step
    return {}


def _copy_case_template(case_dir: Path, template_path: str) -> None:
    src = Path(template_path).expanduser()
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"OpenFOAM case template not found: {src}")
    shutil.copytree(src, case_dir, dirs_exist_ok=True)


def _ensure_stub_case(case_dir: Path) -> None:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    readme = case_dir / "README.txt"
    if not readme.exists():
        readme.write_text(
            "OpenFOAM case stub. Provide system/ and constant/ dictionaries for a real run.\n",
            encoding="utf-8",
        )


def _find_mesh(artifacts: ArtifactBundle) -> Optional[Path]:
    for kind in ("geometry.mesh.stl", "cad.stl"):
        refs = artifacts.by_kind(kind)
        if refs:
            return refs[0].path
    return None


def _normalize_commands(value: Any) -> List[List[str]]:
    if not value:
        return []
    commands: List[List[str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)):
                commands.append([str(x) for x in item])
            elif isinstance(item, str):
                commands.append([item])
    elif isinstance(value, str):
        commands.append([value])
    return commands


def _wrap_command(cmd: List[str], shell: Optional[str]) -> List[str]:
    if shell:
        import shlex

        shell_parts = shlex.split(str(shell))
        cmd_str = shlex.join(cmd)
        return shell_parts + [cmd_str]
    if len(cmd) == 1 and " " in cmd[0]:
        import shlex

        return shlex.split(cmd[0])
    return cmd


def _sandbox_from_config(cfg: Dict[str, Any], ctx, mode: str) -> Optional[SandboxConfig]:
    sandbox_cfg = cfg.get("sandbox")
    if sandbox_cfg is None:
        return None
    if isinstance(sandbox_cfg, bool):
        enabled = sandbox_cfg
    else:
        enabled = True
    if not enabled:
        return None
    extra_roots: List[Path] = []
    if isinstance(sandbox_cfg, dict):
        for entry in sandbox_cfg.get("extra_roots", []) or []:
            extra_roots.append(Path(entry))
    return SandboxConfig(
        enabled=True,
        wrapper=f"{sys.executable} -m enginegen.core.sandbox_wrapper",
        env_mode="clean",
        allowed_root=ctx.run_dir,
        policy=str((sandbox_cfg or {}).get("policy", "strict")),
        deny_network=bool((sandbox_cfg or {}).get("deny_network", True)),
        require_fs_isolation=bool((sandbox_cfg or {}).get("require_fs_isolation", True)),
        fs_mode=str((sandbox_cfg or {}).get("fs_mode", "auto")),
        extra_roots=extra_roots,
    )


def _mock_metrics(spec: Dict[str, Any]) -> Dict[str, Any]:
    geom = _get_geometry(spec)
    nozzle = geom.get("nozzle", {}) or {}
    throat_radius = _as_float(nozzle.get("throat_radius"), 0.04)
    exit_radius = _as_float(nozzle.get("exit_radius"), throat_radius * 4.5)
    expansion_ratio = (exit_radius / max(throat_radius, 1e-6)) ** 2
    chamber_pressure = _target_value(spec, "chamber_pressure", 5.0e6)

    exit_area = math.pi * exit_radius * exit_radius
    thrust_coeff = 1.35 + min(expansion_ratio, 10.0) * 0.02
    thrust = chamber_pressure * exit_area * thrust_coeff

    isp = 240.0 + min(expansion_ratio, 12.0) * 2.5
    mass_flow = thrust / (isp * 9.80665)
    exit_pressure = chamber_pressure * max(0.04, 0.18 / max(expansion_ratio, 1.0))

    return {
        "mode": "mock",
        "metrics": {
            "thrust_n": thrust,
            "chamber_pressure_pa": chamber_pressure,
            "exit_pressure_pa": exit_pressure,
            "mass_flow_kg_s": mass_flow,
            "isp_s": isp,
            "expansion_ratio": expansion_ratio,
        },
        "notes": {"source": "synthetic"},
    }


def _load_openfoam_metrics(case_dir: Path) -> Dict[str, Any]:
    metrics_path = case_dir / "openfoam.metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "metrics" in data:
                data.setdefault("mode", "run")
                return data
        except Exception:
            pass
    return {"mode": "run", "metrics": {}, "notes": {"status": "missing"}}


def _get_geometry(spec: Dict[str, Any]) -> Dict[str, Any]:
    return (
        spec.get("geometry")
        or (spec.get("metadata") or {}).get("geometry")
        or (spec.get("extensions") or {}).get("geometry")
        or {}
    )


def _target_value(spec: Dict[str, Any], key: str, default: float) -> float:
    targets = ((spec.get("requirements") or {}).get("targets") or {})
    target = targets.get(key, {})
    value = target.get("target") if isinstance(target, dict) else None
    if isinstance(value, dict) and "value" in value:
        return _as_float(value.get("value"), default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default
