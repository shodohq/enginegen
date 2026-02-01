from __future__ import annotations

import subprocess
import time
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class ExternalRunResult:
    cmd: list[str]
    returncode: int
    stdout: Path
    stderr: Path
    elapsed_s: float
    sandboxed: bool
    sandbox_wrapper: Optional[str]


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool = True
    wrapper: Optional[str] = None
    env_mode: str = "inherit"  # "inherit" or "clean"
    allowed_root: Optional[Path] = None
    require_wrapper: bool = True
    policy: str = "strict"  # "strict" or "permissive"
    deny_network: bool = True
    require_fs_isolation: bool = True
    fs_mode: str = "auto"  # "auto", "bwrap", "none"
    extra_roots: Sequence[Path] = field(default_factory=tuple)


def run_external(
    cmd: Sequence[str],
    cwd: Path,
    logs_dir: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
    name: str = "external",
    sandbox: Optional[SandboxConfig] = SandboxConfig(),
) -> ExternalRunResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    start = time.time()
    sandbox_wrapper = None
    run_cmd = list(cmd)
    wrapper = None
    if sandbox and sandbox.enabled:
        wrapper = sandbox.wrapper or os.environ.get("ENGINEGEN_SANDBOX_CMD")
        if sandbox.require_wrapper and not wrapper:
            raise RuntimeError("Sandbox wrapper is required but not configured")
        if wrapper:
            sandbox_wrapper = wrapper
            run_cmd = shlex.split(wrapper) + run_cmd
        if sandbox.allowed_root is not None:
            try:
                cwd.resolve().relative_to(sandbox.allowed_root.resolve())
            except Exception as exc:
                raise ValueError(f"Sandbox cwd must be within {sandbox.allowed_root}") from exc

    sandboxed = bool(sandbox and sandbox.enabled and sandbox_wrapper)

    if env is None:
        if sandboxed and sandbox.env_mode == "clean":
            env = {"PATH": os.environ.get("PATH", "")}
        else:
            env = os.environ.copy()
    if sandbox and sandbox.enabled:
        env = dict(env) if env else {}
        env["ENGINEGEN_SANDBOX_ENV_MODE"] = sandbox.env_mode
        env["ENGINEGEN_SANDBOX_POLICY"] = sandbox.policy
        env["ENGINEGEN_SANDBOX_DENY_NETWORK"] = "1" if sandbox.deny_network else "0"
        env["ENGINEGEN_SANDBOX_REQUIRE_FS"] = "1" if sandbox.require_fs_isolation else "0"
        env["ENGINEGEN_SANDBOX_FS_MODE"] = sandbox.fs_mode
        if sandbox.allowed_root is not None:
            env["ENGINEGEN_SANDBOX_ALLOWED_ROOT"] = str(sandbox.allowed_root)
        if sandbox.extra_roots:
            env["ENGINEGEN_SANDBOX_EXTRA_ROOTS"] = ":".join(str(p) for p in sandbox.extra_roots)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        proc = subprocess.run(
            run_cmd,
            cwd=cwd,
            env=dict(env) if env else None,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    elapsed = time.time() - start
    return ExternalRunResult(
        cmd=run_cmd,
        returncode=proc.returncode,
        stdout=stdout_path,
        stderr=stderr_path,
        elapsed_s=elapsed,
        sandboxed=sandboxed,
        sandbox_wrapper=sandbox_wrapper,
    )
