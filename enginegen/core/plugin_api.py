from __future__ import annotations

from abc import ABC, abstractmethod
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .external import ExternalRunResult


API_VERSION = "1.0.0"


@dataclass(frozen=True)
class PluginMeta:
    name: str
    api_version: str
    plugin_version: str
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRef:
    kind: str
    path: Path
    content_hash: Optional[str] = None
    producer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactBundle:
    items: List[ArtifactRef] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    notes: Optional[Dict[str, Any]] = None
    external_runs: List["ExternalRunRecord"] = field(default_factory=list)

    def add(self, ref: ArtifactRef) -> None:
        self.items.append(ref)

    def extend(self, refs: Iterable[ArtifactRef]) -> None:
        self.items.extend(refs)

    def by_kind(self, kind: str) -> List[ArtifactRef]:
        return [item for item in self.items if item.kind == kind]

    def add_external_run(self, run: "ExternalRunRecord") -> None:
        self.external_runs.append(run)


@dataclass
class HostContext:
    run_id: str
    run_dir: Path
    artifact_dir: Path
    cache_dir: Path
    logs_dir: Path
    env: Dict[str, str] = field(default_factory=dict)
    export_formats: Optional[List[str]] = None
    seed: Optional[int] = None
    nondeterminism: Dict[str, Any] = field(default_factory=dict)
    audit: Optional["AuditLogger"] = None
    access: Optional["AccessController"] = None


class AuditLogger(Protocol):
    def record(self, event: Dict[str, Any]) -> None: ...


class AccessController(Protocol):
    def check(self, action: str, context: Dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ExternalRunRecord:
    name: str
    cmd: List[str]
    returncode: int
    stdout: Path
    stderr: Path
    elapsed_s: float
    tool_version: Optional[str] = None
    sandboxed: bool = False
    sandbox_wrapper: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cmd": self.cmd,
            "returncode": self.returncode,
            "stdout": str(self.stdout),
            "stderr": str(self.stderr),
            "elapsed_s": self.elapsed_s,
            "tool_version": self.tool_version,
            "sandboxed": self.sandboxed,
            "sandbox_wrapper": self.sandbox_wrapper,
        }


class Plugin(ABC):
    @abstractmethod
    def meta(self) -> PluginMeta:
        raise NotImplementedError


class SynthesizerPlugin(Plugin):
    @abstractmethod
    def synthesize(
        self,
        spec: Dict[str, Any],
        graph: Dict[str, Any],
        ctx: HostContext,
        feedback: Optional["ArtifactBundle"] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class GeometryBackendPlugin(Plugin):
    @abstractmethod
    def compile(
        self, ir: Dict[str, Any], ctx: HostContext, *, export_formats: Optional[List[str]] = None
    ) -> ArtifactBundle:
        raise NotImplementedError

    def validate(self, ir: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        return artifacts


class AdapterPlugin(Plugin):
    @abstractmethod
    def run(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        raise NotImplementedError

    def record_external_run(
        self,
        result: ExternalRunResult,
        ctx: HostContext,
        bundle: ArtifactBundle,
        *,
        name: str,
        tool_version: Optional[str] = None,
    ) -> None:
        if not tool_version:
            raise ValueError(f"tool_version is required for external run {name}")
        summary = {
            "name": name,
            "cmd": result.cmd,
            "returncode": result.returncode,
            "stdout": str(result.stdout),
            "stderr": str(result.stderr),
            "elapsed_s": result.elapsed_s,
            "tool_version": tool_version,
            "sandboxed": result.sandboxed,
            "sandbox_wrapper": result.sandbox_wrapper,
        }
        summary_path = ctx.logs_dir / f"{name}.run.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        producer = self.meta().name
        bundle.add(
            ArtifactRef(
                kind="log.external.run",
                path=summary_path,
                producer=producer,
                metadata=summary,
            )
        )
        bundle.add(
            ArtifactRef(
                kind="log.external.stdout",
                path=result.stdout,
                producer=producer,
                metadata={"name": name},
            )
        )
        bundle.add(
            ArtifactRef(
                kind="log.external.stderr",
                path=result.stderr,
                producer=producer,
                metadata={"name": name},
            )
        )
        bundle.add_external_run(
            ExternalRunRecord(
                name=name,
                cmd=result.cmd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_s=result.elapsed_s,
                tool_version=tool_version,
                sandboxed=result.sandboxed,
                sandbox_wrapper=result.sandbox_wrapper,
            )
        )


class AnalysisPlugin(Plugin):
    @abstractmethod
    def evaluate(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        raise NotImplementedError


class OptimizationPlugin(Plugin):
    @abstractmethod
    def optimize(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        raise NotImplementedError
