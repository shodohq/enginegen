from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .plugin_api import (
    AdapterPlugin,
    AnalysisPlugin,
    ArtifactBundle,
    ArtifactRef,
    ExternalRunRecord,
    GeometryBackendPlugin,
    HostContext,
    OptimizationPlugin,
    Plugin,
    PluginMeta,
    SynthesizerPlugin,
)


NATIVE_SUFFIXES = {".so", ".dylib", ".dll"}


def is_native_lib(value: str) -> bool:
    path = Path(value)
    return path.suffix.lower() in NATIVE_SUFFIXES


def discover_native_plugins(libs: Iterable[str]) -> List[Tuple[str, str, Callable[[], Plugin]]]:
    providers: List[Tuple[str, str, Callable[[], Plugin]]] = []
    for lib in libs:
        path = Path(lib)
        if not is_native_lib(lib):
            continue
        if not path.exists():
            continue
        descriptor = _NativeLib(path).descriptor
        kind = descriptor.get("kind")
        name = descriptor.get("name")
        if not kind or not name:
            raise RuntimeError(f"Native plugin {path} missing kind/name in descriptor")
        wrapper = _KIND_TO_WRAPPER.get(kind)
        if wrapper is None:
            raise RuntimeError(f"Native plugin {path} has unsupported kind: {kind}")

        def _provider(p: Path = path, cls: Callable[[Path], Plugin] = wrapper) -> Plugin:
            return cls(p)

        providers.append((kind, name, _provider))
    return providers


@dataclass(frozen=True)
class _NativeLib:
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "lib", ctypes.CDLL(str(self.path)))
        lib = self.lib
        lib.enginegen_plugin_descriptor.restype = ctypes.c_char_p
        lib.enginegen_plugin_call.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.enginegen_plugin_call.restype = ctypes.c_int
        lib.enginegen_plugin_free.argtypes = [ctypes.c_void_p]
        lib.enginegen_plugin_free.restype = None
        descriptor_raw = lib.enginegen_plugin_descriptor()
        if not descriptor_raw:
            raise RuntimeError(f"Native plugin {self.path} returned empty descriptor")
        descriptor = json.loads(descriptor_raw.decode("utf-8"))
        object.__setattr__(self, "descriptor", descriptor)

    def invoke(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        buffer = ctypes.create_string_buffer(data)
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        code = self.lib.enginegen_plugin_call(
            method.encode("utf-8"),
            ctypes.cast(buffer, ctypes.c_void_p),
            len(data),
            ctypes.byref(out_ptr),
            ctypes.byref(out_len),
        )
        out_bytes = b""
        if out_ptr.value:
            out_bytes = ctypes.string_at(out_ptr, out_len.value)
            self.lib.enginegen_plugin_free(out_ptr)
        if code != 0:
            message = out_bytes.decode("utf-8", errors="ignore") if out_bytes else ""
            raise RuntimeError(message or f"native plugin call failed ({method}, code={code})")
        if not out_bytes:
            return {}
        return json.loads(out_bytes.decode("utf-8"))


class _NativePluginBase(Plugin):
    def __init__(self, path: Path) -> None:
        self._lib = _NativeLib(path)
        desc = self._lib.descriptor
        self._meta = PluginMeta(
            name=desc.get("name", path.stem),
            api_version=desc.get("api_version", "0.0.0"),
            plugin_version=desc.get("plugin_version", "0.0.0"),
            capabilities=desc.get("capabilities", {}),
        )

    def meta(self) -> PluginMeta:
        return self._meta

    def _ctx_payload(self, ctx: HostContext) -> Dict[str, Any]:
        return {
            "run_id": ctx.run_id,
            "run_dir": str(ctx.run_dir),
            "artifact_dir": str(ctx.artifact_dir),
            "cache_dir": str(ctx.cache_dir),
            "logs_dir": str(ctx.logs_dir),
            "env": dict(ctx.env),
            "export_formats": ctx.export_formats,
            "seed": ctx.seed,
            "nondeterminism": dict(ctx.nondeterminism),
        }

    def _bundle_payload(self, bundle: ArtifactBundle) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "kind": ref.kind,
                    "path": str(ref.path),
                    "content_hash": ref.content_hash,
                    "producer": ref.producer,
                    "metadata": ref.metadata,
                }
                for ref in bundle.items
            ],
            "metrics": bundle.metrics,
            "notes": bundle.notes,
            "external_runs": [run.to_dict() for run in bundle.external_runs],
        }

    def _bundle_from_payload(self, payload: Dict[str, Any]) -> ArtifactBundle:
        bundle = ArtifactBundle()
        for item in payload.get("items", []) or []:
            bundle.add(
                ArtifactRef(
                    kind=item.get("kind", "unknown"),
                    path=Path(item.get("path", "")),
                    content_hash=item.get("content_hash"),
                    producer=item.get("producer"),
                    metadata=item.get("metadata") or {},
                )
            )
        bundle.metrics = payload.get("metrics")
        bundle.notes = payload.get("notes")
        for run in payload.get("external_runs", []) or []:
            try:
                bundle.add_external_run(
                    ExternalRunRecord(
                        name=run["name"],
                        cmd=run["cmd"],
                        returncode=int(run["returncode"]),
                        stdout=Path(run["stdout"]),
                        stderr=Path(run["stderr"]),
                        elapsed_s=float(run["elapsed_s"]),
                        tool_version=run.get("tool_version"),
                        sandboxed=bool(run.get("sandboxed")),
                        sandbox_wrapper=run.get("sandbox_wrapper"),
                    )
                )
            except Exception:
                continue
        return bundle


class NativeSynthesizer(_NativePluginBase, SynthesizerPlugin):
    def synthesize(
        self,
        spec: Dict[str, Any],
        graph: Dict[str, Any],
        ctx: HostContext,
        feedback: Optional[ArtifactBundle] = None,
    ) -> Dict[str, Any]:
        payload = {"spec": spec, "graph": graph, "ctx": self._ctx_payload(ctx)}
        if feedback is not None:
            payload["feedback"] = self._bundle_payload(feedback)
        result = self._lib.invoke("synthesize", payload)
        return result.get("ir", result)


class NativeGeometryBackend(_NativePluginBase, GeometryBackendPlugin):
    def compile(
        self, ir: Dict[str, Any], ctx: HostContext, *, export_formats: Optional[List[str]] = None
    ) -> ArtifactBundle:
        payload = {"ir": ir, "ctx": self._ctx_payload(ctx), "export_formats": export_formats}
        result = self._lib.invoke("compile", payload)
        return self._bundle_from_payload(result.get("artifacts", result))

    def validate(self, ir: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        payload = {
            "ir": ir,
            "artifacts": self._bundle_payload(artifacts),
            "ctx": self._ctx_payload(ctx),
        }
        result = self._lib.invoke("validate", payload)
        return self._bundle_from_payload(result.get("artifacts", result))


class NativeAdapter(_NativePluginBase, AdapterPlugin):
    def run(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        payload = {
            "spec": spec,
            "artifacts": self._bundle_payload(artifacts),
            "ctx": self._ctx_payload(ctx),
        }
        result = self._lib.invoke("run", payload)
        return self._bundle_from_payload(result.get("artifacts", result))


class NativeAnalysis(_NativePluginBase, AnalysisPlugin):
    def evaluate(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        payload = {
            "spec": spec,
            "artifacts": self._bundle_payload(artifacts),
            "ctx": self._ctx_payload(ctx),
        }
        result = self._lib.invoke("evaluate", payload)
        return self._bundle_from_payload(result.get("artifacts", result))


class NativeOptimization(_NativePluginBase, OptimizationPlugin):
    def optimize(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx: HostContext) -> ArtifactBundle:
        payload = {
            "spec": spec,
            "artifacts": self._bundle_payload(artifacts),
            "ctx": self._ctx_payload(ctx),
        }
        result = self._lib.invoke("optimize", payload)
        return self._bundle_from_payload(result.get("artifacts", result))


_KIND_TO_WRAPPER: Dict[str, Callable[[Path], Plugin]] = {
    "synthesizer": NativeSynthesizer,
    "geometry_backend": NativeGeometryBackend,
    "adapter": NativeAdapter,
    "analysis": NativeAnalysis,
    "optimization": NativeOptimization,
}
