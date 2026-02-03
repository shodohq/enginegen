from __future__ import annotations

import importlib
import inspect
from importlib import metadata
from typing import Any, Callable, Dict, Optional

from .plugin_api import API_VERSION, Plugin, PluginMeta
from .native_plugin import discover_native_plugins, is_native_lib


PLUGIN_GROUPS = {
    "synthesizer": "enginegen.synthesizer",
    "geometry_backend": "enginegen.geometry_backend",
    "analysis": "enginegen.analysis",
    "adapter": "enginegen.adapter",
    "optimization": "enginegen.optimization",
}


def _major(version: str) -> str:
    return version.split(".")[0]


def _iter_entry_points(group: str):
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))


class PluginRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {
            "synthesizer": {},
            "geometry_backend": {},
            "analysis": {},
            "adapter": {},
            "optimization": {},
        }

    def register(self, kind: str, name: str, provider: Any) -> None:
        self._registry.setdefault(kind, {})[name] = provider

    def discover(self, libs: Optional[list[str]] = None) -> None:
        if libs:
            for lib in libs:
                if is_native_lib(lib):
                    continue
                try:
                    importlib.import_module(lib)
                except ImportError:
                    continue

            for kind, name, provider in discover_native_plugins(libs):
                self.register(kind, name, provider)

        for kind, group in PLUGIN_GROUPS.items():
            for ep in _iter_entry_points(group):
                self.register(kind, ep.name, ep.load)

        # Fallback for editable/source usage without installed entry points.
        self._register_builtins()

    def get(self, kind: str, name: str) -> Plugin:
        if kind not in self._registry or name not in self._registry[kind]:
            raise KeyError(f"Plugin not found: {kind}:{name}")
        plugin = _instantiate_plugin(self._registry[kind][name])
        _check_compatibility(plugin.meta())
        return plugin

    def _register_builtins(self) -> None:
        try:
            from enginegen.plugins_builtin.synth_baseline import BaselineRuleSynth
            from enginegen.plugins_builtin.geometry_simple import SimpleStlBackend
            from enginegen.plugins_builtin.geometry_fidget import FidgetGeometryBackend
            from enginegen.plugins_builtin.analysis_scalar import ScalarMetrics
            from enginegen.plugins_builtin.adapter_noop import NoopAdapter

            if "baseline_rule" not in self._registry.get("synthesizer", {}):
                self.register("synthesizer", "baseline_rule", BaselineRuleSynth)
            if "simple_stl" not in self._registry.get("geometry_backend", {}):
                self.register("geometry_backend", "simple_stl", SimpleStlBackend)
            if "builtin.geometry.fidget" not in self._registry.get("geometry_backend", {}):
                self.register("geometry_backend", "builtin.geometry.fidget", FidgetGeometryBackend)
            if "scalar_metrics" not in self._registry.get("analysis", {}):
                self.register("analysis", "scalar_metrics", ScalarMetrics)
            if "noop" not in self._registry.get("adapter", {}):
                self.register("adapter", "noop", NoopAdapter)
        except Exception:
            # Built-ins are optional; entry points may already provide them.
            return


def _check_compatibility(meta: PluginMeta) -> None:
    if _major(meta.api_version) != _major(API_VERSION):
        raise RuntimeError(
            f"Plugin API version mismatch: host {API_VERSION} vs plugin {meta.api_version}"
        )


def _instantiate_plugin(provider: Any) -> Plugin:
    obj = provider
    if callable(obj):
        obj = obj()
    if inspect.isclass(obj):
        obj = obj()
    if not hasattr(obj, "meta"):
        raise RuntimeError(f"Loaded plugin does not implement meta(): {type(obj)}")
    return obj
