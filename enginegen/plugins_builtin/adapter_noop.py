from __future__ import annotations

import sys
from typing import Any, Dict

from enginegen.core.external import SandboxConfig, run_external
from enginegen.core.plugin_api import AdapterPlugin, ArtifactBundle, PluginMeta


class NoopAdapter(AdapterPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="noop",
            api_version="1.0.0",
            plugin_version="0.1.0",
            capabilities={
                "capabilities_version": "1.0.0",
                "plugin_kind": "adapter",
                "io": {"inputs": [], "outputs": []},
                "features": {"description": "No-op adapter"},
            },
        )

    def run(self, spec: Dict[str, Any], artifacts: ArtifactBundle, ctx) -> ArtifactBundle:
        result = run_external(
            [sys.executable, "-c", "print('noop')"],
            cwd=ctx.run_dir,
            logs_dir=ctx.logs_dir,
            name="noop",
            sandbox=SandboxConfig(
                enabled=True,
                wrapper=f"{sys.executable} -m enginegen.core.sandbox_wrapper",
                env_mode="clean",
                allowed_root=ctx.run_dir,
                policy="strict",
                deny_network=True,
                require_fs_isolation=True,
            ),
        )
        self.record_external_run(result, ctx, artifacts, name="noop", tool_version="noop")
        return artifacts
