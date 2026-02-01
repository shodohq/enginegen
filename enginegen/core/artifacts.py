from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .plugin_api import ArtifactBundle, ArtifactRef

MANIFEST_VERSION = "1.0.0"


@dataclass
class ArtifactRecord:
    kind: str
    path: str
    content_hash: str
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    created_at: Optional[str] = None
    producer: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "content_hash": self.content_hash,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.producer is not None:
            payload["producer"] = self.producer
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class ArtifactManifest:
    manifest_version: str
    run_id: str
    created_at: str
    status: str
    host: Dict[str, Any]
    inputs: Dict[str, Any]
    plugins: List[Dict[str, Any]]
    artifacts: List[ArtifactRecord]
    metrics: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    extensions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "status": self.status,
            "host": self.host,
            "inputs": self.inputs,
            "plugins": self.plugins,
            "artifacts": [record.to_dict() for record in self.artifacts],
        }
        if self.metrics is not None:
            payload["metrics"] = self.metrics
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        if self.environment is not None:
            payload["environment"] = self.environment
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.extensions is not None:
            payload["extensions"] = self.extensions
        return payload


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create_run(self, run_id: str) -> Path:
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "artifacts").mkdir()
        (run_dir / "logs").mkdir()
        (run_dir / "cache").mkdir()
        return run_dir

    def write_inputs(self, run_dir: Path, inputs: Dict[str, Any]) -> None:
        for name, payload in inputs.items():
            target = run_dir / name
            if isinstance(payload, (dict, list)):
                target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            else:
                target.write_text(str(payload), encoding="utf-8")

    def add_artifacts(self, run_dir: Path, bundle: ArtifactBundle) -> None:
        artifacts_dir = run_dir / "artifacts"
        for ref in bundle.items:
            if not ref.path.is_absolute():
                ref.path = (artifacts_dir / ref.path).resolve()

    def build_manifest(
        self,
        run_id: str,
        status: str,
        host: Dict[str, Any],
        inputs: Dict[str, Any],
        plugins: List[Dict[str, Any]],
        bundle: ArtifactBundle,
        diagnostics: Dict[str, Any] | None = None,
        environment: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        notes: str | None = None,
        extensions: Dict[str, Any] | None = None,
    ) -> ArtifactManifest:
        run_dir = self.root / "runs" / run_id
        producer_index = {
            record.get("name"): record for record in plugins if isinstance(record, dict)
        }
        records: List[ArtifactRecord] = []
        for ref in bundle.items:
            content_hash = ref.content_hash or hash_file(ref.path)
            producer_ref = None
            if ref.producer and ref.producer in producer_index:
                record = producer_index[ref.producer]
                producer_ref = {
                    "plugin_kind": record.get("plugin_kind"),
                    "name": record.get("name"),
                }
                if record.get("plugin_version"):
                    producer_ref["plugin_version"] = record.get("plugin_version")
            path = ref.path
            try:
                path = ref.path.resolve()
                path = path.relative_to(run_dir.resolve())
            except Exception:
                path = ref.path
            size_bytes = None
            created_at = None
            try:
                stat = ref.path.stat()
                size_bytes = stat.st_size
                created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                pass
            records.append(
                ArtifactRecord(
                    kind=ref.kind,
                    path=str(path),
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    created_at=created_at,
                    producer=producer_ref,
                    metadata=ref.metadata,
                )
            )
        return ArtifactManifest(
            manifest_version=MANIFEST_VERSION,
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            host=host,
            inputs=inputs,
            plugins=plugins,
            artifacts=records,
            diagnostics=diagnostics,
            environment=environment,
            metrics=metrics,
            notes=notes,
            extensions=extensions,
        )

    def write_manifest(self, run_dir: Path, manifest: ArtifactManifest) -> Path:
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pid = os.getpid()
    nonce = secrets.token_hex(4)
    return f"{prefix}-{ts}-{pid}-{time.time_ns()}-{nonce}"


def collect_environment() -> Dict[str, Any]:
    return {
        "os": {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "platform": platform.platform(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "container": _detect_container(),
    }


def _detect_container() -> Dict[str, Any]:
    info: Dict[str, Any] = {"detected": False}
    cgroup_path = Path("/proc/1/cgroup")
    if not cgroup_path.exists():
        return info
    try:
        content = cgroup_path.read_text(encoding="utf-8")
    except Exception:
        return info
    info["detected"] = "docker" in content or "kubepods" in content or "containerd" in content
    info["raw"] = content.strip().splitlines()[:5]
    info["id"] = _extract_container_id(content)
    return info


def _extract_container_id(content: str) -> Optional[str]:
    for token in content.replace("/", " ").split():
        if len(token) >= 12 and all(c in "0123456789abcdef" for c in token.lower()):
            if len(token) >= 32:
                return token
    return None
