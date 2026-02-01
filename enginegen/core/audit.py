from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class NullAuditLogger:
    def record(self, event: Dict[str, Any]) -> None:
        return


class FileAuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class AllowAllAccess:
    def check(self, action: str, context: Dict[str, Any]) -> None:
        return


def load_audit_logger(config: Any, logs_dir: Path) -> Any:
    if not config:
        return NullAuditLogger()
    if config is True:
        return FileAuditLogger(logs_dir / "audit.jsonl")
    if isinstance(config, dict):
        logger_type = config.get("type", "file")
        if logger_type == "file":
            path = config.get("path") or (logs_dir / "audit.jsonl")
            return FileAuditLogger(Path(path))
        if logger_type == "callable" and config.get("callable"):
            return _load_callable(config["callable"], logs_dir)
    if isinstance(config, str):
        return _load_callable(config, logs_dir)
    if hasattr(config, "record"):
        return config
    return NullAuditLogger()


def load_access_controller(config: Any) -> Any:
    if not config:
        return AllowAllAccess()
    if isinstance(config, dict):
        controller_type = config.get("type", "allow_all")
        if controller_type == "allow_all":
            return AllowAllAccess()
        if controller_type == "callable" and config.get("callable"):
            return _load_callable(config["callable"], None)
    if isinstance(config, str):
        return _load_callable(config, None)
    if hasattr(config, "check"):
        return config
    return AllowAllAccess()


def _load_callable(path: str, logs_dir: Optional[Path]) -> Any:
    module_name, _, attr = path.replace(":", ".").rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid callable path: {path}")
    module = importlib.import_module(module_name)
    target = getattr(module, attr)
    if callable(target):
        if logs_dir is not None:
            try:
                return target(logs_dir)
            except TypeError:
                pass
        try:
            return target()
        except TypeError:
            return target
    return target
