from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str, logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / f"{name}.log")
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def get_event_logger(logs_dir: Path) -> EventLogger:
    return EventLogger(logs_dir / "events.jsonl")
