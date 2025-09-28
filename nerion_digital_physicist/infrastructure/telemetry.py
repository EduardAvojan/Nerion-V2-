"""Telemetry utilities for generator metrics."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

TELEMETRY_FILENAME = "telemetry.jsonl"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TelemetryEvent:
    event_type: str
    payload: Dict[str, Any]
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class TelemetryLogger:
    def __init__(self, root: Path):
        self.root = root
        self.file_path = root / TELEMETRY_FILENAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_path.touch(exist_ok=True)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = TelemetryEvent(
            event_type=event_type,
            payload=payload,
            created_at=utc_timestamp(),
        )
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(event.to_json())
            fh.write("\n")
