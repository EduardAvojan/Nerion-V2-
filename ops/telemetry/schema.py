"""Telemetry event schema for the Phase 1 observer bus.

This module centralises the event shape used by Nerion's forthcoming
telemetry/analytics pipeline. It is intentionally independent of any
storage or transport concerns so that sinks (JSONL, SQL, vector
embeddings, etc.) can share the same contract.

Background sampling (Jan 2025):
  • `core.memory.journal.log_event` currently logs snapshots, plan/apply
    outcomes, plugin reloads, etc.  (callers across `selfcoder.*`,
    `plugins.loader`, `selfcoder.vcs.git_ops`).
  • `ops.telemetry.logger.log` emits redacted console lines but does not
    persist structured data.

The schema below covers those existing use cases while adding fields
needed for Phase 1 (Observer telemetry bus):
  - high-level `kind` categorisation (plan, prompt, apply, test, etc.)
  - `source` (component/agent emitting the event)
  - `subject` (resource identifier: file path, interaction id, etc.)
  - `metadata` for light-weight key/value attributes
  - `payload` for structured data (may be redacted downstream)
  - `tags` for quick faceting / filtering (e.g. ["auto", "llm:gpt-5"]).

Future phases can extend this dataclass; downstream sinks should consume
`TelemetryEvent.to_dict()` so additional fields do not break consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


class EventKind(str, Enum):
    """High-level telemetry categories.

    The values are string enums so they serialise cleanly. Additional
    kinds can be appended without breaking existing storage formats.
    """

    PROMPT = "prompt"            # User → agent prompt or instruction
    COMPLETION = "completion"    # Provider completion / model response
    PLAN = "plan"                # Planner output / JSON plan artifact
    APPLY = "apply"              # Apply attempt outcome (success/fail)
    TEST = "test"                # Test/healthcheck result summary
    SNAPSHOT = "snapshot"        # Repo snapshot / restore lifecycle
    PLUGIN = "plugin"            # Plugin load/unload/diagnostic events
    TELEMETRY = "telemetry"      # Internal bus status/heartbeat
    METRIC = "metric"            # Numeric metrics (latency, cost, etc.)
    OTHER = "other"              # Fallback for legacy/unspecified events


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TelemetryEvent:
    """Structured telemetry event.

    Parameters
    ----------
    kind:
        EventKind (or compatible string) describing the event.
    source:
        Component emitting the event (e.g., "selfcoder.orchestrator").
    subject:
        Optional identifier for the entity being acted on (file path,
        interaction id, test suite).
    metadata:
        Shallow key/value pairs for quick slicing (strings, ints, floats).
    payload:
        JSON-serialisable structure with detailed data (diffs, request
        ids, etc.). May be redacted downstream if privacy flags demand.
    tags:
        Lightweight labels, helping vector store faceting later on.
    timestamp:
        ISO8601 timestamp; defaults to time of instantiation.
    redacted:
        Flag indicating payload contains redacted fields.
    """

    kind: EventKind | str
    source: str
    subject: Optional[str] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    payload: Optional[Mapping[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=_utcnow_iso)
    redacted: bool = False

    def as_kind(self) -> EventKind:
        """Return the EventKind enum, coercing strings to EventKind when possible."""
        if isinstance(self.kind, EventKind):
            return self.kind
        try:
            return EventKind(str(self.kind))
        except ValueError:
            return EventKind.OTHER

    def tag(self, *labels: str) -> None:
        """Append one or more labels to the tag list (deduplicated)."""
        for label in labels:
            label = label.strip()
            if label and label not in self.tags:
                self.tags.append(label)

    def redact(self, keys: Iterable[str]) -> None:
        """Mark specific payload keys as redacted (if present)."""
        if not self.payload:
            return
        sanitized: Dict[str, Any] = dict(self.payload)
        mutated = False
        for key in keys:
            if key in sanitized:
                sanitized[key] = "***"
                mutated = True
        if mutated:
            self.payload = sanitized
            self.redacted = True

    def to_dict(self) -> Dict[str, Any]:
        """Plain dictionary suitable for JSON serialisation."""
        resolved_kind = self.as_kind()
        kind_value = resolved_kind.value if isinstance(resolved_kind, EventKind) else str(self.kind)
        return {
            "timestamp": self.timestamp,
            "kind": kind_value,
            "source": self.source,
            "subject": self.subject,
            "metadata": dict(self.metadata),
            "payload": dict(self.payload) if isinstance(self.payload, Mapping) else self.payload,
            "tags": list(self.tags),
            "redacted": self.redacted,
        }


__all__ = ["EventKind", "TelemetryEvent"]
