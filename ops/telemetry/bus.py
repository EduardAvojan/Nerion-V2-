"""Central telemetry bus with pluggable sinks.

Phase 1.1 introduces this module to provide a single publish/subscribe
surface for structured telemetry events. Downstream sinks can attach to
persist events (JSONL, SQL, vector stores) or stream them to external
systems when policies allow.

Key features (v0):
  • thread-safe subscription management
  • batching hooks (configurable batch size + explicit flush)
  • payload redaction support (event-level)
  • best-effort fault isolation per sink

Future extensions (tracked in Phase 1.2+):
  - async/background flush via worker thread
  - per-sink filters and back-pressure metrics
  - integration with provider cost/latency collection
"""

from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

from ops.telemetry.logger import log as _log

from .schema import EventKind, TelemetryEvent


class TelemetrySink(Protocol):
    """Sink interface for telemetry consumers."""

    supports_batch: bool = False

    def emit(self, event: TelemetryEvent) -> None:
        ...

    def emit_batch(self, events: Sequence[TelemetryEvent]) -> None:
        ...


SinkFactory = Callable[[], TelemetrySink]


class TelemetryBus:
    """In-memory dispatcher for telemetry events."""

    def __init__(self, batch_size: int = 1) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._sinks: List[TelemetrySink] = []
        self._lock = RLock()
        self._queue: deque[TelemetryEvent] = deque()
        self._batch_size = batch_size

    # -- subscription -----------------------------------------------------
    def subscribe(self, sink: TelemetrySink) -> None:
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def unsubscribe(self, sink: TelemetrySink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    # -- publication ------------------------------------------------------
    def publish(
        self,
        event: TelemetryEvent,
        *,
        redact_keys: Optional[Iterable[str]] = None,
        extra_tags: Optional[Iterable[str]] = None,
        auto_flush: bool = True,
    ) -> None:
        """Queue an event for sink consumption.

        Parameters are applied eagerly (tagging, redaction). The event is
        appended to the internal queue and flushed to sinks when either
        `auto_flush` is true and the queue size meets `batch_size`, or
        when `flush()` is called manually.
        """

        if redact_keys:
            event.redact(redact_keys)
        if extra_tags:
            event.tag(*extra_tags)

        with self._lock:
            self._queue.append(event)
            should_flush = auto_flush and len(self._queue) >= self._batch_size

        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._queue or not self._sinks:
                self._queue.clear()
                return
            batch = list(self._queue)
            self._queue.clear()
            sinks = list(self._sinks)

        for sink in sinks:
            try:
                if getattr(sink, "supports_batch", False):
                    sink.emit_batch(batch)
                else:
                    for event in batch:
                        sink.emit(event)
            except Exception as exc:  # pragma: no cover - defensive guard
                _log(
                    f"Telemetry sink {sink.__class__.__name__} raised: {exc}",
                    level="ERROR",
                )

    # -- helpers ----------------------------------------------------------
    def publish_dict(
        self,
        *,
        kind: EventKind | str,
        source: str,
        subject: Optional[str] = None,
        metadata: Optional[dict] = None,
        payload: Optional[dict] = None,
        tags: Optional[Iterable[str]] = None,
        redact_keys: Optional[Iterable[str]] = None,
    ) -> TelemetryEvent:
        event = TelemetryEvent(
            kind=kind,
            source=source,
            subject=subject,
            metadata=metadata or {},
            payload=payload,
            tags=list(tags or []),
        )
        self.publish(event, redact_keys=redact_keys, extra_tags=None, auto_flush=True)
        return event


_GLOBAL_BUS = TelemetryBus(batch_size=32)


def get_bus() -> TelemetryBus:
    return _GLOBAL_BUS


def publish(event: TelemetryEvent, **kwargs) -> None:
    _GLOBAL_BUS.publish(event, **kwargs)


def publish_dict(**kwargs) -> TelemetryEvent:
    return _GLOBAL_BUS.publish_dict(**kwargs)


__all__ = [
    "TelemetryBus",
    "TelemetrySink",
    "publish",
    "publish_dict",
    "get_bus",
]

