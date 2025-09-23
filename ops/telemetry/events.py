"""High-level telemetry helpers for common Nerion events."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from . import EventKind, TelemetryEvent, ensure_default_sinks, publish


ensure_default_sinks()


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _hash_blob(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, sort_keys=True)
    except TypeError:
        # Fallback: convert to string to avoid serialization crashes.
        return json.dumps(str(data), ensure_ascii=False)


def _sanitised_text_payload(text: Optional[str], *, allow_raw: bool) -> Dict[str, Any]:
    text = text or ""
    payload: Dict[str, Any] = {
        "char_count": len(text),
        "sha256": _hash_blob(text)[:16] if text else None,
    }
    if allow_raw and text:
        payload["text"] = text
    return payload


def _sanitised_messages_payload(messages: Optional[Sequence[Mapping[str, Any]]], *, allow_raw: bool) -> Dict[str, Any]:
    msgs = list(messages or [])
    serialised = _safe_json(msgs)
    payload: Dict[str, Any] = {
        "message_count": len(msgs),
        "char_count": len(serialised),
        "sha256": _hash_blob(serialised)[:16] if msgs else None,
    }
    if allow_raw and msgs:
        payload["messages"] = msgs
    return payload


def record_prompt(
    *,
    source: str,
    prompt: Optional[str] = None,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
    subject: Optional[str] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
) -> TelemetryEvent:
    allow_raw = _bool_env("NERION_V2_LOG_PROMPTS", False)
    metadata = dict(metadata or {})
    payload: Dict[str, Any]
    if messages is not None:
        metadata.setdefault("input_type", "messages")
        payload = _sanitised_messages_payload(messages, allow_raw=allow_raw)
    else:
        metadata.setdefault("input_type", "prompt")
        payload = _sanitised_text_payload(prompt, allow_raw=allow_raw)
    event = TelemetryEvent(
        kind=EventKind.PROMPT,
        source=source,
        subject=subject,
        metadata=metadata,
        payload=payload,
        tags=list(tags or []),
        redacted=not allow_raw,
    )
    publish(event)
    return event


def record_completion(
    *,
    source: str,
    text: Optional[str],
    subject: Optional[str] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
) -> TelemetryEvent:
    allow_raw = _bool_env("NERION_V2_LOG_PROMPTS", False)
    payload = _sanitised_text_payload(text, allow_raw=allow_raw)
    event = TelemetryEvent(
        kind=EventKind.COMPLETION,
        source=source,
        subject=subject,
        metadata=dict(metadata or {}),
        payload=payload,
        tags=list(tags or []),
        redacted=not allow_raw,
    )
    publish(event)
    return event


def record_plan(
    *,
    source: str,
    instruction: Optional[str],
    plan: Mapping[str, Any],
    subject: Optional[str] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
) -> TelemetryEvent:
    allow_raw = _bool_env("NERION_V2_LOG_PROMPTS", False)
    safe_instruction = _sanitised_text_payload(instruction, allow_raw=allow_raw)
    payload = {
        "instruction": safe_instruction,
        "plan": json.loads(_safe_json(plan)),
    }
    event = TelemetryEvent(
        kind=EventKind.PLAN,
        source=source,
        subject=subject,
        metadata=dict(metadata or {}),
        payload=payload,
        tags=list(tags or []),
        redacted=not allow_raw,
    )
    publish(event)
    return event


__all__ = ["record_prompt", "record_completion", "record_plan"]

