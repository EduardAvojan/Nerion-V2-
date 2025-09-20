from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

# Allowed atomic actions in a plan. Keep conservative.
ALLOWED_ACTIONS = {
    "create_file",
    "insert_function",
    "insert_class",
    "insert_import",
    "replace_node",
    "rename_symbol",
    "append_to_file",
    "delete_lines",
    # Planner may request scaffolding when enabled; allow it for validation parity
    "ensure_test",
    # Formalize previously-legacy docstring actions
    "add_module_docstring",
    "add_function_docstring",
    # Textual-diff fallback (bench/repair)
    "apply_unified_diff",
    # JS/TS (node bridge/textual)
    "update_import",
    "export_default",
    "export_named",
    "insert_interface",
    "insert_type_alias",
}


@dataclass
class PlanAction:
    action: Literal[
        "create_file",
        "insert_function",
        "insert_class",
        "insert_import",
        "replace_node",
        "rename_symbol",
        "append_to_file",
        "delete_lines",
        "ensure_test",
        "add_module_docstring",
        "add_function_docstring",
        "apply_unified_diff",
        "update_import",
        "export_default",
        "export_named",
        "insert_interface",
        "insert_type_alias",
    ]
    # Path is optional; some actions operate on an inferred/target file
    path: Optional[str] = None
    content: Optional[str] = None
    lineno_start: Optional[int] = None
    lineno_end: Optional[int] = None
    symbol: Optional[str] = None


@dataclass
class Plan:
    actions: list[PlanAction]
    description: Optional[str] = None
    origin: Optional[str] = None  # "voice", "cli", "autofix", etc.
    # Optional coordination/safety fields
    preconditions: Optional[list[str]] = None
    postconditions: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    bundle_id: Optional[str] = None


_MAX_PLAN_ACTIONS = 200
_MAX_CONTENT_BYTES = 300_000  # ~300KB safety valve


def _is_safe_relpath(p: str) -> bool:
    if not isinstance(p, str) or not p:
        return False
    if p.startswith("/") or p.startswith("~"):
        return False
    # Normalize and assert no parent escapes
    norm = str(Path(p))
    return (".." not in Path(norm).parts)


def validate_plan(raw: dict[str, Any]) -> Plan:
    """Validate a raw plan dict and return a typed Plan.

    Accepts both legacy action shape {"action": ...} and payload shape
    {"kind": ..., "payload": {...}}.
    """
    if not isinstance(raw, dict):
        raise ValueError("Plan must be an object")

    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, Sequence):
        raise ValueError("Plan.actions must be a list")
    if len(raw_actions) == 0:
        raise ValueError("Plan.actions must not be empty")
    if len(raw_actions) > _MAX_PLAN_ACTIONS:
        raise ValueError(f"Plan has too many actions (>{_MAX_PLAN_ACTIONS})")

    actions: list[PlanAction] = []
    total_bytes = 0

    for idx, a in enumerate(raw_actions):
        if not isinstance(a, dict):
            raise ValueError(f"Action #{idx} must be an object")

        # Support both legacy {action: ...} and new {kind: ..., payload: {...}}
        action = a.get("action") or a.get("kind")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Action #{idx} not allowed: {action!r}")

        payload = a.get("payload") or {}
        if payload is not None and not isinstance(payload, dict):
            raise ValueError(f"Action #{idx} payload must be an object if provided")

        # Derive fields from either top-level or payload
        path = a.get("path") or payload.get("path")
        if path is not None and not _is_safe_relpath(path):
            raise ValueError(f"Action #{idx} has unsafe path: {path!r}")

        content = a.get("content") or payload.get("content") or payload.get("doc")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError(f"Action #{idx} content must be string")
            total_bytes += len(content.encode("utf-8"))
            if total_bytes > _MAX_CONTENT_BYTES:
                raise ValueError("Plan content too large")

        lineno_start = a.get("lineno_start") or payload.get("lineno_start")
        lineno_end = a.get("lineno_end") or payload.get("lineno_end")
        if lineno_start is not None and (not isinstance(lineno_start, int) or lineno_start < 1):
            raise ValueError(f"Action #{idx} lineno_start invalid")
        if lineno_end is not None and (not isinstance(lineno_end, int) or (lineno_start is not None and lineno_end < lineno_start)):
            raise ValueError(f"Action #{idx} lineno_end invalid")

        actions.append(
            PlanAction(
                action=action,  # type: ignore[arg-type]
                path=path,
                content=content,
                lineno_start=lineno_start,
                lineno_end=lineno_end,
                symbol=a.get("symbol") or (payload.get("name") if isinstance(payload, dict) else None),
            )
        )

    # Optional fields for coordination and safety
    preconditions = raw.get("preconditions")
    if preconditions is not None:
        if not isinstance(preconditions, Sequence) or not all(isinstance(x, str) for x in preconditions):
            raise ValueError("Plan.preconditions must be a list of strings")
        preconditions = list(preconditions)

    postconditions = raw.get("postconditions")
    if postconditions is not None:
        if not isinstance(postconditions, Sequence) or not all(isinstance(x, str) for x in postconditions):
            raise ValueError("Plan.postconditions must be a list of strings")
        postconditions = list(postconditions)

    metadata = raw.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("Plan.metadata must be an object")

    bundle_id = raw.get("bundle_id")
    if bundle_id is not None and (not isinstance(bundle_id, str) or not bundle_id.strip()):
        raise ValueError("Plan.bundle_id must be a non-empty string if provided")

    return Plan(
        actions=actions,
        description=raw.get("description"),
        origin=raw.get("origin"),
        preconditions=preconditions,
        postconditions=postconditions,
        metadata=metadata,
        bundle_id=bundle_id,
    )
