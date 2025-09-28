"""Provider-backed planner that requests a structured plan from the configured API coder.

This module falls back to the heuristic planner when the provider is unavailable unless
`NERION_LLM_STRICT=1`, in which case the error is surfaced to the caller.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from .planner import plan_edits_from_nl as _heuristic_plan
from selfcoder.plans.schema import ALLOWED_ACTIONS
from app.parent.coder import Coder
from selfcoder.planner.utils import attach_brief_metadata

try:
    from ops.telemetry.events import record_plan as _telemetry_record_plan
except Exception:  # pragma: no cover - optional telemetry
    def _telemetry_record_plan(*_args, **_kwargs):  # type: ignore
        return None


BriefContext = Optional[Dict[str, Any]]


_SYSTEM = (
    "You are a code-edit planner for the Nerion self-coder. "
    "Given a user instruction (and optional target file), output ONLY a JSON object "
    "with keys: actions (list), optional target_file, optional preconditions, optional postconditions, metadata, bundle_id. "
    "Follow these constraints strictly: the only allowed action kinds are: "
    + ", ".join(sorted(ALLOWED_ACTIONS))
    + ". Do not include any text outside the JSON."
)


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer fenced code blocks with json
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.I)
    if m:
        return m.group(1)
    # Fallback: first balanced curly block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _quoted_from_instruction(text: str) -> Optional[str]:
    """Return the first simple quoted substring from the instruction, if any."""
    if not text:
        return None
    m = re.search(r"['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else None


def _normalize_plan_obj(obj: Dict[str, Any], *, instruction: str, file: Optional[str]) -> Dict[str, Any]:
    plan: Dict[str, Any] = {}
    actions = obj.get("actions") or []
    # Keep only allowed kinds and ensure payload dict
    norm_actions = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        k = (a.get("kind") or a.get("action") or "").strip()
        if k not in ALLOWED_ACTIONS:
            continue
        payload = a.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        # Heuristic fill: ensure minimal payloads for known kinds
        if k == "add_module_docstring":
            doc = payload.get("doc") or payload.get("docstring")
            if not doc:
                # Try to extract from the user's instruction; fall back to a generic placeholder
                doc = _quoted_from_instruction(instruction) or "Auto-generated module docstring."
                payload = dict(payload)
                payload["doc"] = doc
        norm_actions.append({"kind": k, "payload": payload})
    plan["actions"] = norm_actions

    target_file = obj.get("target_file") or file
    if target_file:
        plan["target_file"] = target_file

    pre = obj.get("preconditions")
    if isinstance(pre, list):
        plan["preconditions"] = [str(x) for x in pre]
    post = obj.get("postconditions")
    if isinstance(post, list):
        plan["postconditions"] = [str(x) for x in post]

    meta = obj.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("source", "llm_coder_v2")
    meta.setdefault("instruction", instruction)
    if target_file:
        meta.setdefault("target_file", target_file)
    plan["metadata"] = meta

    if obj.get("bundle_id"):
        plan["bundle_id"] = str(obj["bundle_id"])  # best-effort
    return plan


def _format_brief_context_block(context: Dict[str, Any]) -> str:
    brief = context.get("brief") if isinstance(context, dict) else None
    if not isinstance(brief, dict):
        brief = {}
    lines = []
    component = brief.get("component")
    if component:
        lines.append(f"Component: {component}")
    decision = context.get("decision")
    policy = context.get("policy")
    if decision:
        if policy:
            lines.append(f"Decision: {decision} (policy {policy})")
        else:
            lines.append(f"Decision: {decision}")
    risk = context.get("risk_score")
    effort = context.get("effort_score")
    cost = context.get("estimated_cost")
    metrics = []
    if risk is not None:
        metrics.append(f"risk {risk}")
    if effort is not None:
        metrics.append(f"effort {effort}")
    if cost is not None:
        metrics.append(f"cost ${cost}")
    if metrics:
        lines.append("Signals: " + ", ".join(str(m) for m in metrics))
    summary = brief.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")
    rationale = brief.get("rationale") or []
    for item in rationale[:2]:
        lines.append(f"Rationale: {item}")
    acceptance = brief.get("acceptance_criteria") or []
    if acceptance:
        lines.append(f"Acceptance hint: {acceptance[0]}")
    suggestions = context.get("suggested_targets") or []
    if suggestions:
        suggestion_text = ", ".join(str(s) for s in suggestions[:3])
        lines.append(f"Suggested targets: {suggestion_text}")
    return "\n".join(lines)


def plan_with_llm(
    instruction: str,
    file: Optional[str] = None,
    brief_context: BriefContext = None,
) -> Dict[str, Any]:
    """Plan code edits using the DeepSeek Coder V2 model.

    Behavior:
      - Default: safe fallback to heuristic planner on failures.
      - Strict: if NERION_LLM_STRICT is set (e.g., when --llm is used), raise on
        failures so the CLI can exit non-zero and surface the error.
    Returns a plan dict compatible with selfcoder.orchestrator/apply_plan.
    """
    strict = bool(os.getenv("NERION_LLM_STRICT"))
    strict_json = bool(os.getenv("NERION_JSON_GRAMMAR"))
    if strict_json:
        strict = True

    coder = Coder(role="code")
    # Compose a compact instruction for Coder V2
    allowed = ", ".join(sorted(ALLOWED_ACTIONS))
    prompt_parts = []
    if brief_context:
        prompt_parts.append("Architect brief context:\n" + _format_brief_context_block(brief_context))
    prompt_parts.append(f"Instruction: {instruction.strip()}")
    if file:
        prompt_parts.append(f"Target file hint: {file}")
    prompt_parts.append(
        f"Emit ONLY JSON. Allowed action kinds: {allowed}. Use payloads with minimal keys."
    )
    prompt = "\n\n".join(part for part in prompt_parts if part)
    txt = coder.complete_json(prompt, system=_SYSTEM)
    if not txt:
        if strict:
            raise RuntimeError("LLM returned no content (is the model running?)")
        return _heuristic_plan(instruction, file, scaffold_tests=True, brief_context=brief_context)

    raw = _extract_json_block(txt)
    if raw is None and strict_json:
        # Enforce explicit JSON block when grammar mode is requested
        raise RuntimeError("LLM did not return a JSON block")
    raw = raw or txt
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("planner returned non-object")
    except Exception:
        if strict:
            raise RuntimeError("LLM returned invalid JSON plan")
        return _heuristic_plan(instruction, file, scaffold_tests=True, brief_context=brief_context)

    plan = _normalize_plan_obj(obj, instruction=instruction, file=file)
    # If no actions made it through filtering, fall back to heuristic
    if not plan.get("actions"):
        if strict:
            raise RuntimeError("LLM plan contained no allowed actions")
        return _heuristic_plan(instruction, file, scaffold_tests=True, brief_context=brief_context)
    # Add a default postcondition if none present
    if not plan.get("postconditions"):
        posts = ["no_unresolved_imports", "tests_collect"]
        # For TS/JS targets, add optional gates (best-effort)
        try:
            tgt = str(plan.get("target_file") or (file or ""))
            if tgt.endswith(('.ts', '.tsx')):
                posts += ["eslint_clean", "tsc_ok"]
            elif tgt.endswith(('.js', '.jsx', '.mjs', '.cjs')):
                posts += ["eslint_clean"]
        except Exception:
            pass
        plan["postconditions"] = posts

    if brief_context:
        plan = attach_brief_metadata(plan, brief_context)

    try:
        meta = {
            "planner": "llm",
            "strict": bool(strict),
            "strict_json": bool(strict_json),
            "actions": len(plan.get("actions") or []),
        }
        if file:
            meta["target_file_arg"] = file
        if brief_context and isinstance(brief_context, dict):
            meta["architect_decision"] = brief_context.get("decision")
        _telemetry_record_plan(
            source="selfcoder.planner.llm",
            instruction=instruction,
            plan=plan,
            subject=plan.get("target_file"),
            metadata=meta,
            tags=["planner", "llm"],
        )
    except Exception:  # pragma: no cover
        pass
    return plan

__all__ = ["plan_with_llm"]
