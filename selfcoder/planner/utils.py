from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import os

from selfcoder.plans.schema import validate_plan


def sanitize_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce allowed actions/payloads and normalize to a compact dict.

    - Validates via plans.schema.validate_plan
    - If validation fails due to unknown actions, drops them and re-validates
    - Returns a dict with actions in {kind, payload} shape and optional fields
    """
    try:
        plan = validate_plan(raw)
    except Exception:
        # Attempt to drop invalid actions and re-validate
        acts = raw.get("actions") or []
        if isinstance(acts, list):
            filt = []
            from selfcoder.plans.schema import ALLOWED_ACTIONS
            for a in acts:
                if not isinstance(a, dict):
                    continue
                k = (a.get("kind") or a.get("action") or "").strip()
                if k in ALLOWED_ACTIONS:
                    filt.append(a)
            raw2 = dict(raw)
            raw2["actions"] = filt
            plan = validate_plan(raw2)
        else:
            raise
    # Convert typed Plan to normalized dict
    out: Dict[str, Any] = {
        "actions": [],
    }
    if plan.description:
        out["description"] = plan.description
    if plan.origin:
        out["origin"] = plan.origin
    if plan.preconditions:
        out["preconditions"] = list(plan.preconditions)
    if plan.postconditions:
        out["postconditions"] = list(plan.postconditions)
    if plan.metadata:
        out["metadata"] = dict(plan.metadata)
    if plan.bundle_id:
        out["bundle_id"] = plan.bundle_id
    # Preserve explicit target_file from the raw plan when present
    try:
        tf = raw.get("target_file")
        if isinstance(tf, str) and tf.strip():
            out["target_file"] = tf
    except Exception:
        pass
    # Actions to canonical payload form
    for a in plan.actions:
        payload: Dict[str, Any] = {}
        if a.path is not None:
            payload["path"] = a.path
        if a.content is not None:
            # Prefer 'doc' for doc-bearing actions to align with transformers
            if str(a.action) in {"add_module_docstring", "add_function_docstring", "insert_function", "insert_class"}:
                payload["doc"] = a.content
            else:
                payload["content"] = a.content
        if a.lineno_start is not None:
            payload["lineno_start"] = a.lineno_start
        if a.lineno_end is not None:
            payload["lineno_end"] = a.lineno_end
        if a.symbol is not None:
            payload["symbol"] = a.symbol
        out["actions"].append({"kind": a.action, "payload": payload})
    return out


def repo_fingerprint(root: Path) -> str:
    """Compute a lightweight fingerprint of the repo state for plan caching.

    Prefers git HEAD + index state; falls back to hashing mtimes of .py files.
    """
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
        if head.returncode == 0:
            status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
            return hashlib.sha1((head.stdout.strip() + "\n" + status.stdout).encode("utf-8")).hexdigest()
    except Exception:
        pass
    h = hashlib.sha1()
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith((".py", ".toml", ".yaml", ".yml")):
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                    h.update(str(p.relative_to(root)).encode("utf-8"))
                    # Include size and high-res mtime; also a small content hash
                    h.update(str(st.st_size).encode("utf-8"))
                    h.update(str(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))).encode("utf-8"))
                    try:
                        data = p.read_bytes()
                        h.update(hashlib.sha1(data).digest())
                    except Exception:
                        pass
                except Exception:
                    continue
    return h.hexdigest()


def load_plan_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_plan_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def attach_brief_metadata(plan: Dict[str, Any], brief_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Embed architect brief context into a plan's metadata."""

    if not brief_context:
        return plan

    meta = dict(plan.get("metadata") or {})
    brief = brief_context.get("brief") if isinstance(brief_context, dict) else None
    if not isinstance(brief, dict):
        brief = {}

    meta["architect_brief"] = {
        "id": brief.get("id"),
        "component": brief.get("component"),
        "title": brief.get("title"),
        "decision": brief_context.get("decision"),
        "policy": brief_context.get("policy"),
        "risk_score": brief_context.get("risk_score"),
        "effort_score": brief_context.get("effort_score"),
        "estimated_cost": brief_context.get("estimated_cost"),
        "effective_priority": brief_context.get("effective_priority"),
        "reasons": list(brief_context.get("reasons") or []),
        "summary": brief.get("summary"),
        "suggested_targets": list(brief_context.get("suggested_targets") or []),
        "alternates": list(brief_context.get("alternates") or []),
        "gating": dict(brief_context.get("gating") or {}),
    }
    plan["metadata"] = meta
    return plan
