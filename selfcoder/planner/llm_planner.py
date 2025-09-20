"""LLM-backed planner that asks DeepSeek Coder V2 to produce a structured plan.

This module is optional and degrades gracefully to the heuristic planner when
the local model or langchain_ollama is unavailable. When strict mode is
enabled (NERION_LLM_STRICT=1), failures will raise so the caller can exit
non-zero instead of silently falling back.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from pathlib import Path

from .planner import plan_edits_from_nl as _heuristic_plan
from selfcoder.plans.schema import ALLOWED_ACTIONS


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


def plan_with_llm(instruction: str, file: Optional[str] = None) -> Dict[str, Any]:
    """Plan code edits using the DeepSeek Coder V2 model.

    Behavior:
      - Default: safe fallback to heuristic planner on failures.
      - Strict: if NERION_LLM_STRICT is set (e.g., when --llm is used), raise on
        failures so the CLI can exit non-zero and surface the error.
    Returns a plan dict compatible with selfcoder.orchestrator/apply_plan.
    """
    strict = bool(os.getenv("NERION_LLM_STRICT"))
    strict_json = bool(os.getenv("NERION_JSON_GRAMMAR"))
    # JSON grammar implies strict JSON parsing behavior
    if strict_json:
        strict = True
    # Auto-select backend/model if requested or not set
    if os.getenv("NERION_CODER_AUTO") or (not os.getenv("NERION_CODER_BACKEND") and not os.getenv("NERION_CODER_MODEL")):
        try:
            from app.parent.selector import auto_select_model  # type: ignore
            # Optional task-aware preference
            preferred_family = None
            try:
                from app.parent.task_model_picker import rank_families  # type: ignore
                fams = rank_families(instruction)
                preferred_family = fams[0] if fams else None
            except Exception:
                preferred_family = None
            choice = auto_select_model()
            if choice:
                be, m, base = choice
                if preferred_family and preferred_family not in (m or "").lower():
                    for tok in ("deepseek-coder", "qwen2.5-coder", "starcoder2", "codellama"):
                        if tok in (m or "").lower():
                            m = preferred_family
                            break
                os.environ.setdefault("NERION_CODER_BACKEND", be)
                os.environ.setdefault("NERION_CODER_MODEL", m)
                if base:
                    os.environ.setdefault("NERION_CODER_BASE_URL", base)
        except Exception:
            pass

    # Ensure backend/model availability (may prompt user via message when network is off)
    try:
        from app.parent.provision import ensure_available  # type: ignore
        be = os.getenv("NERION_CODER_BACKEND") or "ollama"
        m = os.getenv("NERION_CODER_MODEL") or "deepseek-coder-v2"
        # If llama_cpp and URL/PATH not provided, try catalog defaults
        if be == "llama_cpp" and not (os.getenv("LLAMA_CPP_MODEL_URL") and os.getenv("LLAMA_CPP_MODEL_PATH")):
            try:
                from app.parent.model_catalog import resolve  # type: ignore
                src = resolve("llama_cpp", m)
                if src and isinstance(src.get("url"), str):
                    os.environ.setdefault("LLAMA_CPP_MODEL_URL", src["url"]) 
                    filename = src.get("filename") or src["url"].rsplit("/", 1)[-1]
                    default_path = str((Path.home() / f".cache/nerion/models/llama.cpp/{filename}").resolve())
                    os.environ.setdefault("LLAMA_CPP_MODEL_PATH", default_path)
            except Exception:
                pass
        ok, msg = ensure_available(be, m)
        if not ok and strict:
            raise RuntimeError(msg)
    except Exception:
        if strict:
            raise

    # Prefer unified multi-backend coder; fallback to DeepSeekCoderV2
    coder = None
    try:
        from app.parent.coder import Coder  # type: ignore
        coder = Coder(model=os.getenv("NERION_CODER_MODEL") or None,
                      backend=os.getenv("NERION_CODER_BACKEND") or None)
    except Exception:
        try:
            from app.parent.coder_v2 import DeepSeekCoderV2  # type: ignore
            coder = DeepSeekCoderV2(model=os.getenv("NERION_CODER_MODEL") or None)
        except Exception as e:
            if strict:
                raise RuntimeError(f"LLM adapter unavailable: {e}")
            return _heuristic_plan(instruction, file, scaffold_tests=True)
    # Compose a compact instruction for Coder V2
    tgt_hint = f" Target file: {file}." if file else ""
    allowed = ", ".join(sorted(ALLOWED_ACTIONS))
    prompt = (
        f"Instruction: {instruction.strip()}\n"
        f"{tgt_hint}\n"
        f"Emit ONLY JSON. Allowed action kinds: {allowed}. "
        f"Use payloads with minimal keys."
    )
    txt = coder.complete_json(prompt, system=_SYSTEM)
    if not txt:
        if strict:
            raise RuntimeError("LLM returned no content (is the model running?)")
        return _heuristic_plan(instruction, file, scaffold_tests=True)

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
        return _heuristic_plan(instruction, file, scaffold_tests=True)

    plan = _normalize_plan_obj(obj, instruction=instruction, file=file)
    # If no actions made it through filtering, fall back to heuristic
    if not plan.get("actions"):
        if strict:
            raise RuntimeError("LLM plan contained no allowed actions")
        return _heuristic_plan(instruction, file, scaffold_tests=True)
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
    return plan

__all__ = ["plan_with_llm"]
