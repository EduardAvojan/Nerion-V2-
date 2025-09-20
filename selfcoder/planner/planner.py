"""Planner module — turns natural language into a Selfcoder action plan."""

from __future__ import annotations
import re
import uuid
from typing import Any

def plan_from_text(instruction: str, *, target_file: str | None = None) -> dict:
    """
    Small heuristic planner -> returns {"actions": [...]} from a natural instruction.
    Adds support for:
      - create file "path.py"
      - add/insert/create function <name> [with docstring "..."]
      - add/insert/create class <Name> [with docstring "..."]
      - "in/into/to <file.py>" to hint target file

    Params:
      - target_file (optional): explicit target file to apply the plan to. If not provided,
        a filename may be inferred from the instruction.

    Behavior:
      - Emits a clarification payload when the instruction is empty or when an intent like
        "insert function"/"insert class" lacks a required symbol name.
    """
    def _quoted(s: str) -> str | None:
        m = re.search(r"['\"]([^'\"]+)['\"]", s)
        return m.group(1) if m else None

    def _find_filename(s: str) -> str | None:
        q = _quoted(s)
        if q and q.strip().endswith(".py"):
            return q.strip()
        m = re.search(r"([A-Za-z0-9_./-]+\.py)\b", s)
        return m.group(1) if m else None

    def _func_name(s: str) -> str | None:
        # Prefer explicit patterns like "to function greet" / "function greet"
        # Handle common phrasing like "function named get_weather" or "function name get_weather".
        m = re.search(r"\bfunction\s+(?:named|name)\s+([A-Za-z_]\w*)\b", s, flags=re.I)
        if m:
            return m.group(1)
        m = re.search(r"\b(?:to|of)\s+function\s+([A-Za-z_]\w*)\b", s, flags=re.I)
        if m:
            return m.group(1)
        # Common case: 'function get_weather' (avoid capturing the word 'named')
        m = re.search(r"\bfunction\s+([A-Za-z_]\w*)\b", s, flags=re.I)
        if m and m.group(1).lower() != "docstring":
            return m.group(1)
        # Fallback: a bare identifier followed by "(" or quotes
        m = re.search(r"\b([A-Za-z_]\w*)\b\s*(?:\(|['\"])", s)
        name = m.group(1) if m else None
        if name and name.lower() == "docstring":
            return None
        return name

    def _class_name(s: str) -> str | None:
        m = re.search(r"\bclass\s+([A-Za-z_]\w*)", s, flags=re.I)
        return m.group(1) if m else None

    def _target_file_hint(s: str) -> str | None:
        m = re.search(r"\b(?:in|into|to)\s+([A-Za-z0-9_./-]+\.py)\b", s, flags=re.I)
        return m.group(1) if m else None

    norm = instruction.lower()
    actions: list[dict[str, Any]] = []
    file_hint: str | None = _target_file_hint(instruction)
    if target_file is None and file_hint:
        target_file = file_hint
    created_target = False  # whether this plan will create the target file

    # Track underspecified intents to produce clearer clarification messages
    missing_fn_name = False
    missing_class_name = False

    # --- file creation ---
    if re.search(r"\b(create|new|add)\s+(?:a\s+)?(?:python\s+)?file\b", norm):
        fname = _find_filename(instruction)
        doc = _quoted(instruction)
        payload = {"path": fname} if fname else {}
        if doc and not (doc.endswith(".py")):
            payload["doc"] = doc
        actions.append({"kind": "create_file", "payload": payload})
        if fname and not target_file:
            target_file = fname
        if fname:
            created_target = True

    # --- function insertion ---
    if re.search(r"\b(add|insert|create)\s+(?:a\s+)?function\b", norm) and "function docstring" not in norm:
        fn = _func_name(instruction)
        doc = _quoted(instruction)
        if fn:
            actions.append({
                "kind": "insert_function",
                "payload": {"name": fn, "doc": (doc if doc and not doc.endswith('.py') else None)}
            })
        else:
            # Intent detected but no function name provided
            missing_fn_name = True

    # --- class insertion ---
    if re.search(r"\b(add|insert|create)\s+(?:a\s+)?class\b", norm):
        cn = _class_name(instruction)
        doc = _quoted(instruction)
        if cn:
            actions.append({
                "kind": "insert_class",
                "payload": {"name": cn, "doc": (doc if doc and not doc.endswith('.py') else None)}
            })
        else:
            # Intent detected but no class name provided
            missing_class_name = True

    # Existing simple intents (remain for backward compatibility)
    if "module docstring" in norm:
        doc = _quoted(instruction) or "Auto-generated module docstring."
        actions.append({"kind": "add_module_docstring", "payload": {"doc": doc}})

    if "function docstring" in norm:
        fn = (_func_name(instruction)
              or (re.search(r"\bto\s+function\s+([A-Za-z_]\w*)\b", instruction, flags=re.I) or re.search(r"\bfunction\s+([A-Za-z_]\w*)\b", instruction, flags=re.I)))
        fn = fn.group(1) if hasattr(fn, "group") else fn
        doc = _quoted(instruction) or "Auto-generated function docstring."
        if fn:
            actions.append({"kind": "add_function_docstring", "payload": {"function": fn, "doc": doc}})

    if "entry log" in norm or "enter log" in norm:
        fn = _func_name(instruction)
        if fn:
            actions.append({"kind": "inject_function_entry_log", "payload": {"function": fn}})

    if "exit log" in norm or "leave log" in norm:
        fn = _func_name(instruction)
        if fn:
            actions.append({"kind": "inject_function_exit_log", "payload": {"function": fn}})

    if ("try" in norm and "except" in norm) or ("try/except" in norm):
        fn = _func_name(instruction)
        if fn:
            actions.append({"kind": "try_except_wrapper", "payload": {"function": fn}})

    plan: dict[str, Any] = {"actions": actions}
    if target_file:
        plan["target_file"] = target_file

    # Add optional coordination fields recognized by the schema
    plan["bundle_id"] = uuid.uuid4().hex
    meta: dict[str, Any] = {"source": "natural_language"}
    if target_file:
        meta["target_file"] = target_file
    plan["metadata"] = meta

    pre: list[str] = []
    # If we're not creating the target, ensure it exists before applying edits
    if target_file and not created_target:
        pre.append(f"file_exists:{target_file}")
    # Avoid inserting an already existing symbol when we can infer the symbol name
    for a in actions:
        if a.get("kind") == "insert_function":
            nm = (a.get("payload") or {}).get("name")
            if nm:
                pre.append(f"symbol_absent:{nm}{'@'+target_file if target_file else ''}")
    plan["preconditions"] = pre or None

    # Basic postcondition hint; more can be added by higher-level tooling
    plan["postconditions"] = ["no_unresolved_imports"]

    # If no actions or missing critical info, emit clarification prompts
    clarify: list[str] = []
    # If there are no actions and no explicit underspecified intents, say so
    if not actions and not (missing_fn_name or missing_class_name):
        clarify.append("No valid action detected from instruction.")
    # If intent was detected but lacked a required symbol name
    if missing_fn_name or missing_class_name or any(
        a.get("kind") in {"insert_function", "insert_class"} and not (a.get("payload") or {}).get("name")
        for a in actions
    ):
        clarify.append("Missing symbol name for insertion.")
    # If we have an insertion intent but no target file provided/hinted
    if not target_file and (missing_fn_name or missing_class_name or any(a.get("kind") in {"insert_function", "insert_class"} for a in actions)):
        clarify.append("No target file specified.")
    if clarify:
        plan["clarify"] = clarify
        # Mark in metadata that a clarification round is required
        plan.setdefault("metadata", {})
        plan["metadata"]["clarify_required"] = True
    else:
        plan.setdefault("metadata", {}).setdefault("clarify_required", False)

    return plan

def plan_edits_from_nl(instruction: str, file: str | None = None, scaffold_tests: bool = True) -> dict:
    """
    Turn a natural-language instruction into a minimal plan dict.
    """
    plan = plan_from_text(instruction)
    if file:
        plan["target_file"] = file
    # Step 12b: pair code actions with test scaffolds
    if scaffold_tests:
        target = plan.get("target_file") or file
        if target:
            seen = set()
            extra = []
            for a in plan.get("actions", []) or []:
                if not isinstance(a, dict):
                    continue
                k = (a.get("kind") or a.get("action") or "").lower()
                p = a.get("payload") or {}
                sym = p.get("name") or p.get("function") or p.get("class_name") or p.get("class") or p.get("symbol")
                typ = "function" if ("function" in k or "function" in str(p)) else ("class" if ("class" in k or p.get("class") or p.get("class_name")) else "unknown")
                if k in {"insert_function", "insert_class", "add_function_docstring", "rename_symbol"} and sym:
                    key = (sym, typ)
                    if key in seen:
                        continue
                    seen.add(key)
                    extra.append({
                        "kind": "ensure_test",
                        "payload": {"symbol": sym, "target_file": target, "type": typ},
                    })
            if extra:
                plan.setdefault("actions", []).extend(extra)

    # Ensure metadata captures context from this helper
    meta = plan.get("metadata") or {}
    meta.setdefault("helper", "plan_edits_from_nl")
    meta["scaffold_tests"] = bool(scaffold_tests)
    if file:
        meta["target_file_override"] = file
    plan["metadata"] = meta

    # Prefer at least one postcondition when tests are scaffolded
    if scaffold_tests:
        posts = list(plan.get("postconditions") or [])
        if "tests_collect" not in posts:
            posts.append("tests_collect")
        plan["postconditions"] = posts

    # If the base plan asked for clarification, propagate
    if plan.get("clarify"):
        plan["metadata"]["clarify_required"] = True

    return plan
