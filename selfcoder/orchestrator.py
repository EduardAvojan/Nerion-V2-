from __future__ import annotations
from pathlib import Path
from selfcoder.plans.schema import validate_plan
from ops.security import fs_guard
from typing import Any, Dict, List, Tuple
import os

import ast
import importlib.util
from contextlib import suppress
import difflib
from dataclasses import dataclass

from selfcoder.security.gate import assess_plan  # preflight security gate
from datetime import datetime, timezone
from selfcoder.scoring import score_plan
from selfcoder.artifacts import PlanArtifact, save_artifact
from selfcoder import testgen as _testgen
try:
    from selfcoder.analysis.symbols import build_import_graph as _build_import_graph
except Exception:
    _build_import_graph = None
from selfcoder.tester.expander import expand_tests as _expand_tests
from selfcoder.reviewers.reviewer import review_predicted_changes as _review_predicted, format_review as _fmt_review
try:
    from selfcoder import coverage_utils as _covu
except Exception:
    _covu = None

# Optional profile resolver import
try:
    from selfcoder.policy.profile_resolver import decide as _decide_profile  # type: ignore
except Exception:
    _decide_profile = None  # type: ignore

from .actions.transformers import (
    apply_actions_via_ast,
    ModuleDocstringAdder,   # kept for optional external use
    FunctionDocstringAdder, # kept for optional external use
    build_test_scaffold,
)
try:
    from selfcoder.actions.js_ts import apply_actions_js_ts as _apply_js_ts  # type: ignore
except Exception:  # pragma: no cover
    def _apply_js_ts(src: str, actions):  # type: ignore
        return src
try:
    from selfcoder.actions.text_patch import preview_unified_diff as _preview_unified_diff
except Exception:
    _preview_unified_diff = None  # type: ignore

REPO_ROOT = fs_guard.infer_repo_root(Path('.'))

try:
    # Preferred: core journal API if available
    from core.memory.journal import log_event as _log_event  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Fallback: app-level JSONL journal
        from app import journal as _journal  # type: ignore
        def _log_event(kind: str, **fields):
            try:
                _journal.append({"kind": kind, **fields})
            except Exception:
                pass
    except Exception:
        def _log_event(*_a, **_kw):
            return None


# --- Smart runtime defaults (everyday user UX) --------------------------------
def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def prepare_for_prompt(instruction: str) -> None:
    """Project-manager style prep for a user prompt.

    - If NERION_MODE=user (default) set safe, helpful defaults:
      - Enable auto model selection (task-aware) and strict JSON planning by default
      - Try to auto-select a backend/model; if missing, print a concise consent message
    - If NERION_MODE=dev, do nothing (developer controls env).
    """
    mode = (os.getenv("NERION_MODE") or "user").strip().lower()
    if mode != "user":
        return
    # Defaults for user mode
    os.environ.setdefault("NERION_CODER_AUTO", "1")
    os.environ.setdefault("NERION_JSON_GRAMMAR", "1")
    # Optional: keep strict so planner errors are visible
    os.environ.setdefault("NERION_LLM_STRICT", "1")

    # Task-aware family preference is handled in llm_planner; here we ensure availability
    try:
        # Auto-select if missing
        if not (os.getenv("NERION_CODER_BACKEND") and os.getenv("NERION_CODER_MODEL")):
            from app.parent.selector import auto_select_model  # type: ignore
            choice = auto_select_model()
            if choice:
                be, m, base = choice
                os.environ.setdefault("NERION_CODER_BACKEND", be)
                os.environ.setdefault("NERION_CODER_MODEL", m)
                if base:
                    os.environ.setdefault("NERION_CODER_BASE_URL", base)
        # Ensure availability; if missing, print consent message (do not raise here)
        from app.parent.provision import ensure_available  # type: ignore
        be = os.getenv("NERION_CODER_BACKEND") or "ollama"
        m = os.getenv("NERION_CODER_MODEL") or "deepseek-coder-v2"
        ok, msg = ensure_available(be, m)
        if not ok:
            print(f"[orchestrator] {msg}")
    except Exception:
        # Non-fatal; heuristic planner will still work
        pass

# (imports above)

# A lightweight registry for higher-level callers (optional use)
_TRANSFORMERS: Dict[str, Any] = {
    "add_module_docstring": ModuleDocstringAdder,
    "add_function_docstring": FunctionDocstringAdder,
}

def _should_skip(file_path: Path) -> Tuple[bool, str]:
    """Best-effort bridge to git_ops.should_skip; fallback to no-skip."""
    try:
        from selfcoder.vcs.git_ops import should_skip  # type: ignore
        return should_skip(file_path)
    except Exception:
        return (False, "")

def _validate_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Accept 'action' as an alias for 'kind', ensure 'kind' is present and non-empty, and normalize payload.
    """
    out: List[Dict[str, Any]] = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        item = dict(a)
        kind = str(item.get("kind") or item.get("action") or "").strip()
        if not kind:
            continue
        item["kind"] = kind  # normalize alias
        if item.get("payload") is None:
            item["payload"] = {}
        out.append(item)
    return out

def _split_fs_and_ast_actions(actions: List[Dict[str, Any]]):
    """Split actions into filesystem-level, diff-level, and AST-level.
    FS-level supports: create_file, ensure_file, ensure_test.
    DIFF-level supports: apply_unified_diff (payload: {diff: str} or {diff_file: str}).
    Returns (fs_actions, diff_actions, ast_actions).
    """
    fs_kinds = {"create_file", "ensure_file", "ensure_test"}
    diff_kinds = {"apply_unified_diff"}
    fs_actions: List[Dict[str, Any]] = []
    diff_actions: List[Dict[str, Any]] = []
    ast_actions: List[Dict[str, Any]] = []
    for a in _validate_actions(actions):
        k = a.get("kind")
        if k in fs_kinds:
            fs_actions.append(a)
        elif k in diff_kinds:
            diff_actions.append(a)
        else:
            ast_actions.append(a)
    return fs_actions, diff_actions, ast_actions


# --- Preconditions / Postconditions helpers ---

def _file_exists(path_str: str) -> bool:
    try:
        p = fs_guard.ensure_in_repo_auto(str(path_str))
        return p.exists()
    except Exception:
        return False


def _symbol_present_in_file(symbol: str, file_path: str) -> bool:
    """Detect whether a top-level function or class with the given name exists in file_path."""
    try:
        p = fs_guard.ensure_in_repo_auto(str(file_path))
        if not p.exists():
            return False
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(node, 'name', None) == symbol:
                return True
        return False
    except Exception:
        return False


def _evaluate_preconditions(preconds: List[str] | None) -> Tuple[bool, List[str]]:
    """Return (ok, reasons). Supports tokens like:
    - "file_exists:path/to/file.py"
    - "symbol_absent:NAME@path/to/file.py"
    Unknown tokens are ignored (treated as satisfied).
    """
    if not preconds:
        return True, []
    failures: List[str] = []
    for raw in preconds:
        if not isinstance(raw, str):
            continue
        if raw.startswith("file_exists:"):
            path = raw.split(":", 1)[1]
            if not _file_exists(path):
                failures.append(f"missing {path}")
        elif raw.startswith("symbol_absent:"):
            payload = raw.split(":", 1)[1]
            if "@" in payload:
                name, file_path = payload.split("@", 1)
            else:
                name, file_path = payload, ""
            if file_path and _symbol_present_in_file(name, file_path):
                failures.append(f"symbol {name} present in {file_path}")
        else:
            # Unknown token: ignore
            continue
    return (len(failures) == 0), failures


def _extract_import_module_names(tree: ast.AST) -> List[str]:
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # Skip relative imports (handled as local modules)
            if node.level and not mod:
                # relative like from . import x — treat as satisfied
                continue
            if mod:
                names.append(mod)
    return names


def _module_resolves(name: str, base_file: Path) -> bool:
    # Try to resolve as local module inside repo
    rel_path = REPO_ROOT.joinpath(*name.split(".")).with_suffix(".py")
    pkg_init = REPO_ROOT.joinpath(*name.split("."), "__init__.py")
    if rel_path.exists() or pkg_init.exists():
        return True
    # Fall back to Python environment modules
    with suppress(Exception):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            return True
    return False


def _unresolved_imports_in_file(path: Path) -> List[str]:
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return []  # If we cannot parse, let other checks catch it
    unresolved: List[str] = []
    for mod in _extract_import_module_names(tree):
        if not _module_resolves(mod, path):
            unresolved.append(mod)
    return unresolved


def _tests_collect_ok() -> bool:
    # Lightweight check: all test files under common dirs parse without SyntaxError
    candidates = [REPO_ROOT / "tests", REPO_ROOT / "selfcoder" / "tests"]
    any_found = False
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob("test_*.py"):
            any_found = True
            try:
                src = p.read_text(encoding="utf-8")
                ast.parse(src)
            except Exception:
                return False
    return True if any_found else True  # no tests found -> treat as pass



# --- Failure analysis helpers ---

def _causal_from_post_failures(posts: List[str], failures: List[str]) -> List[Dict[str, Any]]:
    """Build a structured causal chain from postcondition failures.
    For unresolved imports, extract missing module names by file; for generic tokens, record status.
    """
    out: List[Dict[str, Any]] = []
    if not failures:
        return out
    for f in failures:
        # Expected form for unresolved imports: "path: mod1, mod2, ..."
        if ":" in f and any(tok == "no_unresolved_imports" for tok in (posts or [])):
            path, mods = f.split(":", 1)
            missing = [m.strip() for m in mods.split(",") if m.strip()]
            out.append({
                "type": "no_unresolved_imports",
                "file": path.strip(),
                "missing_modules": missing,
            })
        elif f == "tests did not collect" or any(tok == "tests_collect" for tok in (posts or [])):
            out.append({
                "type": "tests_collect",
                "status": "failed",
            })
        else:
            out.append({"type": "unknown_postcondition_failure", "detail": f})
    return out

# --- Transactional application of AST actions ---

def _apply_ast_actions_transactional(paths: List[Path | str], actions: List[Dict[str, Any]], *, dry_run: bool = False) -> Tuple[List[Path], Dict[Path, str]]:
    """Apply AST actions across files as a transaction.
    Returns (modified_paths, backups) where backups maps Path->original_text for rollback.
    """
    modified: List[Path] = []
    backups: Dict[Path, str] = {}
    norm_actions = _validate_actions(actions)
    for path in paths or []:
        p = fs_guard.ensure_in_repo_auto(str(path))
        if not p.exists():
            continue
        skip, reason = _should_skip(p)
        if skip:
            print(f"[SKIP] {p.as_posix()} ({reason})")
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            new_src = apply_actions_via_ast(src, norm_actions)
        except Exception as e:
            print(f"[ERR] AST transform failed for {p.as_posix()}: {e}")
            # Abort transaction
            return [], {}
        # Security preflight
        try:
            result = assess_plan({p.as_posix(): new_src}, fs_guard.infer_repo_root(p))
            if not result.proceed:
                print(f"[SECURITY] BLOCK — {result.reason}")
                for f in result.findings[:20]:
                    print(f" - [{getattr(f, 'severity', '')}] {getattr(f, 'rule_id', '')} {getattr(f, 'filename', '')}:{getattr(f, 'line', 0)} — {getattr(f, 'message', '')}")
                return [], {}
        except Exception:
            pass
        if new_src == src:
            continue
        if dry_run:
            print(f"[DRYRUN] Would write: {p.as_posix()}")
            modified.append(p)
            continue
        # Write with backup for potential rollback
        try:
            backups[p] = src
            p.write_text(new_src, encoding="utf-8")
            print(f"[WRITE] {p.as_posix()}")
            modified.append(p)
        except Exception as e:
            print(f"[ERR] write failed for {p.as_posix()}: {e}")
            return [], {}
    return modified, backups


# --- Preview (unified diff) helpers ---

def _apply_actions_preview(paths: List[Path | str], actions: List[Dict[str, Any]]) -> Dict[Path, Tuple[str, str]]:
    """Compute new source for each path without writing. Returns {Path: (old, new)}."""
    previews: Dict[Path, Tuple[str, str]] = {}
    norm_actions = _validate_actions(actions)
    for path in paths or []:
        p = fs_guard.ensure_in_repo_auto(str(path))
        if not p.exists():
            continue
        skip, _ = _should_skip(p)
        if skip:
            continue
        try:
            old = p.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            ext = p.suffix.lower()
            if ext in {'.js', '.ts', '.tsx'}:
                new = _apply_js_ts(old, norm_actions)
            else:
                new = apply_actions_via_ast(old, norm_actions)
        except Exception:
            # If transform fails, skip preview for this file
            continue
        if new != old:
            previews[p] = (old, new)
    return previews


def _unified_diff_for_file(path: Path, old: str, new: str) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=f"a/{path.as_posix()}", tofile=f"b/{path.as_posix()}")
    return "".join(diff)


def _preview_bundle(paths: List[Path | str], actions: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    previews = _apply_actions_preview(paths, actions)
    for p, (old, new) in previews.items():
        d = _unified_diff_for_file(p, old, new)
        if d:
            chunks.append(d)
    return "\n".join(chunks)


# --- Healers (opt-in, conservative) ---

_ALLOWED_HEALERS = {"format", "isort", "imports"}



def _healer_format(text: str) -> str:
    """Trivial safe formatting: strip trailing whitespace and ensure trailing newline."""
    lines = [ln.rstrip() for ln in text.splitlines()]  # drop trailing spaces
    out = "\n".join(lines)
    if not out.endswith("\n"):
        out += "\n"
    return out

# --- Optional isort healer ---
try:
    import isort  # type: ignore
    def _healer_isort(text: str, file_path: Path | None = None) -> str:
        try:
            return isort.api.sort_code_string(text)  # type: ignore[attr-defined]
        except Exception:
            return text
except Exception:  # pragma: no cover
    def _healer_isort(text: str, file_path: Path | None = None) -> str:
        return text


def _run_healers(paths: List[Path], selected: List[str]) -> Dict[str, Any]:
    """Run selected healers in-place. Returns a report dict with per-file info.
    Only 'format' is implemented internally; 'isort' and 'imports' are placeholders.
    """
    report: Dict[str, Any] = {"applied": [], "skipped": []}
    enabled = [h for h in selected if h in _ALLOWED_HEALERS]
    if not enabled:
        return report
    for p in paths:
        try:
            before = p.read_text(encoding="utf-8")
        except Exception:
            report["skipped"].append({"file": p.as_posix(), "reason": "unreadable"})
            continue
        after = before
        if "format" in enabled:
            after = _healer_format(after)
        if "isort" in enabled:
            after = _healer_isort(after, p)
        if "imports" in enabled:
            pass
        if after != before:
            try:
                p.write_text(after, encoding="utf-8")
                report["applied"].append({"file": p.as_posix(), "healers": enabled})
            except Exception:
                report["skipped"].append({"file": p.as_posix(), "reason": "write_failed"})
        else:
            report["skipped"].append({"file": p.as_posix(), "reason": "no_change"})
    return report


def _apply_fs_actions(fs_actions: List[Dict[str, Any]], default_target: str | None = None, *, dry_run: bool = False) -> List[Path]:
    """Apply simple filesystem actions like `create_file` and `ensure_file`.
    Returns a list of created/ensured Paths.
    """
    created: List[Path] = []
    for a in fs_actions or []:
        kind = a.get("kind")
        payload = dict(a.get("payload") or {})
        if kind == "ensure_test":
            source_path_str = payload.get("source") or default_target
            if not source_path_str:
                continue
            p = fs_guard.ensure_in_repo_auto(str(source_path_str))

            symbol = payload.get("symbol")
            symbol_kind = payload.get("symbol_kind") or payload.get("kind") or "function"

            try:
                tp, scaffold = build_test_scaffold(p.as_posix(), symbol, symbol_kind)
            except TypeError:
                out = build_test_scaffold(p.as_posix(), symbol, symbol_kind)
                if isinstance(out, dict):
                    tp = out.get("path")
                    scaffold = out.get("content", "")
                else:
                    tp, scaffold = out[0], out[1]
            test_path = fs_guard.ensure_in_repo_auto(str(tp))

            skip, reason = _should_skip(test_path)
            if skip:
                print(f"[SKIP] {test_path.as_posix()} ({reason})")
                continue

            if test_path.exists():
                try:
                    existing = test_path.read_text(encoding="utf-8")
                except Exception:
                    existing = ""
                if scaffold.strip() in existing:
                    # Idempotent: scaffold already present
                    created.append(test_path)
                    continue
                new_content = existing.rstrip() + "\n\n" + scaffold
            else:
                new_content = scaffold
                try:
                    test_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

            if dry_run:
                print(f"[DRYRUN] Would write test: {test_path.as_posix()}")
                created.append(test_path)
                continue

            try:
                test_path.write_text(new_content, encoding="utf-8")
                print(f"[WRITE] {test_path.as_posix()}")
                created.append(test_path)
            except Exception as e:
                print(f"[ERR] ensure_test failed for {test_path.as_posix()}: {e}")

            try:
                _log_event(
                    "ensure_test",
                    rationale="orchestrator._apply_fs_actions",
                    source=p.as_posix(),
                    symbol=symbol,
                    symbol_kind=symbol_kind,
                    test_path=test_path.as_posix(),
                    dry_run=bool(dry_run),
                )
            except Exception:
                pass
            continue

        path_str = payload.get("path") or default_target
        if not path_str:
            continue
        p = fs_guard.ensure_in_repo_auto(str(path_str))

        # Respect skip rules
        skip, reason = _should_skip(p)
        if skip:
            print(f"[SKIP] {p.as_posix()} ({reason})")
            continue

        if kind == "ensure_file" and p.exists():
            created.append(p)
            continue

        if p.exists() and not payload.get("overwrite"):
            # No-op: file exists and overwrite not requested
            continue

        content = payload.get("content")
        if content is None:
            content = ""  # default empty content

        if dry_run:
            print(f"[DRYRUN] Would create: {p.as_posix()}")
            created.append(p)
            continue

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(content), encoding="utf-8")
            print(f"[WRITE] {p.as_posix()}")
            created.append(p)
        except Exception as e:
            print(f"[ERR] create_file failed for {p.as_posix()}: {e}")
    try:
        _log_event(
            "fs_actions",
            rationale="orchestrator._apply_fs_actions",
            created=[str(c) for c in created],
            count=len(fs_actions or []),
            dry_run=bool(dry_run),
        )
    except Exception:
        pass
    return created

def run_actions_on_file(path: Path | str, actions: List[Dict[str, Any]], *, dry_run: bool = False) -> bool:
    """
    Read a file, apply AST actions, and write back if changed.
    Returns True if a write occurred (or would have in dry_run), else False.
    """
    p = fs_guard.ensure_in_repo_auto(str(path))
    if not p.exists():
        print(f"[MISS] {p.as_posix()} (does not exist)")
        return False

    skip, reason = _should_skip(p)
    if skip:
        print(f"[SKIP] {p.as_posix()} ({reason})")
        return False

    src = p.read_text(encoding="utf-8")
    actions = _validate_actions(actions)
    if not actions:
        print(f"[NO-OP] {p.as_posix()} (no valid actions)")
        return False

    # Choose transformer based on file extension
    ext = p.suffix.lower()
    if ext in {'.js', '.ts', '.tsx'}:
        new_src = _apply_js_ts(src, actions)
    else:
        new_src = apply_actions_via_ast(src, actions)
    # Security preflight: assess predicted new content
    try:
        result = assess_plan({p.as_posix(): new_src}, fs_guard.infer_repo_root(p))
        if not result.proceed:
            print(f"[SECURITY] BLOCK — {result.reason}")
            for f in result.findings[:20]:
                print(f" - [{f.severity}] {f.rule_id} {f.filename}:{f.line} — {f.message}")
            try:
                _log_event(
                    "security_block",
                    reason=result.reason,
                    findings=[
                        {
                            "severity": getattr(f, "severity", ""),
                            "rule_id": getattr(f, "rule_id", ""),
                            "filename": getattr(f, "filename", ""),
                            "line": getattr(f, "line", 0),
                            "message": getattr(f, "message", ""),
                        }
                        for f in result.findings[:20]
                    ],
                    file=p.as_posix(),
                )
            except Exception:
                pass
            return False
    except Exception:
        # Fail-open: do not block if the gate itself errors
        pass
    if new_src == src:
        print(f"[NO-OP] {p.as_posix()} (no change needed)")
        return False

    if dry_run:
        print(f"[DRYRUN] Would write: {p.as_posix()}")
        return True

    p.write_text(new_src, encoding="utf-8")
    print(f"[WRITE] {p.as_posix()}")
    return True

def run_actions_on_files(paths: List[Path | str], actions: List[Dict[str, Any]], *, dry_run: bool = False) -> List[Path]:
    """
    Apply the given actions to each file in paths.
    Returns a list of Paths for which the file was modified (or would have been in dry_run).
    """
    modified_files: List[Path] = []
    for path in paths:
        p = fs_guard.ensure_in_repo_auto(str(path))
        if run_actions_on_file(p, actions, dry_run=dry_run):
            modified_files.append(p)
    try:
        _log_event(
            "apply_actions",
            rationale="orchestrator.run_actions_on_files",
            files=[str(Path(p)) for p in paths],
            modified=[str(m) for m in modified_files],
            dry_run=bool(dry_run),
            actions_count=len(actions or []),
        )
    except Exception:
        pass
    return modified_files


# Helper to apply a planner-produced plan to one or more files.
def apply_plan(plan: Dict[str, Any], *, dry_run: bool = False, preview: bool = False, healers: List[str] | None = None) -> List[Path]:
    """
    Execute a simple planner-produced plan against one or more files.

    Expected plan schema (minimal):
        {
            "actions": [ { "kind": "...", "payload": {...} }, ... ],
            # either
            "target_file": "path/to/file.py",
            # or
            "files": ["a.py", "b.py", ...]
        }

    Returns the list of Paths that were modified (or would be in dry_run).
    """
    if not isinstance(plan, dict):
        return []

    # Security: validate plan structure (prefer schema, allow legacy fallback), but keep original dict
    try:
        validate_plan(plan)
    except Exception as e:
        acts = plan.get("actions") if isinstance(plan, dict) else None
        legacy_ok = isinstance(acts, list) and any(isinstance(a, dict) and (a.get("kind") or a.get("action")) for a in acts)
        if legacy_ok:
            print(f"[SECURITY] Legacy plan accepted without schema validation: {e}")
        else:
            print(f"[SECURITY] Invalid plan rejected: {e}")
            return []

    raw_actions = plan.get("actions") or []
    if not isinstance(raw_actions, list) or not raw_actions:
        return []

    actions = _validate_actions(raw_actions)
    fs_actions, diff_actions, ast_actions = _split_fs_and_ast_actions(actions)

    # Gather files from either 'files' or single 'target_file'
    files_field = plan.get("files")
    targets: List[Path] = []
    if isinstance(files_field, list) and files_field:
        targets = [Path(str(p)) for p in files_field]
    else:
        tf = plan.get("target_file")
        if isinstance(tf, str) and tf:
            targets = [Path(tf)]

    # Policy: enforce action allow/deny before any writes
    try:
        from selfcoder.security.policy import load_policy as _load_pol, enforce_actions as _enf_acts
        pol = _load_pol(REPO_ROOT)
        ok_pol, why_pol = _enf_acts(actions, pol)
        if not ok_pol:
            print(f"[policy] BLOCK actions: {why_pol}")
            return []
    except Exception:
        pass

    # Optional: preview unified diff for the whole bundle (no writes)
    if preview:
        diff_text = _preview_bundle(targets, _validate_actions(raw_actions)) if targets else ""
        if diff_text:
            print(diff_text)
        try:
            _log_event(
                "apply_plan_preview",
                rationale="orchestrator.apply_plan",
                bundle_id=str(plan.get("bundle_id") or ""),
                targets=[str(t) for t in targets],
                actions_count=len(raw_actions or []),
                diff_bytes=len(diff_text.encode("utf-8")) if diff_text else 0,
            )
        except Exception:
            pass
        # Do not apply anything when preview is requested
        return []

    # Evaluate preconditions, if any
    pc_ok, pc_failures = _evaluate_preconditions(plan.get("preconditions"))
    if not pc_ok:
        print("[PRECONDITION] Blocked:")
        for r in pc_failures:
            print(f" - {r}")
        try:
            _log_event(
                "precondition_block",
                rationale="orchestrator.apply_plan",
                failures=list(pc_failures),
                bundle_id=str(plan.get("bundle_id") or ""),
            )
        except Exception:
            pass
        return []

    # Early postcondition guard: if 'no_unresolved_imports' is requested and targets already
    # have unresolved imports, skip applying edits to avoid churn (keeps file pristine).
    posts_tokens = plan.get("postconditions") or []
    if (not dry_run) and any(tok == "no_unresolved_imports" for tok in posts_tokens):
        try:
            for t in (targets or []):
                if _unresolved_imports_in_file(Path(t)):
                    print("[POSTCONDITION] Unresolved imports present; skipping apply")
                    return []
        except Exception:
            pass

    # Apply file-system actions first (e.g., create_file)
    created = _apply_fs_actions(fs_actions, targets[0].as_posix() if targets else None, dry_run=dry_run)

    # If no explicit targets were given but FS actions created files, use them
    if not targets and created:
        targets = list(created)

    modified: List[Path] = []
    backups: Dict[Path, str] = {}
    
    # Apply unified diff actions next (before AST) to allow text-only fixes
    if diff_actions:
        for a in diff_actions:
            payload = dict(a.get("payload") or {})
            diff_text = payload.get("diff")
            if not diff_text and payload.get("diff_file"):
                try:
                    p = fs_guard.ensure_in_repo(REPO_ROOT, str(payload.get("diff_file")))
                    diff_text = Path(p).read_text(encoding="utf-8")
                except Exception:
                    diff_text = None
            if not diff_text:
                print("[diff] missing diff payload; skipping")
                continue
            if _preview_unified_diff is None:
                print("[diff] unified diff previewer unavailable; skipping")
                continue
            try:
                previews, perr = _preview_unified_diff(diff_text, REPO_ROOT)
            except Exception as _e:
                previews, perr = {}, ["parse_failed"]
            for e in perr or []:
                print(f"[diff] {e}")
            if not previews:
                continue
            predicted = {p.as_posix(): new for p, (_old, new) in previews.items()}
            # Policy: enforce path/limits for predicted changes
            try:
                from selfcoder.security.policy import load_policy as _load_pol, enforce_paths as _enf_paths, enforce_limits as _enf_limits
                pol = _load_pol(REPO_ROOT)
                rels = []
                for k in predicted.keys():
                    p = Path(k)
                    try:
                        rel = p.relative_to(REPO_ROOT)
                    except Exception:
                        rel = p
                    rels.append(rel)
                okp, why, viol = _enf_paths(rels, pol)
                if not okp:
                    print(f"[policy] BLOCK paths: {why} — {[v.as_posix() for v in viol]}")
                    continue
                okL, whyL = _enf_limits(predicted, pol)
                if not okL:
                    print(f"[policy] BLOCK limits: {whyL}")
                    continue
            except Exception:
                pass
            # Security preflight for changed files
            try:
                result = assess_plan(predicted, REPO_ROOT, plan_actions=actions)
                if not result.proceed:
                    print("[SECURITY] BLOCK (diff) — " + str(result.reason))
                    continue
            except Exception:
                pass
            # Reviewer gating (preview only)
            blocked = False
            try:
                rep = _review_predicted(predicted, REPO_ROOT)
                # Auto profile hint (non-invasive)
                try:
                    if _decide_profile:
                        style_hints = sum(len(v or []) for v in (rep.get('style') or {}).values())
                        sec_count = int(((rep.get('security') or {}).get('findings') and len((rep.get('security') or {}).get('findings'))) or 0)
                        files_count = len(previews)
                        delta_bytes = 0
                        for _p, (old, new) in previews.items():
                            try:
                                delta_bytes += (len(new) - len(old))
                            except Exception:
                                pass
                        kinds_ast_only = False
                        has_rename = any((a.get('kind') or a.get('action')) == 'rename_symbol' for a in (actions or []))
                        dec = _decide_profile('apply_plan', preview=predicted, signals={
                            'security_findings': sec_count,
                            'style_hints': style_hints,
                            'files_count': files_count,
                            'delta_bytes': delta_bytes,
                            'kinds_ast_only': kinds_ast_only,
                            'has_rename': has_rename,
                        })
                        if dec and dec.name:
                            print(f"[profile] hint: {dec.name} ({dec.why})")
                except Exception:
                    pass
                strict = (os.getenv('NERION_REVIEW_STRICT') or '').strip().lower() in {'1','true','yes','on'}
                try:
                    from selfcoder.config import get_policy as _get_policy
                    _pol = _get_policy()
                except Exception:
                    _pol = 'balanced'
                if strict or _pol == 'safe':
                    if (not rep.get('security', {}).get('proceed', True)):
                        print("[review] strict/safe: blocking diff due to security findings")
                        blocked = True
                # Optional style thresholds
                try:
                    style_max = os.getenv('NERION_REVIEW_STYLE_MAX')
                    if style_max is not None and str(style_max).strip() != '':
                        try:
                            limit = int(style_max)
                        except Exception:
                            limit = -1
                        if limit >= 0:
                            total_hints = sum(len(v or []) for v in (rep.get('style') or {}).values())
                            if total_hints > limit:
                                print(f"[review] blocking diff: style_hints {total_hints} > limit {limit}")
                                blocked = True
                except Exception:
                    pass
            except Exception:
                pass
            if blocked:
                continue
            # Write or dry-run (apply profile scoped if resolver recommends)
            _profile_scope = None
            try:
                if _decide_profile:
                    dec2 = _decide_profile('apply_plan', preview=predicted, signals={'security_findings': 0, 'files_count': len(previews), 'delta_bytes': 0, 'kinds_ast_only': False})
                    from selfcoder.policy.profile_resolver import apply_env_scoped as _apply_env_scoped  # type: ignore
                    _profile_scope = _apply_env_scoped(dec2)
            except Exception:
                _profile_scope = None
            try:
                for p, (old, new) in previews.items():
                    if new == old:
                        continue
                    if dry_run:
                        print(f"[DRYRUN] Would patch: {p.as_posix()}")
                        modified.append(p)
                        continue
                    try:
                        backups[p] = old
                        p.write_text(new, encoding="utf-8")
                        print(f"[WRITE] {p.as_posix()}")
                        modified.append(p)
                    except Exception as e:
                        print(f"[ERR] diff write failed for {p.as_posix()}: {e}")
            finally:
                if _profile_scope and hasattr(_profile_scope, '__exit__'):
                    try:
                        _profile_scope.__exit__(None, None, None)
                    except Exception:
                        pass
    if ast_actions and targets:
        # Pre-apply Reviewer (preview only; do not write yet)
        try:
            previews = _apply_actions_preview(targets, ast_actions)
            predicted = {p.as_posix(): new for p, (_old, new) in previews.items()}
            if predicted:
                rep = _review_predicted(predicted, REPO_ROOT)
                print("[review]\n" + _fmt_review(rep))
                # Auto profile hint (non-invasive)
                try:
                    if _decide_profile:
                        style_hints = sum(len(v or []) for v in (rep.get('style') or {}).values())
                        sec_count = int(((rep.get('security') or {}).get('findings') and len((rep.get('security') or {}).get('findings'))) or 0)
                        files_count = len(previews)
                        delta_bytes = 0
                        for _p, (old, new) in previews.items():
                            try:
                                delta_bytes += (len(new) - len(old))
                            except Exception:
                                pass
                        kinds_ast_only = True
                        has_rename = any((a.get('kind') or a.get('action')) == 'rename_symbol' for a in (ast_actions or []))
                        dec = _decide_profile('apply_plan', preview=predicted, signals={
                            'security_findings': sec_count,
                            'style_hints': style_hints,
                            'files_count': files_count,
                            'delta_bytes': delta_bytes,
                            'kinds_ast_only': kinds_ast_only,
                            'has_rename': has_rename,
                        })
                        if dec and dec.name:
                            print(f"[profile] hint: {dec.name} ({dec.why})")
                except Exception:
                    pass
                strict = (os.getenv('NERION_REVIEW_STRICT') or '').strip().lower() in {'1','true','yes','on'}
                # Policy-profiles: default behaviors if env thresholds are unset
                try:
                    from selfcoder.config import get_policy as _get_policy
                    _pol = _get_policy()
                except Exception:
                    _pol = 'balanced'
                if strict or _pol == 'safe':
                    # In safe mode, block on any security finding and enforce style_max=0 by default
                    if (not rep.get('security', {}).get('proceed', True)):
                        print("[review] strict/safe: blocking due to security findings")
                        return []
                    if (os.getenv('NERION_REVIEW_STYLE_MAX') or '').strip() == '':
                        total_hints = sum(len(v or []) for v in (rep.get('style') or {}).values())
                        if total_hints > 0:
                            print("[review] safe: blocking due to style hints > 0")
                            return []
                if strict and (not rep.get('security', {}).get('proceed', True)):
                    print("[review] strict mode: blocking apply due to security findings")
                    return []
                # Optional style/external gating via env thresholds
                try:
                    style_max = os.getenv('NERION_REVIEW_STYLE_MAX')
                    if style_max is not None and str(style_max).strip() != '':
                        try:
                            limit = int(style_max)
                        except Exception:
                            limit = -1
                        if limit >= 0:
                            total_hints = sum(len(v or []) for v in (rep.get('style') or {}).values())
                            if total_hints > limit:
                                print(f"[review] blocking: style_hints {total_hints} > limit {limit}")
                                return []
                    ext = rep.get('external') or {}
                    def _exceeds(name: str, env_key: str) -> bool:
                        v = os.getenv(env_key)
                        if v is None or str(v).strip() == '':
                            return False
                        try:
                            limit = int(v)
                        except Exception:
                            return False
                        count = int(((ext.get(name) or {}).get('count')) or 0)
                        return (limit >= 0) and (count > limit)
                    if _exceeds('ruff', 'NERION_REVIEW_RUFF_MAX'):
                        print("[review] blocking: ruff issues exceed NERION_REVIEW_RUFF_MAX")
                        return []
                    if _exceeds('pydocstyle', 'NERION_REVIEW_PYDOCSTYLE_MAX'):
                        print("[review] blocking: pydocstyle issues exceed NERION_REVIEW_PYDOCSTYLE_MAX")
                        return []
                    if _exceeds('mypy', 'NERION_REVIEW_MYPY_MAX'):
                        print("[review] blocking: mypy issues exceed NERION_REVIEW_MYPY_MAX")
                        return []
                except Exception:
                    # Never block on threshold parsing errors
                    pass
        except Exception:
            pass
        # Apply AST actions under a scoped profile if recommended
        _profile_scope = None
        try:
            if _decide_profile:
                dec2 = _decide_profile('apply_plan', preview={}, signals={'security_findings': 0, 'files_count': len(targets), 'delta_bytes': 0, 'kinds_ast_only': True})
                from selfcoder.policy.profile_resolver import apply_env_scoped as _apply_env_scoped  # type: ignore
                _profile_scope = _apply_env_scoped(dec2)
        except Exception:
            _profile_scope = None
        try:
            modified, backups = _apply_ast_actions_transactional(targets, ast_actions, dry_run=dry_run)
        finally:
            if _profile_scope and hasattr(_profile_scope, '__exit__'):
                try:
                    _profile_scope.__exit__(None, None, None)
                except Exception:
                    pass
    else:
        # No AST actions to apply; preserve any modifications from diff/fs stages
        pass

    # Optional: run conservative healers before postconditions, included in rollback scope
    healer_report = None
    if not dry_run and healers:
        # If no AST edits happened, allow explicit healers to run over the targets
        candidates = list({*(modified or []), *(created or []), *(targets or [])})
        if candidates:
            # Ensure any healer writes are covered by backups
            for p in candidates:
                if p not in backups:
                    try:
                        backups[p] = p.read_text(encoding="utf-8")
                    except Exception:
                        pass
            healer_report = _run_healers(candidates, list(healers))

    # Evaluate postconditions over touched files
    touched = list({*(modified or []), *(created or [])})
    posts = plan.get("postconditions") or []
    # Allow disabling postconditions (e.g., in environments without deps installed)
    if os.getenv("NERION_DISABLE_POSTCONDS"):
        posts = []
    post_failures: List[str] = []
    if posts and not dry_run:
        # Consider also original targets for checks (in case no writes occurred)
        paths_for_post = list({*touched, *[Path(t) for t in (targets or [])]})
        for token in posts:
            if token == "no_unresolved_imports":
                for p in paths_for_post:
                    unresolved = _unresolved_imports_in_file(p)
                    if unresolved:
                        post_failures.append(f"{p.as_posix()}: {', '.join(unresolved)}")
            elif token == "tests_collect":
                if not _tests_collect_ok():
                    post_failures.append("tests did not collect")
            elif token in {"eslint_clean", "tsc_ok"}:
                try:
                    from selfcoder.security import extlinters as _ext
                    # Convert to repo-relative paths when possible
                    root = Path.cwd()
                    rels = []
                    for p in paths_for_post:
                        try:
                            rel = p.relative_to(root)
                        except Exception:
                            rel = p
                        rels.append(rel)
                    findings = _ext.run_on_dir(root, rels)
                    if token == "eslint_clean" and any(f.get('tool') == 'eslint' for f in findings or []):
                        post_failures.append("eslint violations present")
                    if token == "tsc_ok" and any(f.get('tool') == 'tsc' for f in findings or []):
                        post_failures.append("tsc reported type errors")
                except Exception:
                    # If tools unavailable, do not fail postconditions
                    pass
            else:
                # Unknown token -> ignore
                continue
        if post_failures:
            print("[POSTCONDITION] Failing; rolling back edits")
            # Roll back AST edits only (robust to path normalization)
            try:
                bk_map = {str(Path(k).resolve()): v for k, v in backups.items()}
            except Exception:
                bk_map = {str(k): v for k, v in backups.items()}
            # Restore exactly the files we backed up (robust to path representation)
            for key, text in bk_map.items():
                with suppress(Exception):
                    Path(key).write_text(text, encoding="utf-8")
            # Log event
            try:
                _log_event(
                    "postcondition_rollback",
                    rationale="orchestrator.apply_plan",
                    failures=list(post_failures),
                    bundle_id=str(plan.get("bundle_id") or ""),
                    touched=[str(t) for t in touched],
                )
            except Exception:
                pass
            # Build and save a failure artifact with a causal chain for easier diagnosis
            try:
                causal = _causal_from_post_failures(posts, post_failures)
                score, why = score_plan(plan, None)
                artifact = PlanArtifact(
                    ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                    origin="orchestrator.apply_plan",
                    score=score,
                    rationale=why,
                    plan=plan,
                    files_touched=[str(x) for x in (touched or [])],
                    sim=None,
                    meta={
                        "dry_run": bool(dry_run),
                        "bundle_id": plan.get("bundle_id"),
                        "has_pre": bool(plan.get("preconditions")),
                        "has_post": bool(plan.get("postconditions")),
                        "preview": bool(preview),
                        "healers": healer_report or None,
                        "actions_count": len(actions or []),
                        "targets_count": len(targets or []),
                        "modified_count": len(modified or []),
                        "created_count": len(created or []),
                        "touched_count": len(touched or []),
                        "postconditions": {
                            "checked": list(posts or []),
                            "failures": list(post_failures or []),
                            "status": "failed",
                        },
                        "causal_chain": causal,
                    },
                )
                save_artifact(artifact)
            except Exception:
                pass
            return []

    # If we actually changed files (and not just preview), reload plugins so newly added/updated tools are live
    if (modified or created) and not dry_run:
        try:
            from plugins.loader import reload_plugins_auto as _reload_plugins_auto
            _reload_plugins_auto()
        except Exception:
            pass
    try:
        _log_event(
            "apply_plan",
            rationale="orchestrator.apply_plan",
            dry_run=bool(dry_run),
            targets=[str(t) for t in targets],
            modified=[str(m) for m in modified],
            created=[str(c) for c in created],
            actions_count=len(actions or []),
        )
    except Exception:
        pass

    # --- Test impact guidance (opt-in via NERION_TEST_IMPACT=1) ---
    try:
        _TI = (os.getenv('NERION_TEST_IMPACT') or '').strip().lower() in {'1','true','yes','on'}
    except Exception:
        _TI = False
    if (not dry_run) and _TI:
        try:
            impacted = _predict_impacted_tests(modified or created)
            if not impacted:
                # Scaffold a minimal generated test for the first target
                tgt = str(targets[0]) if targets else None
                if tgt:
                    try:
                        code = _testgen.generate_tests_for_plan(plan, tgt)
                        out_dir = Path('selfcoder/tests/generated')
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"test_auto_{Path(tgt).stem}.py"
                        _testgen.write_test_file(code, out_path)
                        print(f"[impact] scaffolded: {out_path}")
                        impacted = [out_path]
                    except Exception:
                        impacted = []
            # Optional Tester expansion (NERION_TESTER=1)
            try:
                _TE = (os.getenv('NERION_TESTER') or '').strip().lower() in {'1','true','yes','on'}
            except Exception:
                _TE = False
            if _TE:
                tgt = str(targets[0]) if targets else None
                if tgt:
                    try:
                        edge_code = _expand_tests(plan, tgt)
                        out_dir = Path('selfcoder/tests/generated')
                        out_dir.mkdir(parents=True, exist_ok=True)
                        edge_path = out_dir / f"test_edge_{Path(tgt).stem}.py"
                        _testgen.write_test_file(edge_code, edge_path)
                        print(f"[impact] tester expanded: {edge_path}")
                        impacted = ([edge_path] + (impacted or [])) if impacted else [edge_path]
                    except Exception:
                        pass
            if impacted:
                print(f"[impact] running {len(impacted)} impacted test file(s)…")
                rc_imp = _testgen.run_pytest_on_paths(impacted)
                print(f"[impact] impacted tests exit code: {rc_imp}")
            smoke_dirs = [p for p in [Path('tests'), Path('selfcoder/tests')] if p.exists()]
            if smoke_dirs:
                print("[impact] running smoke suite…")
                rc_smoke = _testgen.run_pytest_on_paths(smoke_dirs)
                print(f"[impact] smoke exit code: {rc_smoke}")
        except Exception as _e:
            print(f"[impact] skipped due to error: {_e}")
    # --- Score plan and log artifact ---
    try:
        score, why = score_plan(plan, None)
        artifact = PlanArtifact(
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            origin="orchestrator.apply_plan",
            score=score,
            rationale=why,
            plan=plan,
            files_touched=[str(x) for x in (modified or created or [])],
            sim=None,
            meta={
                "dry_run": bool(dry_run),
                "bundle_id": plan.get("bundle_id"),
                "has_pre": bool(plan.get("preconditions")),
                "has_post": bool(plan.get("postconditions")),
                "preview": bool(preview),
                "healers": healer_report or None,
                # Counts and summaries
                "actions_count": len(actions or []),
                "targets_count": len(targets or []),
                "modified_count": len(modified or []),
                "created_count": len(created or []),
                "touched_count": len(touched or []),
                "postconditions": {
                    "checked": list(posts or []),
                    "failures": list(post_failures or []),
                    "status": "ok" if not (posts and post_failures) else "failed",
                },
            },
        )
        save_artifact(artifact)
    except Exception:
        pass
    return modified or created

__all__ = [
    "apply_actions_via_ast",
    "_TRANSFORMERS",
    "run_actions_on_file",
    "run_actions_on_files",
    "apply_plan",
    # Legacy exports for backward compatibility
    "run_ast_actions",
    "dry_run_orchestrate",
    "Orchestrator",
    "OrchestrateResult",
]


# Backwards compatibility exports (functional shims)
@dataclass(frozen=True)
class OrchestrateResult:
    modified_files: List[Path]
    in_file_edits: int
    crossfile_edits: int

def run_ast_actions(src: str, actions: List[Dict[str, Any]] | None = None) -> str:
    """Legacy alias for apply_actions_via_ast (kept for older callers)."""
    return apply_actions_via_ast(src, actions or [])


def dry_run_orchestrate(src: str, actions: List[Dict[str, Any]] | None = None) -> str:
    """Legacy dry-run helper that simply calls run_ast_actions."""
    return run_ast_actions(src, actions)


class Orchestrator:
    """Minimal orchestrator preserved for backward compatibility.

    Applies in-file AST actions to files and (optionally) writes the results
    when `preview` is False. Cross-file operations are not implemented in this
    shim; callers requiring them should use newer utilities.
    """

    def __init__(self, actions: List[Dict[str, Any]] | None = None) -> None:
        self.actions: List[Dict[str, Any]] = list(actions or [])

    @staticmethod
    def apply_plan(plan: Dict[str, Any], *, dry_run: bool = False) -> List[Path]:
        """Compatibility shim: delegate to module-level apply_plan."""
        return apply_plan(plan, dry_run=dry_run)

    def apply_in_file(self, source: str) -> Tuple[str, int]:
        new_src = apply_actions_via_ast(source, self.actions)
        known = sum(
            1 for a in self.actions if isinstance(a, dict) and a.get("kind") in _TRANSFORMERS
        )
        return new_src, known

    def orchestrate_files(
        self,
        files: List[Path],
        root: Path | None = None,
        preview: bool = True,
    ) -> OrchestrateResult:
        modified: List[Path] = []
        in_file_count = 0

        for p in files:
            jp = fs_guard.ensure_in_repo_auto(str(p))
            try:
                text = jp.read_text(encoding="utf-8")
            except Exception:
                continue
            new_text, n = self.apply_in_file(text)
            in_file_count += int(n)
            if new_text != text:
                try:
                    if not preview:
                        jp.write_text(new_text, encoding="utf-8")
                    modified.append(jp)
                except Exception:
                    # Ignore write errors at this layer
                    pass

        # Cross-file not supported in this shim; report zero.
        return OrchestrateResult(
            modified_files=modified,
            in_file_edits=in_file_count,
            crossfile_edits=0,
        )



def _normalize_actions_for_apply(actions):
    norm = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        a = dict(a)
        if 'kind' not in a and 'action' in a:
            a['kind'] = a.pop('action')
        payload = dict(a.get('payload') or {})
        if 'doc' not in payload and 'docstring' in payload:
            payload['doc'] = payload.pop('docstring')
        a['payload'] = payload
        norm.append(a)
    return norm
def run_batch_actions_on_files(files, actions, dry_run=False):
    from pathlib import Path
    changed = []
    for f in files or []:
        fp = Path(f)
        try:
            if run_actions_on_file(fp, _normalize_actions_for_apply(actions), dry_run=bool(dry_run)):
                changed.append(str(fp))
        except Exception:
            pass
    return changed

# ------------------- Test Impact Helpers -------------------
def _predict_impacted_tests(modified_files: List[Path]) -> List[Path]:
    """Return a list of likely impacted test files given modified sources.

    Heuristics:
      - match by stem in test filenames under tests/ and selfcoder/tests/
      - include direct sibling tests (same directory test_*.py) if present
    """
    roots = [Path('tests'), Path('selfcoder/tests')]
    roots = [r for r in roots if r.exists()]
    if not roots or not modified_files:
        return []
    stems = {Path(p).stem for p in modified_files}
    out: List[Path] = []
    seen = set()
    for r in roots:
        try:
            for p in r.rglob('test_*.py'):
                name = p.name.lower()
                if any(s in name for s in stems):
                    key = p.resolve()
                    if key not in seen:
                        seen.add(key)
                        out.append(p)
        except Exception:
            continue
    for m in modified_files:
        try:
            for sib in m.parent.glob('test_*.py'):
                key = sib.resolve()
                if key not in seen:
                    seen.add(key)
                    out.append(sib)
        except Exception:
            pass
    # Heuristic 3: import-graph based (tests that import modified modules)
    try:
        if _build_import_graph is not None:
            for r in roots:
                ig = _build_import_graph(r)
                # Precompute candidate module paths for modified files
                mod_paths = {Path(m).resolve() for m in modified_files}
                for test_file, mods in ig.items():
                    for m in mods:
                        try:
                            cand = Path('.').resolve().joinpath(*m.split('.')).with_suffix('.py')
                        except Exception:
                            cand = None
                        if cand and cand.resolve() in mod_paths:
                            key = test_file.resolve()
                            if key not in seen:
                                seen.add(key)
                                out.append(test_file)
    except Exception:
        pass
    # Heuristic 4: coverage-context mapping (opt-in via NERION_COV_CONTEXT=1)
    try:
        _CC = (os.getenv('NERION_COV_CONTEXT') or '').strip().lower() in {'1','true','yes','on'}
    except Exception:
        _CC = False
    if _CC and _covu is not None:
        try:
            cov = _covu.run_pytest_with_coverage(pytest_args=['-q'], cov_context='test')
            # Best-effort parse of context map
            ctx_map = {}
            files_meta = (cov.get('files') or {}) if isinstance(cov, dict) else {}
            for fname, meta in files_meta.items():
                ctxs = meta.get('contexts') if isinstance(meta, dict) else None
                if isinstance(ctxs, dict):
                    for ctx_id, lines in ctxs.items():
                        if not isinstance(ctx_id, str):
                            continue
                        ctx_map.setdefault(ctx_id, set()).add(fname)
            if ctx_map:
                mod_set = {str(Path(m).resolve()) for m in modified_files}
                test_files = set()
                for ctx_id, files in ctx_map.items():
                    try:
                        tfile = ctx_id.split('::', 1)[0]
                    except Exception:
                        tfile = ctx_id
                    # Normalize file paths for comparison
                    for f in list(files):
                        try:
                            p = Path(f).resolve()
                        except Exception:
                            continue
                        if str(p) in mod_set:
                            tf = Path(tfile)
                            if tf.exists() and tf.name.startswith('test_'):
                                test_files.add(tf)
                for tf in sorted(test_files):
                    key = tf.resolve()
                    if key not in seen:
                        seen.add(key)
                        out.append(tf)
        except Exception:
            pass
    return out
