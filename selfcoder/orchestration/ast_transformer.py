"""
AST transformation utilities for orchestration.

This module provides utilities for applying AST-level code transformations
across multiple files with transaction support and security checking.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple
from ops.security import fs_guard
from selfcoder.actions.transformers import apply_actions_via_ast
from selfcoder.security.gate import assess_plan

# Import for JS/TS support
try:
    from selfcoder.actions.js_ts import apply_actions_js_ts as _apply_js_ts  # type: ignore
except Exception:  # pragma: no cover
    def _apply_js_ts(src: str, actions):  # type: ignore
        return src


def apply_ast_actions_transactional(
    paths: List[Path | str],
    actions: List[Dict[str, Any]],
    *,
    dry_run: bool = False
) -> Tuple[List[Path], Dict[Path, str]]:
    """Apply AST actions across files as a transaction.

    Args:
        paths: List of file paths to transform
        actions: List of action dictionaries
        dry_run: If True, don't actually write files

    Returns:
        Tuple of (modified_paths, backups) where backups maps Path->original_text for rollback
    """
    from .validators import validate_actions, should_skip

    modified: List[Path] = []
    backups: Dict[Path, str] = {}
    norm_actions = validate_actions(actions)
    for path in paths or []:
        p = fs_guard.ensure_in_repo_auto(str(path))
        if not p.exists():
            continue
        skip, reason = should_skip(p)
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


def apply_actions_preview(
    paths: List[Path | str],
    actions: List[Dict[str, Any]]
) -> Dict[Path, Tuple[str, str]]:
    """Compute new source for each path without writing.

    Args:
        paths: List of file paths to preview
        actions: List of action dictionaries

    Returns:
        Dictionary mapping Path to (old_source, new_source) tuples
    """
    from .validators import validate_actions, should_skip

    previews: Dict[Path, Tuple[str, str]] = {}
    norm_actions = validate_actions(actions)
    for path in paths or []:
        p = fs_guard.ensure_in_repo_auto(str(path))
        if not p.exists():
            continue
        skip, _ = should_skip(p)
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


def run_ast_actions(src: str, actions: List[Dict[str, Any]] | None = None) -> str:
    """Legacy alias for apply_actions_via_ast (kept for older callers).

    Args:
        src: Source code to transform
        actions: List of action dictionaries

    Returns:
        Transformed source code
    """
    return apply_actions_via_ast(src, actions or [])


def dry_run_orchestrate(src: str, actions: List[Dict[str, Any]] | None = None) -> str:
    """Legacy dry-run helper that simply calls run_ast_actions.

    Args:
        src: Source code to transform
        actions: List of action dictionaries

    Returns:
        Transformed source code
    """
    return run_ast_actions(src, actions)
