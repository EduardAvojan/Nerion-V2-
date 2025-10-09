"""
Validation utilities for orchestration.

This module provides validation functions for actions, files, and
pre/post-conditions in the orchestration pipeline.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple
from ops.security import fs_guard


def should_skip(file_path: Path) -> Tuple[bool, str]:
    """Best-effort bridge to git_ops.should_skip; fallback to no-skip.

    Args:
        file_path: Path to check

    Returns:
        Tuple of (should_skip: bool, reason: str)
    """
    try:
        from selfcoder.vcs.git_ops import should_skip  # type: ignore
        return should_skip(file_path)
    except Exception:
        return (False, "")


def validate_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize actions.

    Accept 'action' as an alias for 'kind', ensure 'kind' is present and
    non-empty, and normalize payload.

    Args:
        actions: List of action dictionaries

    Returns:
        List of validated and normalized actions
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


def split_fs_and_ast_actions(actions: List[Dict[str, Any]]):
    """Split actions into filesystem-level, diff-level, and AST-level.

    FS-level supports: create_file, ensure_file, ensure_test.
    DIFF-level supports: apply_unified_diff (payload: {diff: str} or {diff_file: str}).

    Args:
        actions: List of action dictionaries

    Returns:
        Tuple of (fs_actions, diff_actions, ast_actions)
    """
    fs_kinds = {"create_file", "ensure_file", "ensure_test"}
    diff_kinds = {"apply_unified_diff"}
    fs_actions: List[Dict[str, Any]] = []
    diff_actions: List[Dict[str, Any]] = []
    ast_actions: List[Dict[str, Any]] = []
    for a in validate_actions(actions):
        k = a.get("kind")
        if k in fs_kinds:
            fs_actions.append(a)
        elif k in diff_kinds:
            diff_actions.append(a)
        else:
            ast_actions.append(a)
    return fs_actions, diff_actions, ast_actions


def file_exists(path_str: str) -> bool:
    """Check if a file exists within the repository.

    Args:
        path_str: Path string to check

    Returns:
        True if file exists, False otherwise
    """
    try:
        p = fs_guard.ensure_in_repo_auto(str(path_str))
        return p.exists()
    except Exception:
        return False
