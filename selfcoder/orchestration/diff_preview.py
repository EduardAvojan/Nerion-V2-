"""
Diff and preview utilities for orchestration.

This module provides utilities for generating unified diffs and
previewing code changes before applying them.
"""
import difflib
from pathlib import Path
from typing import Any, Dict, List


def unified_diff_for_file(path: Path, old: str, new: str) -> str:
    """Generate a unified diff for a single file.

    Args:
        path: File path for diff header
        old: Old source code
        new: New source code

    Returns:
        Unified diff as a string
    """
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=f"a/{path.as_posix()}", tofile=f"b/{path.as_posix()}")
    return "".join(diff)


def preview_bundle(paths: List[Path | str], actions: List[Dict[str, Any]]) -> str:
    """Generate a unified diff bundle for multiple files.

    Args:
        paths: List of file paths to preview
        actions: List of action dictionaries

    Returns:
        Combined unified diff for all modified files
    """
    from .ast_transformer import apply_actions_preview

    chunks: List[str] = []
    previews = apply_actions_preview(paths, actions)
    for p, (old, new) in previews.items():
        d = unified_diff_for_file(p, old, new)
        if d:
            chunks.append(d)
    return "\n".join(chunks)
