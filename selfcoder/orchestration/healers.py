"""
Code healer utilities for orchestration.

This module provides utilities for applying safe code formatting
and import sorting to files.
"""
from pathlib import Path
from typing import Any, Dict, List


# Allowed healers for safety
ALLOWED_HEALERS = {"format", "isort", "imports"}


def healer_format(text: str) -> str:
    """Trivial safe formatting: strip trailing whitespace and ensure trailing newline.

    Args:
        text: Source code to format

    Returns:
        Formatted source code
    """
    lines = [ln.rstrip() for ln in text.splitlines()]  # drop trailing spaces
    out = "\n".join(lines)
    if not out.endswith("\n"):
        out += "\n"
    return out


# Optional isort healer
try:
    import isort  # type: ignore
    def healer_isort(text: str, file_path: Path | None = None) -> str:
        """Sort imports using isort.

        Args:
            text: Source code to sort imports
            file_path: Optional file path for isort configuration

        Returns:
            Source code with sorted imports
        """
        try:
            return isort.api.sort_code_string(text)  # type: ignore[attr-defined]
        except Exception:
            return text
except Exception:  # pragma: no cover
    def healer_isort(text: str, file_path: Path | None = None) -> str:
        """Placeholder isort healer when isort is not available.

        Args:
            text: Source code
            file_path: Optional file path

        Returns:
            Unchanged source code
        """
        return text


def run_healers(paths: List[Path], selected: List[str]) -> Dict[str, Any]:
    """Run selected healers in-place.

    Only 'format' is implemented internally; 'isort' and 'imports' are placeholders.

    Args:
        paths: List of file paths to heal
        selected: List of healer names to run

    Returns:
        Report dictionary with per-file info
    """
    report: Dict[str, Any] = {"applied": [], "skipped": []}
    enabled = [h for h in selected if h in ALLOWED_HEALERS]
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
            after = healer_format(after)
        if "isort" in enabled:
            after = healer_isort(after, p)
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
