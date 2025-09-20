from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Iterable
import os
import re
from fnmatch import fnmatch

"""Cross-file rename utilities for Nerion self-coder.

This module provides a tiny, dependency-free helper that scans a project tree
and applies safe, text-based renames of modules and (optionally) attributes
across files. It intentionally avoids importing the target project code to
prevent side effects and circular imports.
"""


# ----------------------------- Data structures ----------------------------- #

@dataclass(frozen=True)
class RenameSpec:
    """Describe a rename operation."""
    old_module: str
    new_module: str
    old_attr: str | None = None
    new_attr: str | None = None


# ------------------------------- File walkers ------------------------------ #

_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "_archive",
    "backups",
}

def _to_paths(items: Iterable[Union[str, Path]]) -> List[Path]:
    out: List[Path] = []
    for it in items:
        p = Path(it) if not isinstance(it, Path) else it
        out.append(p)
    return out


def _iter_pyfiles(
    root: Path,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
):
    """Yield Python source files under *root* while skipping tooling caches."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if exclude and any(fnmatch(rel, pat) for pat in exclude):
                continue
            if include and not any(fnmatch(rel, pat) for pat in include):
                continue
            yield p


# ------------------------------- Edit helpers ------------------------------ #

def _apply_to_text(text: str, specs: Sequence[RenameSpec]) -> Tuple[str, int]:
    """Return (updated_text, edits_count) after applying *specs* to *text*."""
    total_edits = 0

    for spec in specs:
        original_before = text  # snapshot for context-sensitive checks
        # First, handle module renames (imports + dotted references) so later attr rules can match both forms.
        # Rule 1: `from old_module import ...` -> `from new_module import ...`
        pattern_from = re.compile(
            rf"(^\s*from\s+){re.escape(spec.old_module)}(\s+import\b)", re.M
        )
        text, n = pattern_from.subn(rf"\1{spec.new_module}\2", text)
        total_edits += n

        # Rule 2: `import old_module` -> `import new_module`
        pattern_import = re.compile(
            rf"(^\s*import\s+){re.escape(spec.old_module)}(\b)", re.M
        )
        text, n = pattern_import.subn(rf"\1{spec.new_module}\2", text)
        total_edits += n

        # Rule 3: Dotted module references: `old_module.` -> `new_module.`
        pattern_module_dot = re.compile(rf"\b{re.escape(spec.old_module)}\.")
        text, n = pattern_module_dot.subn(rf"{spec.new_module}.", text)
        total_edits += n

        # If an attribute rename is requested, handle import-line attr and qualified/standalone uses.
        if spec.old_attr:
            new_attr = spec.new_attr or spec.old_attr
            # After module rename above, update import tails for either old or new module forms.
            for mod in (spec.old_module, spec.new_module):
                if not mod:
                    continue
                pattern_from_attr = re.compile(
                    rf"(^\s*from\s+){re.escape(mod)}(\s+import\s+)([^\n#]+)",
                    re.M,
                )
                def _sub_from_attr(m: re.Match) -> str:
                    head, kw, tail = m.group(1), m.group(2), m.group(3)
                    new_tail = re.sub(rf"(?<!\w){re.escape(spec.old_attr)}(?!\w)", new_attr, tail)
                    return f"{head}{mod}{kw}{new_tail}"
                text, n = pattern_from_attr.subn(_sub_from_attr, text)
                total_edits += n

            # Qualified dotted uses: old_module.old_attr or new_module.old_attr -> new_module.new_attr
            for mod in (spec.old_module, spec.new_module):
                pattern_qual = re.compile(rf"\b{re.escape(mod)}\.{re.escape(spec.old_attr)}\b")
                text, n = pattern_qual.subn(f"{spec.new_module}.{new_attr}", text)
                total_edits += n

            # Standalone symbol rename only when imported from either form
            imported_attr = re.search(
                rf"^\s*from\s+{re.escape(spec.old_module)}\s+import\s+.*\b{re.escape(spec.old_attr)}\b",
                original_before,
                flags=re.M,
            ) is not None
            if imported_attr:
                pattern_standalone = re.compile(rf"\b{re.escape(spec.old_attr)}\b")
                text, n = pattern_standalone.subn(new_attr, text)
                total_edits += n
        
    return text, total_edits


# ------------------------------- Public API -------------------------------- #

def apply_crossfile_rename(
    specs: Sequence[RenameSpec],
    *,
    files: Sequence[Union[str, Path]] | None = None,
    project_root: Path | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> Tuple[int, List[Path]]:
    """Apply *specs* under *project_root* or to specific *files*."""
    root = Path(project_root or Path.cwd())

    if not specs:
        return 0, []

    edits = 0
    changed: List[Path] = []

    paths = _to_paths(files) if files is not None else _iter_pyfiles(root, include=include, exclude=exclude)

    for path in paths:
        path = Path(path)
        if path.suffix != ".py" or not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated, n = _apply_to_text(original, specs)
        if n and updated != original:
            path.write_text(updated, encoding="utf-8")
            edits += n
            changed.append(path)

    return edits, changed


def preview_crossfile_rename(
    specs: Sequence[RenameSpec],
    *,
    files: Sequence[Union[str, Path]],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> Dict[str, str]:
    """
    Preview which files would change and show the changed content.
    This function does NOT write to disk.
    """
    would_change: Dict[str, str] = {}
    for path in _to_paths(files):
        p = Path(path)
        if p.suffix != ".py" or not p.exists():
            continue
        
        rel_s = p.as_posix()
        if exclude and any(fnmatch(rel_s, pat) for pat in exclude):
            continue
        if include and not any(fnmatch(rel_s, pat) for pat in include):
            continue
            
        original = p.read_text(encoding="utf-8")
        updated, n = _apply_to_text(original, specs)
        if n and updated != original:
            would_change[str(p)] = updated
            
    return would_change


def update_import_paths(
    root: Path,
    old: str,
    new: str,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    preview: bool = False,
) -> Dict[str, str] | List[Path]:
    """
    Update import statements project-wide from module `old` to `new`.

    If preview=True, returns a dict of {path: new_content}.
    Otherwise, applies in-place and returns list of changed Paths.
    """
    specs = [RenameSpec(old_module=old, new_module=new)]
    root = Path(root)
    results: Dict[str, str] = {}
    changed: List[Path] = []

    for path in _iter_pyfiles(root, include=include, exclude=exclude):
        original = path.read_text(encoding="utf-8")
        updated, n = _apply_to_text(original, specs)
        if n and updated != original:
            if preview:
                results[str(path)] = updated
            else:
                path.write_text(updated, encoding="utf-8")
                changed.append(path)
    return results if preview else changed


__all__ = ["RenameSpec", "apply_crossfile_rename", "preview_crossfile_rename", "update_import_paths"]
