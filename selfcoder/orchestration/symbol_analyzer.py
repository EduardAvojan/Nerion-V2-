"""
Symbol and import analysis utilities for orchestration.

This module provides utilities for analyzing Python code symbols,
imports, and resolving module dependencies.
"""
import ast
import importlib.util
from contextlib import suppress
from pathlib import Path
from typing import List, Tuple
from ops.security import fs_guard


# Repository root for module resolution
REPO_ROOT = fs_guard.infer_repo_root(Path('.'))


def symbol_present_in_file(symbol: str, file_path: str) -> bool:
    """Detect whether a top-level function or class with the given name exists in file_path.

    Args:
        symbol: Name of the symbol to check
        file_path: Path to the file to check

    Returns:
        True if symbol is present, False otherwise
    """
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


def evaluate_preconditions(preconds: List[str] | None) -> Tuple[bool, List[str]]:
    """Evaluate preconditions for an action.

    Supports tokens like:
    - "file_exists:path/to/file.py"
    - "symbol_absent:NAME@path/to/file.py"
    Unknown tokens are ignored (treated as satisfied).

    Args:
        preconds: List of precondition strings

    Returns:
        Tuple of (ok: bool, failures: List[str])
    """
    from .validators import file_exists

    if not preconds:
        return True, []
    failures: List[str] = []
    for raw in preconds:
        if not isinstance(raw, str):
            continue
        if raw.startswith("file_exists:"):
            path = raw.split(":", 1)[1]
            if not file_exists(path):
                failures.append(f"missing {path}")
        elif raw.startswith("symbol_absent:"):
            payload = raw.split(":", 1)[1]
            if "@" in payload:
                name, file_path = payload.split("@", 1)
            else:
                name, file_path = payload, ""
            if file_path and symbol_present_in_file(name, file_path):
                failures.append(f"symbol {name} present in {file_path}")
        else:
            # Unknown token: ignore
            continue
    return (len(failures) == 0), failures


def extract_import_module_names(tree: ast.AST) -> List[str]:
    """Extract all imported module names from an AST.

    Args:
        tree: AST tree to analyze

    Returns:
        List of imported module names
    """
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


def module_resolves(name: str, base_file: Path) -> bool:
    """Check if a module name can be resolved.

    Args:
        name: Module name to resolve
        base_file: Base file path for relative resolution

    Returns:
        True if module can be resolved, False otherwise
    """
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


def unresolved_imports_in_file(path: Path) -> List[str]:
    """Find all unresolved imports in a file.

    Args:
        path: Path to the file to analyze

    Returns:
        List of unresolved module names
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return []  # If we cannot parse, let other checks catch it
    unresolved: List[str] = []
    for mod in extract_import_module_names(tree):
        if not module_resolves(mod, path):
            unresolved.append(mod)
    return unresolved
