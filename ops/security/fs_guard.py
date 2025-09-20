from __future__ import annotations
from pathlib import Path
import os
import tempfile
import io


class RepoPathViolation(ValueError):
    """Raised when a path attempts to escape the repository root."""
    ...

# --- Repo root inference helpers -------------------------------------------------

def _find_marker_dir(start: Path) -> Path | None:
    """
    Walk upward from `start` looking for common project markers that indicate the
    repository root. Returns the Path of the directory that contains a marker,
    or None if no marker is found.
    Markers considered:
      - .git
      - pyproject.toml
      - selfcoder/ (project package)
      - .nerion/ (project meta)
    """
    cur = start
    # Ensure we start from a directory
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / "selfcoder").exists():
            return parent
        if (parent / ".nerion").exists():
            return parent
    return None

def infer_repo_root(path_like: str | os.PathLike | Path) -> Path:
    """
    Infer a repository root given any path inside a project. This allows temp
    projects created in tests to be treated as their own repos (so operations on
    /tmp/... files are permitted within that temp project rather than rejected
    as "outside" the main repo).

    If no markers are found, fall back to the containing directory of the given
    path (or the path itself if it is a directory).
    """
    p = Path(path_like).resolve()
    base = p if p.is_dir() else p.parent
    marker_root = _find_marker_dir(base)
    return marker_root or base


def ensure_in_repo_auto(path_like: str | os.PathLike | Path) -> Path:
    """
    Convenience: infer the repo root from `path_like` and ensure the path is
    inside that root. Useful for one-off operations that only touch a single
    path and where the path itself defines the project boundary (e.g., tests
    using tmp_path).
    """
    root = infer_repo_root(path_like)
    return ensure_in_repo(root, path_like)

def _resolve_repo_root(repo_root: Path | str) -> Path:
    root = Path(repo_root).resolve()
    if not root.exists():
        # Do not auto-create here; callers should pass a valid repo root
        raise ValueError(f"Repository root does not exist: {root}")
    return root

def ensure_in_repo(repo_root: Path | str, path_like: str | os.PathLike) -> Path:
    """
    Resolve a candidate path against repo_root and ensure it does not escape.
    Returns the resolved absolute Path if safe; raises RepoPathViolation otherwise.
    """
    root = _resolve_repo_root(repo_root)
    candidate = (root / Path(path_like)).resolve()
    # Allow the root itself or any descendant
    if candidate == root or root in candidate.parents:
        return candidate
    raise RepoPathViolation(f"Path escapes repository: {candidate}")

def atomic_write_text(repo_root: Path | str, relpath: str, data: str, encoding: str = "utf-8") -> Path:
    """
    Atomically write text under repo_root: tmp file -> fsync -> rename.
    Ensures the target path is inside the repo and creates parent dirs as needed.
    """
    target = ensure_in_repo(repo_root, relpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile with delete=False so we can rename cross-platform
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(target.parent), encoding=encoding) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)
    return target

def read_text(repo_root: Path | str, relpath: str, encoding: str = "utf-8") -> str:
    """Read a text file constrained to the repository root."""
    p = ensure_in_repo(repo_root, relpath)
    return p.read_text(encoding=encoding)

def open_text(repo_root: Path | str, relpath: str, mode: str = "r", encoding: str = "utf-8") -> io.TextIOWrapper:
    """
    Open a text file constrained to the repository root. Only text modes are allowed.
    Creates parent directories for write modes.
    """
    if set(mode) & set("b"):
        raise ValueError("Binary mode not allowed via open_text; use an explicit binary-safe helper if needed.")
    p = ensure_in_repo(repo_root, relpath)
    if any(flag in mode for flag in ("w", "a", "+")):
        p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode, encoding=encoding)

# Backwards-compatible helper (preferred: use ensure_in_repo)
# Kept for callers that previously used is_allowed(path) with a global ALLOWLIST.
# New signature requires explicit repo_root to avoid hidden, incorrect roots.

def is_allowed(path: str, *, repo_root: Path | str = ".") -> bool:
    """
    Return True if the given path (relative or absolute) resides within repo_root.
    Prefer using ensure_in_repo(...) to get a validated resolved Path or an exception.
    """
    try:
        ensure_in_repo(repo_root, path)
        return True
    except RepoPathViolation:
        return False