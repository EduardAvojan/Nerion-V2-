from pathlib import Path
import os
import shutil
from typing import Union, IO, Optional
from ops.security import fs_guard

# --- SAFE mode helpers ---
def _truthy(val: Optional[str]) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _is_safe_mode() -> bool:
    # Prefer explicit self-improve SAFE toggle; fall back to generic NERION_SAFE
    return _truthy(os.getenv("NERION_SELFIMPROVE_SAFE", os.getenv("NERION_SAFE", "1")))

def _log_mutation(tag: str, message: str) -> None:
    prefix = f"[SIM{tag}]" if _is_safe_mode() else f"[{tag}]"
    print(f"{prefix} {message}")

PathLike = Union[str, os.PathLike, Path]

def _safe_path(path: PathLike, repo_root: PathLike = ".") -> Path:
    return fs_guard.ensure_in_repo(Path(repo_root), str(path))

# --- file writes ---
def write_text(path: PathLike, data: str, repo_root: PathLike = ".") -> None:
    p = _safe_path(path, repo_root)
    _log_mutation("WRITE", str(p))
    p.write_text(data, encoding="utf-8")

def write_bytes(path: PathLike, data: bytes, repo_root: PathLike = ".") -> None:
    p = _safe_path(path, repo_root)
    _log_mutation("WRITE", str(p))
    p.write_bytes(data)

def open_write(path: PathLike, mode: str = "w", repo_root: PathLike = ".", **kw) -> IO:
    if "w" not in mode and "a" not in mode and "x" not in mode and "+" not in mode:
        raise ValueError("open_write() requires a write-capable mode")
    p = _safe_path(path, repo_root)
    _log_mutation("OPENWRITE", f"{p} mode={mode}")
    return open(p, mode, **kw)

# --- directory ops ---
def mkdir(path: PathLike, exist_ok: bool = True, repo_root: PathLike = ".") -> None:
    p = _safe_path(path, repo_root)
    _log_mutation("MKDIR", str(p))
    p.mkdir(parents=True, exist_ok=exist_ok)

def makedirs(path: PathLike, exist_ok: bool = True, repo_root: PathLike = ".") -> None:
    mkdir(path, exist_ok=exist_ok, repo_root=repo_root)

# --- mutations ---
def remove(path: PathLike, repo_root: PathLike = ".") -> None:
    p = _safe_path(path, repo_root)
    _log_mutation("REMOVE", str(p))
    os.remove(p)

def rename(src: PathLike, dst: PathLike, repo_root: PathLike = ".") -> None:
    s = _safe_path(src, repo_root)
    d = _safe_path(dst, repo_root)
    _log_mutation("RENAME", f"{s} -> {d}")
    os.rename(s, d)

def copy(src: PathLike, dst: PathLike, repo_root: PathLike = ".") -> None:
    s = _safe_path(src, repo_root)
    d = _safe_path(dst, repo_root)
    _log_mutation("COPY", f"{s} -> {d}")
    shutil.copy(s, d)

def copy2(src: PathLike, dst: PathLike, repo_root: PathLike = ".") -> None:
    s = _safe_path(src, repo_root)
    d = _safe_path(dst, repo_root)
    _log_mutation("COPY2", f"{s} -> {d}")
    shutil.copy2(s, d)

def move(src: PathLike, dst: PathLike, repo_root: PathLike = ".") -> None:
    s = _safe_path(src, repo_root)
    d = _safe_path(dst, repo_root)
    _log_mutation("MOVE", f"{s} -> {d}")
    shutil.move(s, d)

def rmtree(path: PathLike, repo_root: PathLike = ".") -> None:
    p = _safe_path(path, repo_root)
    _log_mutation("RMTREE", str(p))
    shutil.rmtree(p, ignore_errors=False)
