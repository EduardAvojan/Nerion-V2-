from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

from selfcoder.orchestrator import apply_actions_via_ast
from selfcoder.logging_config import setup_logging
from ops.security import fs_guard
logger = setup_logging(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in {"1", "true", "yes", "on"}



DRY_RUN = _env_bool("SELFCODER_DRYRUN", False)
SAFE_MODE = _env_bool("SELFCODER_SAFE_MODE", True)


def _should_skip(file_path: Path) -> Tuple[bool, str]:
    """Best-effort bridge to git_ops.should_skip; fallback to no-skip."""
    try:
        from selfcoder.vcs.git_ops import should_skip  # type: ignore
        return should_skip(file_path)
    except Exception:
        return (False, "")


def _write(path: Path, text: str, *, dry_run: bool) -> None:
    logger.info("Writing to %s (dry_run=%s)", path.as_posix(), dry_run)
    # Enforce repo jail before any write
    safe_p = fs_guard.ensure_in_repo_auto(str(path))
    if dry_run:
        print(f"[DRYRUN] Would write: {safe_p.as_posix()}")
        return
    safe_p.write_text(text, encoding="utf-8")
    print(f"[WRITE] {safe_p.as_posix()}")


def _insert_module_docstring(file_path: Path, doc: str, *, dry_run: bool = DRY_RUN) -> None:
    logger.debug("Inserting module docstring into %s", file_path)
    path = (PROJECT_ROOT / file_path) if not file_path.is_absolute() else file_path
    skip, reason = _should_skip(path)
    if skip:
        print(f"[SKIP] {path.as_posix()} ({reason})")
        return
    if not path.exists():
        print(f"[MISS] {path.as_posix()} (does not exist)")
        return
    src = path.read_text(encoding="utf-8")
    actions = [{"kind": "add_module_docstring", "payload": {"doc": doc}}]
    new_src = apply_actions_via_ast(src, actions)
    if new_src == src:
        print(f"[NO-OP] {path.as_posix()} (no change needed)")
        return
    _write(path, new_src, dry_run=dry_run)


def _insert_function_docstring(file_path: Path, function: str, doc: str, *, dry_run: bool = DRY_RUN) -> None:
    logger.debug("Inserting function docstring into %s::%s", file_path, function)
    path = (PROJECT_ROOT / file_path) if not file_path.is_absolute() else file_path
    skip, reason = _should_skip(path)
    if skip:
        print(f"[SKIP] {path.as_posix()} ({reason})")
        return
    if not path.exists():
        print(f"[MISS] {path.as_posix()} (does not exist)")
        return
    src = path.read_text(encoding="utf-8")
    actions = [{
        "kind": "add_function_docstring",
        "payload": {"function": function, "doc": doc},
    }]
    new_src = apply_actions_via_ast(src, actions)
    if new_src == src:
        print(f"[NO-OP] {path.as_posix()} (no change needed)")
        return
    _write(path, new_src, dry_run=dry_run)


def run_once(instruction: str) -> None:
    norm = instruction.strip().lower()
    logger.info("Running instruction: %s", instruction)

    if norm == "function docstring":
        target_file = os.getenv("SELFCODER_TARGET_FILE")
        target_func = os.getenv("SELFCODER_TARGET_FUNCTION")
        doc = os.getenv("SELFCODER_DOC", "").strip()
        if not (target_file and target_func and doc):
            print("Missing one of SELFCODER_TARGET_FILE / SELFCODER_TARGET_FUNCTION / SELFCODER_DOC.\n"
                  "Example:\n"
                  "  export SELFCODER_TARGET_FILE=selfcoder/io_utils.py\n"
                  "  export SELFCODER_TARGET_FUNCTION=read_text\n"
                  "  export SELFCODER_DOC=\"Return file contents as text.\"")
            return
        _insert_function_docstring(Path(target_file), target_func, doc)
        return

    if norm == "module docstring" or ("docstring" in norm and "function" not in norm):
        # Demo default: io_utils.py
        doc = (
            "Helper functions for simple file I/O (read_text, write_text). "
            "Internal utility module; this change only adds documentation—no behavior changes."
        )
        _insert_module_docstring(Path("selfcoder/io_utils.py"), doc)
        return

    print("No matching action found for the instruction (demo dispatcher).")


if __name__ == "__main__":
    instruction = os.getenv("SELFCODER_INSTRUCTION", "").strip()
    if instruction:
        logger.info("Starting Selfcoder main with instruction=%r", instruction)
        run_once(instruction)
    else:
        print("📝 Selfcoder Dry Run Mode is ON" if DRY_RUN else "")
