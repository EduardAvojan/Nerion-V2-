"""
Filesystem action utilities for orchestration.

This module provides utilities for applying filesystem-level actions
like creating files and ensuring test scaffolds exist.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
from ops.security import fs_guard
from selfcoder.actions.transformers import build_test_scaffold

# Optional journal logging
try:
    from core.memory.journal import log_event as _log_event  # type: ignore
except Exception:  # pragma: no cover
    try:
        from app import journal as _journal  # type: ignore
        def _log_event(kind: str, **fields):
            try:
                _journal.append({"kind": kind, **fields})
            except Exception:
                pass
    except Exception:
        def _log_event(*_a, **_kw):
            return None


def apply_fs_actions(
    fs_actions: List[Dict[str, Any]],
    default_target: str | None = None,
    *,
    dry_run: bool = False
) -> List[Path]:
    """Apply simple filesystem actions like `create_file` and `ensure_file`.

    Args:
        fs_actions: List of filesystem action dictionaries
        default_target: Default target path if not specified in action
        dry_run: If True, don't actually create files

    Returns:
        List of created/ensured Paths
    """
    from .validators import should_skip

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

            skip, reason = should_skip(test_path)
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
                    rationale="orchestrator.apply_fs_actions",
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
        skip, reason = should_skip(p)
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
            rationale="orchestrator.apply_fs_actions",
            created=[str(c) for c in created],
            count=len(fs_actions or []),
            dry_run=bool(dry_run),
        )
    except Exception:
        pass
    return created
