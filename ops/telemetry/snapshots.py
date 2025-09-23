"""Timeline snapshot utilities for Nerion telemetry."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ops.security.fs_guard import ensure_in_repo_auto

from .store import TelemetryStore

SNAPSHOT_DIR = Path("out/telemetry/snapshots")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git(args: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def _git_state() -> Dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run_git(["status", "--short"]))
    }


def _provider_defaults() -> Dict[str, Any]:
    try:
        from app.chat.providers.base import ProviderRegistry

        registry = ProviderRegistry.from_files()
        roles = {}
        for role in ("chat", "code", "planner", "embeddings"):
            roles[role] = {
                "default": registry.default_provider(role),
                "active": registry.active_provider(role),
            }
        return roles
    except Exception as exc:
        return {"error": str(exc)}


def _telemetry_counts(store: TelemetryStore) -> Dict[str, Any]:
    try:
        counts = store.counts_by_kind()
        by_source = store.counts_by_source()
        return {
            "by_kind": counts,
            "top_sources": sorted(by_source.items(), key=lambda kv: kv[1], reverse=True)[:20],
        }
    except Exception as exc:
        return {"error": str(exc)}


def write_snapshot(path: Optional[str | os.PathLike[str]] = None) -> Path:
    timestamp = _utcnow_iso()
    snapshot_dir = Path(ensure_in_repo_auto(Path(path))) if path else Path(ensure_in_repo_auto(SNAPSHOT_DIR))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    telemetry_block: Dict[str, Any]
    try:
        store = TelemetryStore()
    except Exception as exc:
        telemetry_block = {"error": str(exc)}
    else:
        telemetry_block = _telemetry_counts(store)
        store.close()

    data = {
        "timestamp": timestamp,
        "git": _git_state(),
        "providers": _provider_defaults(),
        "telemetry": telemetry_block,
    }

    safe_ts = timestamp.replace(":", "").replace("-", "").replace(".", "")
    filename = snapshot_dir / f"snapshot_{safe_ts}.json"
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    filename.write_text(payload, encoding="utf-8")
    return filename


__all__ = ["write_snapshot"]
