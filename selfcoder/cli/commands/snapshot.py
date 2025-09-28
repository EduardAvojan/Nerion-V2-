

from __future__ import annotations
import click
import datetime
import shutil
from pathlib import Path
from typing import Optional

SNAPSHOT_DIR = Path(".nerion/snapshots")

@click.command("snapshot")
@click.argument("label", required=False)
def cli(label: Optional[str] = None) -> None:
    """
    Create a filesystem snapshot under .nerion/snapshots with an optional label.
    Ensures snapshot path is confined within the repository root.
    """
    repo_root = Path(".").resolve()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_name = f"{ts}" + (f"_{label}" if label else "")
    snap_path = ensure_in_repo(repo_root, SNAPSHOT_DIR / snap_name)

    try:
        snap_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"[ERR] Snapshot already exists: {snap_path}")
        raise SystemExit(1)

    # For now, copy key project dirs (selfcoder, plugins, ops) into the snapshot
    for sub in ("selfcoder", "plugins", "ops"):
        src = repo_root / sub
        if src.exists():
            dst = snap_path / sub
            shutil.copytree(src, dst)

    print(f"[SNAPSHOT] Created at {snap_path}")