import os
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import shutil
from ops.security import fs_guard
from selfcoder.config import self_improve_safe

try:
    from core.memory.journal import log_event as _log_event  # type: ignore
except Exception:  # pragma: no cover
    def _log_event(*_a, **_kw):
        return None

# SAFE controls import (kept from main)
try:
    from selfcoder.main import SAFE_MODE, DRY_RUN  # type: ignore
except Exception:
    import os
    def _env_bool(name: str, default: bool=False) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        v = v.strip().lower()
        return v in {'1','true','yes','on'}
    SAFE_MODE = _env_bool('SELFCODER_SAFE_MODE', True)
    DRY_RUN  = _env_bool('SELFCODER_DRYRUN', False)

# Allow env to override imported SAFE_MODE/DRY_RUN (useful in tests)
_def_true = {'1', 'true', 'yes', 'on'}

def _override_bool(current: bool, env_name: str) -> bool:
    v = os.getenv(env_name)
    if v is None:
        return current
    return str(v).strip().lower() in _def_true

SAFE_MODE = _override_bool(SAFE_MODE, 'SELFCODER_SAFE_MODE')
DRY_RUN  = _override_bool(DRY_RUN,  'SELFCODER_DRYRUN')

# Resolve effective flags at call time (env wins)
_def_false = {'0', 'false', 'no', 'off'}

def _effective_safe() -> bool:
    v = os.getenv('SELFCODER_SAFE_MODE')
    if v is None:
        return SAFE_MODE
    vv = v.strip().lower()
    if vv in _def_true:
        return True
    if vv in _def_false:
        return False
    return SAFE_MODE

def _effective_dry() -> bool:
    v = os.getenv('SELFCODER_DRYRUN')
    if v is None:
        return DRY_RUN
    vv = v.strip().lower()
    if vv in _def_true:
        return True
    if vv in _def_false:
        return False
    return DRY_RUN


# Optional runtime controls
VERBOSE_SKIPS = os.getenv("SELFCODER_VERBOSE_SKIPS", "0") == "1"
MAX_LIST = int(os.getenv("SELFCODER_MAX_LIST", "50"))

# Hard default ignores (always skipped)
HARD_IGNORES = [
    ".git/**",
    ".vscode/**",
    ".idea/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    ".venv/**",
    "**/node_modules/**",
    "backups/**",
    "**/*.bak",
]


def _load_ignore_patterns() -> list:
    """Load .selfcoderignore (if present) as additional glob patterns."""
    ignore_file = Path(".selfcoderignore")
    patterns = []
    if ignore_file.exists():
        for line in ignore_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns


def _allow_only_patterns() -> list:
    """Comma-separated glob patterns in SELFCODER_ALLOW_ONLY (optional)."""
    raw = os.getenv("SELFCODER_ALLOW_ONLY", "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _matches_any(posix_rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(posix_rel, pat) for pat in patterns)


def _should_include(rel_path: Path, allow_only: list[str], ignores: list[str]) -> tuple[bool, str]:
    posix = rel_path.as_posix()

    # Hard ignores first
    if _matches_any(posix, HARD_IGNORES):
        return False, "hard-ignored"

    # .selfcoderignore
    if _matches_any(posix, ignores):
        return False, ".selfcoderignore"

    # allow-only (if provided): include only matches
    if allow_only:
        if _matches_any(posix, allow_only):
            return True, "include"
        return False, "not-allowed"

    return True, "include"


def should_skip(file_path: Path) -> tuple[bool, str]:
    """
    Public helper used by other modules to decide if a path should be skipped.
    Returns a tuple `(skip: bool, reason: str)` where `reason` is non-empty when
    `skip` is True. If the file is outside the repo root, we do not skip it but
    return a best-effort reason.
    """
    root = Path(".").resolve()
    try:
        rel = Path(file_path).resolve().relative_to(root)
    except Exception:
        # If it's outside the repo root, don't skip, but mark reason.
        return False, "outside-root?"

    allow_only = _allow_only_patterns()
    extra_ignores = _load_ignore_patterns()
    ok, reason = _should_include(rel, allow_only, extra_ignores)
    return (not ok, reason if not ok else "")


def snapshot(message: Optional[str] = None):
    """
    Create a file list snapshot with concise output.

    Behaviour:
      - If SAFE_MODE: do NOT process files; print only a short summary.
      - If DRY_RUN=1: print up to MAX_LIST of included files and a skip summary.
      - If DRY_RUN=0 and not SAFE_MODE: write manifest to backups/snapshots/manifest_YYYYmmdd_HHMMSS.txt
    Controls:
      - SELFCODER_VERBOSE_SKIPS=1 to print per-file skip lines.
      - SELFCODER_MAX_LIST (int) to cap printed included files in dry run.
      - SELFCODER_ALLOW_ONLY="glob1,glob2" to restrict inclusion.
    """
    root = Path(".").resolve()

    SAFE = bool(self_improve_safe(default=_effective_safe()))
    DRY  = _effective_dry()

    allow_only = _allow_only_patterns()
    extra_ignores = _load_ignore_patterns()

    included: list[Path] = []
    skipped_counts: dict[str, int] = {}
    printed_included = 0

    print("📦 Selfcoder snapshot (concise)")

    # Walk the tree; prune ignored directories early
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)

        # directory pruning
        for dn in list(dirnames):
            dp = dpath / dn
            rel = dp.relative_to(root)
            ok, reason = _should_include(rel, allow_only, extra_ignores)
            if not ok:
                dirnames.remove(dn)
                skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
                if VERBOSE_SKIPS:
                    print(f"[SKIP dir:{reason}] {rel}")

        # files
        for fn in filenames:
            fp = dpath / fn
            rel = fp.relative_to(root)
            ok, reason = _should_include(rel, allow_only, extra_ignores)
            if not ok:
                skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
                if VERBOSE_SKIPS:
                    print(f"[SKIP:{reason}] {rel}")
                continue

            included.append(rel)
            if DRY and printed_included < MAX_LIST:
                print(f"[DRYRUN] Would snapshot: {rel}")
                printed_included += 1

    total_included = len(included)
    total_skipped = sum(skipped_counts.values())

    mode = "safe" if SAFE else ("dry" if DRY else "real")

    # If SAFE_MODE, don't do anything destructive; just summarise
    if SAFE:
        if total_included > printed_included and DRY:
            remaining = total_included - printed_included
            print(f"…and {remaining} more files would be included (showing first {MAX_LIST}).")
        print("\n🧹 Skip summary:")
        if skipped_counts:
            for k in sorted(skipped_counts.keys()):
                print(f"  - {k}: {skipped_counts[k]}")
        else:
            print("  (none)")
        print(f"  Total skipped: {total_skipped}")
        print(f"✅ SAFE mode: no files processed. Candidate include count: {total_included}")
        try:
            _log_event(
                "snapshot",
                rationale="snapshot (SAFE)",
                mode=mode,
                included_count=total_included,
                skipped=skipped_counts,
            )
        except Exception:
            pass
        return included

    # Dry run summary
    if DRY:
        if total_included > printed_included:
            remaining = total_included - printed_included
            print(f"…and {remaining} more files would be included (showing first {MAX_LIST}).")
        print("\n🧹 Skip summary:")
        if skipped_counts:
            for k in sorted(skipped_counts.keys()):
                print(f"  - {k}: {skipped_counts[k]}")
        else:
            print("  (none)")
        print(f"  Total skipped: {total_skipped}")
        print(f"✅ Snapshot file count: {total_included}")
        try:
            _log_event(
                "snapshot",
                rationale="snapshot (DRYRUN)",
                mode=mode,
                included_count=total_included,
                skipped=skipped_counts,
            )
        except Exception:
            pass
        return included

    # Real run: write a manifest file and copy files into a timestamped snapshot dir
    outdir = root / "backups" / "snapshots"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backward-compatible top-level manifest
    outfile = outdir / f"manifest_{ts}.txt"
    # New directory-based snapshot layout
    snap_dir = outdir / ts
    files_dir = snap_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # Write manifests (both legacy and new)
    manifest_lines = []
    if message:
        manifest_lines.append(f"# {message}")
    manifest_lines.extend(rel.as_posix() for rel in included)
    with outfile.open("w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    with (snap_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))

    # Copy files into snapshot directory
    for rel in included:
        # Enforce repo jail for the source path
        try:
            src_abs = fs_guard.ensure_in_repo(root, str(root / rel))
        except Exception:
            # Skip anything that cannot be jailed to the repo (defense-in-depth)
            continue
        dst_fp = files_dir / rel
        dst_fp.parent.mkdir(parents=True, exist_ok=True)
        # Avoid following potentially unsafe symlinks
        if src_abs.is_symlink():
            continue
        try:
            shutil.copy2(src_abs, dst_fp)
        except Exception:
            # Best-effort; skip if copy fails
            pass

    print(f"📝 Wrote snapshot manifest with {total_included} files:\n   {outfile}")
    print(f"📁 Snapshot directory:\n   {snap_dir}")

    print("\n🧹 Skip summary:")
    if skipped_counts:
        for k in sorted(skipped_counts.keys()):
            print(f"  - {k}: {skipped_counts[k]}")
    else:
        print("  (none)")
    print(f"  Total skipped: {total_skipped}")
    print("✅ Done.")
    try:
        _log_event(
            "snapshot",
            rationale="snapshot (REAL)",
            mode=mode,
            included_count=total_included,
            skipped=skipped_counts,
            manifest=str(outfile),
            snapshot_id=ts,
            snapshot_dir=str(snap_dir),
        )
    except Exception:
        pass
    return ts


def _find_latest_snapshot_dir(root: Path) -> Optional[Path]:
    """Return the latest snapshot directory (backups/snapshots/YYYYmmdd_HHMMSS) if any."""
    base = root / "backups" / "snapshots"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Sort by directory name which is timestamp-formatted
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def restore_snapshot(snapshot_ts: Optional[str] = None,
                     files: Optional[List[Path]] = None,
                     verbose: bool = True) -> Tuple[int, Optional[Path]]:
    """
    Restore files from a snapshot directory created by `snapshot()`.

    Args:
        snapshot_ts: Timestamp string (YYYYmmdd_HHMMSS) of the snapshot directory to use.
                     If None, the latest snapshot directory is used.
        files: Optional list of file paths (relative to repo root) to restore.
               If None, all files listed in the snapshot manifest are restored.
        verbose: Print per-file actions.

    Returns:
        (restored_count, snapshot_dir)
    """
    root = Path(".").resolve()
    base = root / "backups" / "snapshots"

    snap_dir: Optional[Path]
    if snapshot_ts:
        snap_dir = base / snapshot_ts
        if not snap_dir.exists() or not snap_dir.is_dir():
            if verbose:
                print(f"[RESTORE] snapshot not found: {snap_dir}")
            try:
                _log_event("restore", rationale="snapshot not found", snapshot_id=snapshot_ts, restored=0)
            except Exception:
                pass
            return 0, None
    else:
        snap_dir = _find_latest_snapshot_dir(root)
        if snap_dir is None:
            if verbose:
                print("[RESTORE] no snapshot directories found")
            try:
                _log_event("restore", rationale="no snapshots available", restored=0)
            except Exception:
                pass
            return 0, None

    files_dir = snap_dir / "files"
    manifest_path = snap_dir / "manifest.txt"

    # Determine files to restore
    rel_paths: List[Path] = []
    if files:
        for p in files:
            cand = Path(p)
            # Normalize to a repo-relative path via jail check
            try:
                abs_ok = fs_guard.ensure_in_repo(root, str(cand))
                rel_paths.append(abs_ok.relative_to(root))
            except Exception:
                # Ignore paths outside the repo
                continue
    else:
        if manifest_path.exists():
            content = manifest_path.read_text(encoding="utf-8").splitlines()
            for line in content:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    abs_ok = fs_guard.ensure_in_repo(root, str(line))
                    rel_paths.append(abs_ok.relative_to(root))
                except Exception:
                    # Skip invalid/outside entries
                    continue
        else:
            # Fallback: gather from files_dir (these are already repo-relative)
            if files_dir.exists():
                for fp in files_dir.rglob("*"):
                    if fp.is_file():
                        rel_paths.append(fp.relative_to(files_dir))

    restored = 0
    for rel in rel_paths:
        src_fp = files_dir / rel
        if not src_fp.exists():
            # file was perhaps deleted before snapshot or copy failed; skip quietly
            continue
        # Enforce repo jail for destination
        try:
            dst_abs = fs_guard.ensure_in_repo(root, str(root / rel))
        except Exception:
            if verbose:
                print(f"[RESTORE-FAIL] {rel}: outside repo")
            continue
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        # Avoid restoring symlinks from snapshot
        if src_fp.is_symlink():
            continue
        try:
            shutil.copy2(src_fp, dst_abs)
            restored += 1
            if verbose:
                print(f"[RESTORE] {rel}")
        except Exception as e:
            if verbose:
                print(f"[RESTORE-FAIL] {rel}: {e}")

    if verbose:
        print(f"[RESTORE] restored {restored} file(s) from {snap_dir}")
    try:
        _log_event(
            "restore",
            rationale="restore from snapshot",
            snapshot_dir=str(snap_dir) if snap_dir else None,
            restored=int(restored),
            files_count=len(rel_paths),
        )
    except Exception:
        pass
    return restored, snap_dir