from __future__ import annotations
import json
import os
import sys
from ops.security.safe_subprocess import safe_run
from ops.security import fs_guard
from pathlib import Path
from typing import Dict, List, Tuple, Optional


BASELINE_DIR = Path(".nerion")
BASELINE_FILE = BASELINE_DIR / "coverage_baseline.json"


def _should_enforce_repo_jail(p: Path) -> bool:
    """
    Enforce jail only for the default baseline file or any relative path.
    Explicit absolute paths (e.g., test tmp dirs) are allowed.
    """
    try:
        return (p.resolve() == BASELINE_FILE.resolve()) or (not p.is_absolute())
    except Exception:
        # If resolution fails, be conservative
        return True

def run_pytest_with_coverage(
    pytest_args: List[str],
    include: Optional[List[str]] = None,
    omit: Optional[List[str]] = None,
    json_out: Optional[Path] = None,
    cov_context: Optional[str] = None,
) -> Dict:
    """
    Run pytest under coverage and return parsed JSON report.
    Requires pytest-cov to be installed.
    """
    env = os.environ.copy()
    cov_cmd = [
        sys.executable, "-m", "pytest",
        "--maxfail=1",
        "--disable-warnings",
        "--cov=selfcoder",
        "--cov-report=term",
        "--cov-report=json:coverage.json",
    ]
    if cov_context:
        # pytest-cov may not support this flag on older versions; ignore errors later
        cov_cmd.append(f"--cov-context={cov_context}")
    if include:
        # allow multiple --cov entries by appending; first entry already set above
        for inc in include:
            cov_cmd.append(f"--cov={inc}")

    # omit filters are typically handled via .coveragerc; keep env untouched here
    full_cmd = cov_cmd + (pytest_args or [])
    # Use guarded subprocess (no shell, limited env)
    try:
        safe_run(full_cmd, env=env, check=False)
    except Exception:
        # Intentionally swallow to allow reading whatever coverage.json exists
        pass

    data: Dict = {}
    cov_json = Path("coverage.json")
    if cov_json.exists():
        data = json.loads(cov_json.read_text(encoding="utf-8"))
        if json_out:
            try:
                root = fs_guard._resolve_repo_root(".")
                safe_out = fs_guard.ensure_in_repo(root, str(json_out))
                safe_out.parent.mkdir(parents=True, exist_ok=True)
                safe_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                # refuse to write outside repo
                pass
    return data

def overall_percent(cov_json: Dict) -> float:
    totals = cov_json.get("totals") or {}
    covered = float(totals.get("covered_lines", 0))
    total = float(totals.get("num_statements", 0))
    return (covered / total * 100.0) if total else 0.0

def load_baseline(path: Path = BASELINE_FILE) -> Optional[Dict]:
    p = Path(path)
    if _should_enforce_repo_jail(p):
        try:
            root = fs_guard._resolve_repo_root(".")
            p = fs_guard.ensure_in_repo(root, str(p))
        except Exception:
            return None
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def save_baseline(data: Dict, path: Path = BASELINE_FILE) -> None:
    p = Path(path)
    if _should_enforce_repo_jail(p):
        try:
            root = fs_guard._resolve_repo_root(".")
            p = fs_guard.ensure_in_repo(root, str(p))
        except Exception:
            return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

def compare_to_baseline(current: Dict, baseline: Optional[Dict]) -> Tuple[float, float]:
    """Returns (current_pct, delta_vs_baseline)."""
    cur = overall_percent(current)
    if not baseline:
        return cur, 0.0
    prev = overall_percent(baseline)
    return cur, (cur - prev)

def suggest_targets(cov_json: Dict, top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Suggest files with most missing lines (descending).
    Returns list of (filename, missing_line_count).
    """
    files = cov_json.get("files") or {}
    scores: List[Tuple[str, int]] = []
    for fname, meta in files.items():
        summary = meta.get("summary") or {}
        missing = int(summary.get("missing_lines", 0))
        if missing > 0:
            scores.append((fname, missing))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]
