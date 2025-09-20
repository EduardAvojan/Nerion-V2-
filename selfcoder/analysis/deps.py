from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import sys
import os
from typing import Any, Dict, List, Tuple, Optional, Iterable

from ops.security.safe_subprocess import safe_run


def _python() -> str:
    return sys.executable or "python"


def pip_freeze(timeout: int = 60) -> List[str]:
    """Return list of installed packages from `pip freeze`. Safe on failure (empty list)."""
    try:
        cp = safe_run([_python(), "-m", "pip", "freeze"], timeout=timeout, inherit_env=True)
        if cp.returncode == 0:
            return [ln.strip() for ln in (cp.stdout or "").splitlines() if ln.strip()]
    except Exception:
        pass
    return []


def pip_outdated(timeout: int = 60) -> List[Dict[str, Any]]:
    """Return `pip list --outdated --format json`. Safe on failure (empty list)."""
    try:
        cp = safe_run([_python(), "-m", "pip", "list", "--outdated", "--format", "json"], timeout=timeout, inherit_env=True)
        if cp.returncode == 0:
            return json.loads(cp.stdout or "[]")
    except Exception:
        pass
    return []


def pip_audit(timeout: int = 120) -> Dict[str, Any]:
    """Run `python -m pip_audit -f json` if available; else return {available: False}."""
    try:
        cp = safe_run([_python(), "-m", "pip_audit", "-f", "json"], timeout=timeout, inherit_env=True)
        if cp.returncode == 0:
            try:
                data = json.loads(cp.stdout or "{}")
            except json.JSONDecodeError:
                data = {"raw": cp.stdout}
            return {"available": True, "result": data}
        return {"available": True, "rc": cp.returncode, "error": (cp.stderr or "").strip()}
    except Exception as e:
        return {"available": False, "error": str(e)}


def scan(
    timeout_freeze: int = 60,
    timeout_outdated: int = 60,
    timeout_audit: int = 120,
    *,
    providers: Dict[str, Any] | None = None,
    offline: bool | None = None,
) -> Dict[str, Any]:
    """Collect freeze, outdated, audit into a single report dict."""
    # Determine offline mode (explicit arg wins; else env var)
    if offline is None:
        offline = os.environ.get("SELFCODER_DEPS_OFFLINE", "").strip() in {"1", "true", "yes"}

    # Base providers
    _pf = pip_freeze
    _po = pip_outdated
    _pa = pip_audit

    # Apply dependency injection overrides
    if providers:
        _pf = providers.get("pip_freeze", _pf)
        _po = providers.get("pip_outdated", _po)
        _pa = providers.get("pip_audit", _pa)

    # If offline, replace with fast stubs
    if offline:
        def _pf(timeout: int = 60) -> List[str]:
            return []
        def _po(timeout: int = 60) -> List[Dict[str, Any]]:
            return []
        def _pa(timeout: int = 120) -> Dict[str, Any]:
            return {"available": False}

    return {
        "freeze": _pf(timeout_freeze),
        "outdated": _po(timeout_outdated),
        "audit": _pa(timeout_audit),
        "tooling": {"python": sys.version.split()[0]},
    }


def _parse_ver(ver: str) -> Tuple[int, int, int]:
    """Tiny semver-ish parser for numeric parts only."""
    parts = (ver or "").split(".")
    nums: List[int] = []
    for p in parts[:3]:
        try:
            nums.append(int("".join(ch for ch in p if ch.isdigit())))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


def make_upgrade_plan(
    outdated: List[Dict[str, Any]],
    policy: str = "patch",
    *,
    only: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build upgrade plan from pip --outdated entries.

    policy: one of {"patch", "minor", "major"} controlling allowed version jumps.
    Optional filters: `only` and `exclude` package name iterables (case-insensitive).
    """
    if policy not in {"patch", "minor", "major"}:
        policy = "patch"

    # Normalize filters (case-insensitive)
    only_set = {s.strip().lower() for s in (only or []) if isinstance(s, str) and s.strip()}
    exclude_set = {s.strip().lower() for s in (exclude or []) if isinstance(s, str) and s.strip()}

    def _entry_name(d: Dict[str, Any]) -> str:
        return (d.get("name") or d.get("project") or "").strip().lower()

    # Apply filtering to the outdated list
    if only_set or exclude_set:
        filtered_outdated: List[Dict[str, Any]] = []
        for it in (outdated or []):
            nm = _entry_name(it)
            if only_set and nm not in only_set:
                continue
            if exclude_set and nm in exclude_set:
                continue
            filtered_outdated.append(it)
    else:
        filtered_outdated = list(outdated or [])

    upgrades: List[Dict[str, Any]] = []
    for item in filtered_outdated:
        name = item.get("name") or item.get("project")
        cur = item.get("version") or item.get("installed_version")
        latest = item.get("latest_version") or item.get("latest")
        if not name or not cur or not latest:
            continue
        cM, cm, cp = _parse_ver(cur)
        lM, lm, lp = _parse_ver(latest)
        allowed = (
            (lM == cM and lm == cm and lp >= cp) if policy == "patch" else
            (lM == cM and lm >= cm) if policy == "minor" else
            True
        )
        if allowed:
            upgrades.append({"name": name, "from": cur, "to": latest})

    return {
        "policy": policy,
        "upgrades": upgrades,
        "preconditions": ["backup_snapshot", "tests_green", "healthcheck_ok"],
        "postconditions": ["tests_green", "healthcheck_ok", "no_new_vulns"],
    }


def apply_plan(plan: Dict[str, Any], timeout: int = 300, dry_run: bool = True) -> Dict[str, Any]:
    """Apply upgrades using pinned targets. Default is dry-run.

    Returns: {applied, dry_run, commands, results?}
    """
    cmds: List[List[str]] = []
    for up in plan.get("upgrades", []):
        pkg = up.get("name")
        target = up.get("to")
        if not pkg or not target:
            continue
        cmds.append([_python(), "-m", "pip", "install", "--upgrade", f"{pkg}=={target}"])

    if dry_run:
        return {"applied": False, "dry_run": True, "commands": cmds}

    results: List[Dict[str, Any]] = []
    for argv in cmds:
        try:
            cp = safe_run(argv, timeout=timeout, inherit_env=True)
            results.append({"argv": argv, "rc": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr})
            if cp.returncode != 0:
                break
        except Exception as e:
            results.append({"argv": argv, "error": str(e)})
            break
    ok = all(r.get("rc", 1) == 0 for r in results) if results else True
    return {"applied": ok, "dry_run": False, "commands": cmds, "results": results}


def make_scan_providers_offline() -> Dict[str, Any]:
    """Return provider callables that implement an offline scan (fast, no subprocess)."""
    return {
        "pip_freeze": lambda timeout=60: [],
        "pip_outdated": lambda timeout=60: [],
        "pip_audit": lambda timeout=120: {"available": False},
    }


# --- Persistence helpers for scan/plan/apply artifacts ---

def _outdir(sub: str) -> Path:
    base = Path("out") / sub
    base.mkdir(parents=True, exist_ok=True)
    return base


def persist_scan(report: Dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _outdir("analysis_reports") / f"deps_scan_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def persist_plan(plan: Dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _outdir("improvement_plans") / f"deps_plan_{ts}.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def persist_apply(bundle: Dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _outdir("analysis_reports") / f"deps_apply_{ts}.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return path