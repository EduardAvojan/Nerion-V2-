from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from selfcoder.analysis.static_checks import run_all
from selfcoder.analysis.smells import normalize_reports, Smell
from selfcoder.planner.plan_from_smells import smells_to_plan
from selfcoder.diagnostics import analyze_exception, persist_analysis

# Add fs_guard import
from ops.security import fs_guard

# Orchestrated apply & safety nets
from selfcoder.orchestrator import Orchestrator
from selfcoder.vcs.git_ops import snapshot, restore_snapshot
from selfcoder.healthcheck import run_healthcheck
try:
    from selfcoder.simulation import SimulationMode as _RealSimulationMode
except Exception:
    class _RealSimulationMode:  # fallback shim if simulation module/class is unavailable
        @staticmethod
        def apply(plan_json):
            # No-op simulation; real application path remains available via Orchestrator
            return None
SimulationMode = _RealSimulationMode

# Optional: programmatic plugin reload after successful apply
try:
    from plugins.loader import reload_plugins_auto as _reload_plugins_auto
except Exception:
    def _reload_plugins_auto(*_args, **_kwargs):
        return None

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

# Output locations
REPORT_DIR = Path("out/analysis_reports")
PLAN_DIR = Path("out/improvement_plans")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLAN_DIR.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# Helper: Only enforce repo jail for relative paths; allow explicit absolute tmp dirs (for tests, etc.)
def _should_enforce_repo_jail(p: Path) -> bool:
    """Enforce jail only for relative paths; allow explicit absolute tmp dirs set by tests."""
    try:
        return not p.is_absolute()
    except Exception:
        return True


def scan(paths: Optional[List[str]] = None) -> Path:
    """Run analyzers and write a raw combined JSON report. Return report path."""
    raw = run_all(paths)
    p = REPORT_DIR / f"report_{_ts()}.json"
    if _should_enforce_repo_jail(p):
        p = fs_guard.ensure_in_repo(Path('.'), str(p))
    p.write_text(json.dumps(raw, indent=2))
    try:
        # best-effort metrics
        metric = None
        if isinstance(raw, dict):
            metric = {"keys": len(raw)}
        elif isinstance(raw, list):
            metric = {"items": len(raw)}
        _log_event(
            "scan",
            rationale="static analysis",
            paths=paths,
            report_path=str(p),
            metrics=metric,
        )
    except Exception:
        pass
    return p


def plan(report_file: Path) -> Path:
    """Read a raw report, normalize to Smells, produce an AST plan JSON. Return plan path."""
    raw = json.loads(Path(report_file).read_text())
    smells: List[Smell] = normalize_reports(raw)
    plan_json = smells_to_plan(smells)
    p = PLAN_DIR / f"plan_{_ts()}.json"
    if _should_enforce_repo_jail(p):
        p = fs_guard.ensure_in_repo(Path('.'), str(p))
    p.write_text(json.dumps(plan_json, indent=2))
    try:
        actions_count = 0
        if isinstance(plan_json, dict):
            actions_count = len(plan_json.get("actions", []))
        _log_event(
            "plan",
            rationale="normalized smells → plan",
            report_path=str(report_file),
            plan_path=str(p),
            actions_count=actions_count,
        )
    except Exception:
        pass
    return p


def _get_ts_from_snapshot(snap: Any) -> str:
    """Robustly extract the timestamp string from a snapshot object."""
    if isinstance(snap, str):
        return snap
    # Handle the dictionary structure from your snapshot function
    if isinstance(snap, dict):
        # Check for common keys
        for key in ("ts", "timestamp", "snapshot_id"):
            val = snap.get(key)
            if val:
                return str(val)
    if isinstance(snap, (list, tuple)) and snap:
        return str(snap[0])
    return str(snap)


def _stringify_snapshot(snap: Any) -> str:
    """Robustly convert a snapshot object to a string for JSON."""
    # This handles the complex object you showed in the last successful run
    if isinstance(snap, dict):
        # Convert all Path objects in the 'files' list to strings
        files = snap.get("files", [])
        str_files = [str(p) for p in files]
        # Create a new dictionary with the stringified list
        snap_copy = snap.copy()
        snap_copy["files"] = str_files
        return str(snap_copy)
    return str(snap)


def apply(plan_file: Path, simulate: bool = True) -> Dict[str, Any]:
    """
    Apply a generated plan with full safety:
      - snapshot → apply (simulation or real) → healthcheck/tests → rollback on failure

    Returns a dict summarizing outcome. JSON contract:
      • When simulate=True (SAFE):
          {"applied": false, "simulated": true, "writes_blocked_by": "SAFE", ...}
        No repo state should be rolled back because no writes happened.
      • When simulate=False (real):
          {"applied": true, "simulated": false, ...}
        On healthcheck/exception failure we restore from snapshot and set rolled_back=True.
    """
    try:
        plan_json = json.loads(Path(plan_file).read_text())
    except Exception as e:
        analysis = analyze_exception(e)
        analysis_path = persist_analysis(analysis)
        result = {
            "applied": False, "rolled_back": False, "simulated": bool(simulate), "error": str(e),
            "error_type": e.__class__.__name__, "analysis": analysis,
            "snapshot": None, "analysis_path": str(analysis_path),
        }
        return result

    snap = snapshot()
    ts_for_restore = _get_ts_from_snapshot(snap)
    
    try:
        staged_artifacts: Optional[List[str]] = None
        if simulate:
            sim_ret = SimulationMode.apply(plan_json)
            # If the simulation returns any info about staged artifacts, capture it
            if isinstance(sim_ret, dict):
                # try common keys, but do not fail if missing
                for key in ("staged_artifacts", "artifacts", "tmp_writes"):
                    v = sim_ret.get(key)
                    if isinstance(v, (list, tuple)):
                        staged_artifacts = [str(x) for x in v]
                        break
        else:
            Orchestrator.apply_plan(plan_json)

        health_result = run_healthcheck()
        ok = health_result[0] if isinstance(health_result, tuple) else health_result

        if not ok:
            if simulate:
                # No repo writes occurred, so do not restore. Make the status explicit.
                result = {
                    "applied": False,
                    "simulated": True,
                    "rolled_back": False,
                    "writes_blocked_by": "SAFE",
                    "snapshot": _stringify_snapshot(snap),
                    "reason": "healthcheck_failed",
                }
                if staged_artifacts:
                    result["staged_artifacts"] = staged_artifacts
                _log_event(
                    "apply",
                    rationale="self-improve apply (healthcheck failed, simulation)",
                    plan_file=str(plan_file),
                    simulate=True,
                    snapshot_id=ts_for_restore,
                    outcome=False,
                    rolled_back=False,
                    writes_blocked_by="SAFE",
                )
                return result
            else:
                restore_snapshot(snapshot_ts=ts_for_restore, verbose=False)
                result = {
                    "applied": False,
                    "simulated": False,
                    "rolled_back": True,
                    "snapshot": _stringify_snapshot(snap),
                    "reason": "healthcheck_failed",
                }
                _log_event(
                    "apply",
                    rationale="self-improve apply (healthcheck failed, real)",
                    plan_file=str(plan_file),
                    simulate=False,
                    snapshot_id=ts_for_restore,
                    outcome=False,
                    rolled_back=True,
                )
                return result

        # Success path
        try:
            _reload_plugins_auto()
        except Exception:
            pass

        if simulate:
            result = {
                "applied": True,
                "simulated": True,
                "rolled_back": False,
                "writes_blocked_by": "SAFE",
                "snapshot": _stringify_snapshot(snap),
            }
            if staged_artifacts:
                result["staged_artifacts"] = staged_artifacts
            _log_event(
                "apply",
                rationale="self-improve apply (simulate)",
                plan_file=str(plan_file),
                simulate=True,
                snapshot_id=ts_for_restore,
                outcome=True,
                rolled_back=False,
                writes_blocked_by="SAFE",
            )
            return result
        else:
            result = {
                "applied": True,
                "simulated": False,
                "rolled_back": False,
                "snapshot": _stringify_snapshot(snap),
            }
            _log_event(
                "apply",
                rationale="self-improve apply (real)",
                plan_file=str(plan_file),
                simulate=False,
                snapshot_id=ts_for_restore,
                outcome=True,
                rolled_back=False,
            )
            return result

    except Exception as e:
        if not simulate:
            # Only restore on real apply exceptions
            restore_snapshot(snapshot_ts=ts_for_restore, verbose=False)
        analysis = analyze_exception(e)
        analysis_path = persist_analysis(analysis)
        result = {
            "applied": False,
            "simulated": bool(simulate),
            "rolled_back": (not simulate),
            "error": str(e),
            "error_type": e.__class__.__name__,
            "analysis": analysis,
            "snapshot": _stringify_snapshot(snap),
            "analysis_path": str(analysis_path),
        }
        if simulate:
            result["writes_blocked_by"] = "SAFE"
        _log_event(
            "apply",
            rationale="self-improve apply (exception)",
            plan_file=str(plan_file),
            simulate=bool(simulate),
            snapshot_id=ts_for_restore,
            outcome=False,
            rolled_back=(not simulate),
            error=str(e),
            analysis=analysis,
            writes_blocked_by=("SAFE" if simulate else None),
        )
        return result
