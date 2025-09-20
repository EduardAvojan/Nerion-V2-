

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from selfcoder.plans.schema import validate_plan
from selfcoder.artifacts import PlanArtifact, save_artifact
from selfcoder.scoring import score_plan

from selfcoder.planner.plan_from_smells import smells_to_plan
try:  # pragma: no cover
    from selfcoder.analysis.static_checks import normalize_static_outputs
except Exception:  # pragma: no cover
    normalize_static_outputs = None  # type: ignore

# Optional analyzers — best-effort; self-audit still works if missing
try:  # pragma: no cover
    from selfcoder.analysis import smells as _smells
except Exception:  # pragma: no cover
    _smells = None  # type: ignore

try:  # pragma: no cover
    from selfcoder.analysis import static_checks as _static
except Exception:  # pragma: no cover
    _static = None  # type: ignore


def collect_findings(root: Path) -> Dict[str, Any]:
    root = Path(root)
    out: Dict[str, Any] = {"root": str(root)}
    try:
        if _smells and hasattr(_smells, "scan_tree"):
            out["smells"] = _smells.scan_tree(root)  # type: ignore[attr-defined]
    except Exception:
        out["smells"] = []
    try:
        if _static and hasattr(_static, "run"):
            out["static"] = _static.run(root)  # type: ignore[attr-defined]
    except Exception:
        out["static"] = []
    return out


def synthesize_plan(root: Path, findings: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize a plan from findings. Falls back to a safe sandbox helper if no actionable items exist."""
    # Collect Smell-like items from smell scanners
    smell_items = []
    try:
        s = findings.get("smells")
        if isinstance(s, list):
            smell_items.extend(s)
    except Exception:
        pass

    # Normalize static check outputs into Smell instances if available
    try:
        static_raw = findings.get("static")
        if static_raw and normalize_static_outputs:
            smell_items.extend(normalize_static_outputs(static_raw))
    except Exception:
        pass

    if smell_items:
        plan = smells_to_plan(smell_items)
        # Ensure metadata notes selfaudit provenance and summary counts
        meta = plan.get("metadata") or {}
        meta["source"] = "selfaudit"
        meta.setdefault("summary", {k: len(v) if isinstance(v, list) else 0 for k, v in findings.items()})
        plan["metadata"] = meta
        validate_plan(plan)
        return plan

    # Fallback: conservative sandbox helper when nothing actionable was found
    target = Path("tmp/selfaudit_helper.py")
    plan: Dict[str, Any] = {
        "actions": [
            {
                "kind": "insert_function",
                "payload": {
                    "name": "_selfaudit_marker",
                    "content": "def _selfaudit_marker():\n    return True\n",
                },
            }
        ],
        "target_file": str(target),
        "metadata": {
            "source": "selfaudit",
            "summary": {k: len(v) if isinstance(v, list) else 0 for k, v in findings.items()},
        },
        "postconditions": ["no_unresolved_imports"],
    }
    validate_plan(plan)
    return plan


def generate_improvement_plan(root: Path) -> Dict[str, Any]:
    root = Path(root)
    findings = collect_findings(root)
    plan = synthesize_plan(root, findings)
    try:
        score, why = score_plan(plan, None)
        artifact = PlanArtifact(
            ts=None,
            origin="selfaudit.generate_improvement_plan",
            score=score,
            rationale=why,
            plan=plan,
            files_touched=[],
            sim=None,
            meta={"root": str(root), "kind": "improvement_plan"},
        )
        save_artifact(artifact)
    except Exception:
        pass
    return plan