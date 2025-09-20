# selfcoder/scoring.py
from __future__ import annotations
from typing import Any, Dict

def score_plan(plan: Dict[str, Any], sim: Dict[str, Any] | None = None) -> tuple[int, str]:
    """
    Returns (score, rationale). Inputs:
      - plan: validated plan dict
      - sim: optional simulation summary {pytest_rc, health_ok, files_touched, diff_size}
    """
    s = 50
    actions = plan.get("actions") or []
    n_actions = len(actions)
    if n_actions <= 5:
        s += 10
    elif n_actions > 40:
        s -= 15
    else:
        s += max(0, 8 - int(n_actions / 5))

    # prefer docstrings/logging/test additions
    kinds = [a.get("action") or a.get("kind") for a in actions]
    hygiene = sum(1 for k in kinds if k and any(x in k for x in ("docstring", "logging", "ensure_test", "fix_imports")))
    s += min(10, hygiene * 3)

    # simulation signals
    rationale_bits = []
    if sim:
        if sim.get("pytest_rc") == 0:
            s += 20
            rationale_bits.append("pytest passed")
        elif sim.get("pytest_rc") is not None:
            s -= 10
            rationale_bits.append("pytest failed")

        if sim.get("health_ok") is True:
            s += 10
            rationale_bits.append("health ok")
        elif sim.get("health_ok") is False:
            s -= 10
            rationale_bits.append("health failed")

        diff_size = sim.get("diff_size")
        if isinstance(diff_size, int):
            if diff_size <= 200:
                s += 10
            elif diff_size > 2000:
                s -= 15

        touched = sim.get("files_touched") or []
        if len(touched) <= 5:
            s += 5
        elif len(touched) > 30:
            s -= 10

    s = max(0, min(100, s))
    rationale = ", ".join(rationale_bits) if rationale_bits else "static heuristics"
    return s, rationale