import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from selfcoder.analysis.smells import Smell


def smells_to_plan(smells: List[Smell]) -> Dict[str, Any]:
    """
    Convert a list of Smell instances into a Nerion AST edit plan JSON.
    This plan is compatible with the existing batch/transformers system.
    """
    actions = []

    # Collect target files for file-scoped actions to avoid applying to unrelated files
    target_files: set[str] = set()

    for s in smells:
        # High complexity → suggest a conservative branching simplification around the reported line
        if s.tool == "radon" and s.code in ("B", "C", "D", "E", "F"):
            # Keep both a legacy extract_function request (for tests) and a conservative helper
            actions.append({
                "action": "extract_function",
                "kind": "extract_function",
                "payload": {"path": s.path, "line": s.line},
                "reason": f"High cyclomatic complexity ({s.message})",
            })
            actions.append({
                "kind": "simplify_branching",
                "payload": {"line": s.line},
                "reason": f"High cyclomatic complexity ({s.message})",
            })
            if s.path:
                target_files.add(s.path)
        # Unused imports (pylint/flake8) → keep a planner action for downstream tools
        if (s.tool == "pylint" and str(s.code) in ("unused-import", "W0611", "C0415")) or (
            s.tool == "flake8" and str(s.code).startswith("F401")
        ):
            actions.append({
                "action": "remove_unused_imports",
                "kind": "remove_unused_imports",
                "payload": {"path": s.path},
                "reason": f"{s.tool}: {s.message}",
            })
            if s.path:
                target_files.add(s.path)
        # Bandit security findings → emit a security_refactor action placeholder
        if s.tool == "bandit":
            actions.append({
                "action": "security_refactor",
                "kind": "security_refactor",
                "payload": {"path": s.path, "line": s.line, "rule": s.code},
                "reason": f"Bandit: {s.message}",
            })
            if s.path:
                target_files.add(s.path)
        # These can be implemented as dedicated transformers in the future.

    plan: Dict[str, Any] = {
        "version": 1,
        "actions": actions,
        "bundle_id": uuid.uuid4().hex,
        "metadata": {
            "source": "smells",
            "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            "count": len(actions),
        },
        "preconditions": [],
        "postconditions": ["no_unresolved_imports"],
    }
    # If we detected specific files, scope the plan to them
    if target_files:
        plan["files"] = sorted(target_files)
    return plan
