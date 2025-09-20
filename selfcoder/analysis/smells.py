from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Smell:
    """Unified representation of a code smell discovered by any analyzer."""
    tool: str                 # "pylint" | "bandit" | "radon" | "flake8" | "custom"
    code: str                 # e.g., "R0912", "B603", "F401"
    message: str
    path: str                 # file path
    line: Optional[int] = None
    symbol: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def normalize_reports(raw: Dict[str, Any]) -> List[Smell]:
    """
    Convert a combined raw report
    { 'pylint': [...], 'bandit': [...], 'radon': [...], 'flake8': [...] }
    into a unified list of Smell instances. Tolerant of missing tools/fields.
    """
    out: List[Smell] = []

    # pylint JSON format: list of dicts
    for item in raw.get("pylint", []) or []:
        out.append(
            Smell(
                tool="pylint",
                code=(item.get("symbol") or item.get("msg_id") or ""),
                message=item.get("message", ""),
                path=(item.get("path") or item.get("filename") or ""),
                line=item.get("line"),
                symbol=item.get("symbol") or None,
                meta=item,
            )
        )

    # bandit JSON format: { "results": [ ... ] }
    for item in raw.get("bandit", []) or []:
        out.append(
            Smell(
                tool="bandit",
                code=item.get("test_id", ""),
                message=item.get("issue_text", ""),
                path=item.get("filename", ""),
                line=item.get("line_number"),
                symbol=None,
                meta=item,
            )
        )

    # radon cc JSON format: { path: [ { name, lineno, complexity, rank }, ... ] }
    for item in raw.get("radon", []) or []:
        out.append(
            Smell(
                tool="radon",
                code=str(item.get("rank", "")),
                message=f"Complexity={item.get('complexity')}",
                path=item.get("path", ""),
                line=item.get("line") or item.get("lineno"),
                symbol=item.get("name") or None,
                meta=item,
            )
        )

    # flake8 parsed lines → list of dicts { path, line, code, text }
    for item in raw.get("flake8", []) or []:
        out.append(
            Smell(
                tool="flake8",
                code=item.get("code", ""),
                message=item.get("text", ""),
                path=item.get("path", ""),
                line=item.get("line"),
                symbol=None,
                meta=item,
            )
        )

    return out