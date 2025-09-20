from enum import Enum
from typing import Optional

class Result(str, Enum):
    OK = "OK"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"

def fmt(component: str, action: str, result: Result, detail: Optional[str] = None) -> str:
    """
    Format a single user-facing line:
      "<component>: <action> → <RESULT> (detail)"
    - component: e.g., "patch", "policy", "bench", "voice", "http"
    - action:    e.g., "apply-hunks", "audit", "test-subset"
    - result:    Result enum
    - detail:    short human context (risk score, rule id, counts, etc.)
    """
    arrow = "→"
    # Use the enum value (e.g., "OK"), not the repr ("Result.OK")
    base = f"{component}: {action} {arrow} {getattr(result, 'value', str(result))}"
    return f"{base} ({detail})" if detail else base
