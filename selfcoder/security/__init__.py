from dataclasses import dataclass
from typing import List, Literal
Severity = Literal["low", "medium", "high", "critical"]
@dataclass
class Finding:
    rule_id: str
    severity: Severity
    message: str
    filename: str
    line: int
    evidence: str
@dataclass
class GateResult:
    proceed: bool
    score: int
    findings: List[Finding]
    reason: str
