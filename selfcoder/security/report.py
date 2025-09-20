import json
from pathlib import Path
from typing import List
from . import Finding, GateResult
from ops.security import fs_guard
from ops.telemetry.logger import log
def findings_to_dict(findings: List[Finding]):
    return [
        {
            "rule_id": f.rule_id,
            "severity": f.severity,
            "message": f.message,
            "filename": f.filename,
            "line": f.line,
            "evidence": f.evidence,
        }
        for f in findings
    ]
def write_report_json(result: GateResult, out_dir: Path) -> Path:
    """
    Write a JSON security report under a repo-jail-validated directory.
    Refuses to write outside the repository boundary.
    """
    # Validate the output directory is inside the repo jail
    safe_dir = fs_guard.ensure_in_repo(Path('.'), str(out_dir))
    safe_dir.mkdir(parents=True, exist_ok=True)

    out = safe_dir / "security_report.json"
    # Double-check the output file path as well
    out = fs_guard.ensure_in_repo(Path('.'), str(out))

    payload = {
        "proceed": result.proceed,
        "score": result.score,
        "reason": result.reason,
        "findings": findings_to_dict(result.findings),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Structured telemetry (non-sensitive): where and basic outcome only
    try:
        log("SECURITY_REPORT", "written", {"path": out.as_posix(), "proceed": result.proceed, "score": result.score})
    except Exception:
        # logging should never break core flow
        pass

    return out
def format_summary(result: GateResult) -> str:
    if not result.findings:
        return "✅ Security scan: no issues detected"
    lines = [f"{'ALLOW' if result.proceed else 'BLOCK'} — score={result.score} — {result.reason}"]
    for f in result.findings[:20]:
        lines.append(f"- [{f.severity.upper()}] {f.rule_id} {f.filename}:{f.line} — {f.message} ({f.evidence})")
    if len(result.findings) > 20:
        lines.append(f"... and {len(result.findings)-20} more")
    return "\n".join(lines)
