


import argparse
from pathlib import Path
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes
from core.ui.progress import progress


def cmd_scan(args: argparse.Namespace) -> int:
    from selfcoder.security.scanner import scan_source
    from selfcoder.security.report import write_report_json, format_summary
    from selfcoder.security import GateResult

    repo = Path('.').resolve()
    findings = []
    with progress("Scan"):
        for f in getattr(args, 'files', []) or []:
            fp = Path(f)
            if not fp.exists():
                print(f"[MISS] {fp} (does not exist)")
                continue
            try:
                src = fp.read_text(encoding="utf-8")
                findings.extend(scan_source(src, str(fp), repo))
            except Exception as e:
                print(f"[WARN] cannot read {fp}: {e}")
    result = GateResult(proceed=True, score=0, findings=findings, reason="scan-only")

    def _sev_rank(s):
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(str(s).lower(), 0)

    rc = 0
    thr = getattr(args, "fail_on", "none").lower()
    if thr != "none":
        worst = max((_sev_rank(getattr(f, "severity", "")) for f in findings), default=0)
        if worst >= _sev_rank(thr):
            rc = 1
    if getattr(args, 'json', False):
        out = write_report_json(result, Path(getattr(args, 'outdir', 'backups/security')))
        print(str(out))
    else:
        print(format_summary(result))
        worst = max(( _sev_rank(getattr(f, 'severity', '')) for f in findings), default=0)
        sev_name = {1:'low',2:'medium',3:'high',4:'critical'}.get(worst,'none')
        print(_fmt_msg('scan', 'summary', _MsgRes.OK if rc == 0 else _MsgRes.FAIL, f"files={len(getattr(args,'files',[]) or [])} findings={len(findings)} worst={sev_name}"))
    return rc


def register(subparsers: argparse._SubParsersAction) -> None:
    sscan = subparsers.add_parser("scan", help="security scan files")
    sscan.add_argument("files", nargs="+")
    sscan.add_argument("--json", action="store_true")
    sscan.add_argument("--outdir", default="backups/security")
    sscan.add_argument("--fail-on", choices=["none","low","medium","high","critical"], default="none")
    sscan.set_defaults(func=cmd_scan)
