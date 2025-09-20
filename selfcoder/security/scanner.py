import ast
from pathlib import Path
from typing import List
from . import Finding
from .rules import DangerVisitor, regex_findings
def scan_source(source: str, filename: str, repo_root: Path) -> List[Finding]:
    findings: List[Finding] = []
    try:
        # Guard: skip obvious binary blobs
        if "\x00" in source:
            return findings
        # Parse AST and run structural checks
        try:
            tree = ast.parse(source)
            v = DangerVisitor(filename=filename, repo_root=repo_root)
            v.visit(tree)
            findings.extend(v.findings)
        except SyntaxError as e:
            findings.append(Finding("PARSE-001", "medium", f"Syntax error while parsing: {e}", filename, getattr(e, "lineno", 1) or 1, ""))
        # Regex-based sweeps last
        findings.extend(regex_findings(source, filename))
    except Exception as _e:  # fail-open
        # We don't block on scanner errors; just return what we have
        pass
    return findings
