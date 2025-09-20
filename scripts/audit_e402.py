#!/usr/bin/env python3
"""Audit E402 occurrences and summarize standardized reasons.

Finds:
- inline E402 ignores with reason tags (e.g., `# noqa: E402 (lazy_import: perf)`)
- per-file E402 ignores in pyproject.toml
Outputs a short summary to stdout for CI and local audits.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RE_INLINE = re.compile(r"#\s*noqa:\s*E402\s*\(([^)]+)\)")

REASONS: dict[str, int] = {}
FILES: dict[str, list[str]] = {}

for path in ROOT.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), start=1):
        m = RE_INLINE.search(line)
        if m:
            reason = m.group(1).strip()
            REASONS[reason] = REASONS.get(reason, 0) + 1
            FILES.setdefault(reason, []).append(f"{path}:{i}")

pyproject = ROOT / "pyproject.toml"
PFI: list[str] = []
if pyproject.exists():
    try:
        txt = pyproject.read_text(encoding="utf-8", errors="ignore")
        in_pfi = False
        for ln in txt.splitlines():
            if ln.strip().startswith("per-file-ignores"):
                in_pfi = True
            if in_pfi:
                PFI.append(ln.rstrip())
            if in_pfi and ln.strip().startswith("]"):
                in_pfi = False
    except Exception:
        pass

print("E402 ignore audit")
print("==================")
print()
if REASONS:
    print("Inline ignores by reason:")
    for reason, count in sorted(REASONS.items(), key=lambda kv: kv[1], reverse=True):
        print(f" - {reason}: {count}")
    print()
else:
    print("No inline E402 ignores with standardized reasons found.")

if PFI:
    print("Per-file ignores (pyproject.toml excerpt):")
    for ln in PFI:
        print(" ", ln)
else:
    print("No per-file ignores section found or empty.")

# Exit code is always 0; report is informational
sys.exit(0)
