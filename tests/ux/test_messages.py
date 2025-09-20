import re
from core.ui.messages import fmt, Result

def test_fmt_basic():
    line = fmt("patch", "apply-hunks", Result.OK)
    assert line == "patch: apply-hunks → OK"

def test_fmt_with_detail():
    line = fmt("policy", "audit", Result.BLOCKED, "deny_paths 'plugins/**'")
    assert "policy: audit → BLOCKED (deny_paths 'plugins/**')" in line
    assert "→" in line  # ensure arrow glyph present
