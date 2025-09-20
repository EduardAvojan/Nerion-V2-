from selfcoder.reviewers.reviewer import format_review

def test_review_unified_and_policy_explain():
    report = {
        "security": {
            "proceed": False,
            "score": 5,
            "reason": "policy violation",
            "findings": [
                {"rule_id": "POLICY:PATH", "severity": "high", "filename": "plugins/x.js", "line": 0, "message": "denied"}
            ],
        },
        "style": {},
        "summary": {"files": 1, "security_ok": False, "security_findings": 1, "style_hints": 0},
    }
    s = format_review(report)
    assert "review: security → BLOCKED" in s
    assert "score=5" in s
    assert "Why:" in s and "Fix:" in s
    assert "policy.md" in s
