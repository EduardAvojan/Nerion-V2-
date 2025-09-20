from __future__ import annotations

from selfcoder.cli_ext.patch import _summarize_tool_counts


def test_summarize_tool_counts_formats_counts():
    findings = [
        {'tool': 'eslint', 'rule_id': 'no-unused', 'filename': 'a.ts', 'line': 1, 'message': '...'},
        {'tool': 'eslint', 'rule_id': 'semi', 'filename': 'a.ts', 'line': 2, 'message': '...'},
        {'tool': 'tsc', 'rule_id': 'TS2322', 'filename': 'b.ts', 'line': 5, 'message': '...'},
        {'tool': 'other', 'rule_id': 'X', 'filename': 'c.ts', 'line': 3, 'message': '...'},
    ]
    s = _summarize_tool_counts(findings)
    assert 'eslint:2' in s
    assert 'tsc:1' in s

