from __future__ import annotations

from pathlib import Path

from selfcoder.security.scanner import scan_source


def _scan(src: str):
    return scan_source(src, "demo.py", Path("."))


def _rule_ids(findings):
    return {f.rule_id for f in findings}


def test_subprocess_popen_flagged():
    findings = _scan("import subprocess\nsubprocess.Popen(['ls'])\n")
    assert "AST-PROC-010" in _rule_ids(findings)


def test_asyncio_shell_flagged():
    findings = _scan("import asyncio\nasyncio.create_subprocess_shell('ls')\n")
    assert "AST-PROC-011" in _rule_ids(findings)


def test_os_remove_flagged():
    findings = _scan("import os\nos.remove('tmp.txt')\n")
    assert "AST-FS-020" in _rule_ids(findings)


def test_slack_token_regex():
    token = "xoxb-123456789012-123456789012-abcdefghijklmn"
    findings = _scan(f"TOKEN = '{token}'\n")
    assert "REG-SECRET-005" in _rule_ids(findings)
