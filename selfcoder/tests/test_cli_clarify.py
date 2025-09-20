

from __future__ import annotations
import argparse
import json

import pytest

from selfcoder.cli_ext import chat


def run_cmd_chat(clarify_text: str, target_file: str = "foo.py"):
    args = argparse.Namespace(clarify=clarify_text, target_file=target_file)
    # Capture printed JSON
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        exit_code = chat.cmd_chat(args)
    finally:
        sys.stdout = old
    out = buf.getvalue().strip()
    plan = json.loads(out) if out else {}
    return exit_code, plan


def test_clarify_required_on_underspecified():
    code, plan = run_cmd_chat("insert function", target_file="foo.py")
    assert code == 2
    assert plan["metadata"]["clarify_required"] is True
    assert "clarify" in plan


def test_no_clarify_on_well_specified():
    code, plan = run_cmd_chat("insert function util_add; doc 'adds two numbers'", target_file="foo.py")
    assert code == 0
    assert plan["metadata"]["clarify_required"] is False
    assert plan["actions"][0]["kind"] == "insert_function"
    assert plan["actions"][0]["payload"]["name"] == "util_add"


def test_class_insert_no_clarify_when_named():
    code, plan = run_cmd_chat("insert class Greeter; doc 'greets politely'", target_file="foo.py")
    assert code == 0
    assert plan["metadata"]["clarify_required"] is False
    assert plan["actions"][0]["kind"] == "insert_class"
    assert plan["actions"][0]["payload"]["name"] == "Greeter"


def test_class_insert_requires_clarify_when_unnamed():
    code, plan = run_cmd_chat("insert class", target_file="foo.py")
    assert code == 2
    assert plan["metadata"]["clarify_required"] is True
    assert "clarify" in plan