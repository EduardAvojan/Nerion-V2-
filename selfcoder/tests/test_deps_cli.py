from __future__ import annotations
import argparse
import json

from selfcoder.cli_ext import deps_cli as deps_cli


def _run_cmd(func, ns: argparse.Namespace):
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = func(ns)
    finally:
        sys.stdout = old
    out = buf.getvalue()
    return code, json.loads(out)


def test_deps_scan_outputs_basic_json():
    code, data = _run_cmd(deps_cli.cmd_scan, argparse.Namespace(offline=True))
    assert code == 0
    assert isinstance(data.get("freeze"), list)
    assert isinstance(data.get("outdated"), list)
    assert isinstance(data.get("audit"), dict)


def test_deps_plan_patch_policy_structure():
    code, plan = _run_cmd(deps_cli.cmd_plan, argparse.Namespace(policy="patch", offline=True))
    assert code == 0
    assert plan.get("policy") == "patch"
    assert "upgrades" in plan and isinstance(plan["upgrades"], list)


def test_deps_apply_dry_run_returns_commands():
    code, bundle = _run_cmd(deps_cli.cmd_apply, argparse.Namespace(policy="patch", dry_run=True, offline=True))
    assert code == 0
    assert "plan" in bundle and "result" in bundle
    assert bundle["result"].get("dry_run") is True
    assert isinstance(bundle["result"].get("commands"), list)


from pathlib import Path


def test_deps_scan_persists_artifact(tmp_path):
    code, data = _run_cmd(deps_cli.cmd_scan, argparse.Namespace(offline=True))
    assert code == 0
    assert "artifact_path" in data
    p = Path(data["artifact_path"])
    assert p.exists() and p.is_file()


def test_deps_plan_accepts_filters_and_persists():
    # offline yields no upgrades, but we still validate the flags are accepted
    ns = argparse.Namespace(policy="patch", offline=True, only="requests,urllib3", exclude="urllib3")
    code, plan = _run_cmd(deps_cli.cmd_plan, ns)
    assert code == 0
    assert plan.get("policy") == "patch"
    assert "artifact_path" in plan


def test_deps_apply_persists_artifact():
    ns = argparse.Namespace(policy="patch", dry_run=True, offline=True)
    code, bundle = _run_cmd(deps_cli.cmd_apply, ns)
    assert code == 0
    assert "artifact_path" in bundle
    assert "plan" in bundle and "result" in bundle