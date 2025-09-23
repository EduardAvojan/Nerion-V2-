from __future__ import annotations

import os
import sys
from pathlib import Path
from difflib import unified_diff

import pytest

from selfcoder import cli
from selfcoder.orchestrator import apply_plan


_RELAXED_POLICY = Path(__file__).parent / "fixtures" / "policy_relaxed.yaml"


@pytest.fixture(autouse=True)
def _relaxed_policy(monkeypatch):
    monkeypatch.setenv("NERION_POLICY_FILE", str(_RELAXED_POLICY))


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_bench_repair_triage_artifacts(tmp_path: Path, monkeypatch):
    # Create a minimal failing task directory that pytest can run
    task = tmp_path / "bench_task_demo"
    src = _write(task / "m.py", "def add(a,b):\n    assert a + b == 3\n    return a+b\n")
    tests_dir = task / "tests"
    _write(tests_dir / "test_add.py", "from m import add\n\n\ndef test_add():\n    add(1,1)\n")

    # Run the bench CLI in-process
    rc = cli.main(["bench", "repair", "--task", str(task)])
    assert rc in (0,)

    # Verify artifacts were written under out/bench/<taskname>/
    out_root = Path("out/bench") / task.name
    assert (out_root / "triage.json").exists()
    assert (out_root / "suspects.json").exists()
    assert (out_root / "context.json").exists()


def test_bench_repair_with_plugin_stub(tmp_path: Path, monkeypatch):
    # Prepare task with a simple failing assertion in module m.py
    task = tmp_path / "bench_task_patch"
    src = _write(task / "m.py", "def add(a,b):\n    assert a + b == 3\n    return a+b\n")
    tests_dir = task / "tests"
    _write(tests_dir / "test_add.py", "from m import add\n\n\ndef test_add():\n    add(1,1)\n")

    # Install a minimal proposer plugin that edits m.py assert 3 -> 2
    plugin_path = Path("plugins/repair_diff.py")
    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_code = (
        "from difflib import unified_diff\n"
        "from pathlib import Path\n"
        "def _disable_asserts(text: str) -> str:\n"
        "    out = []\n"
        "    for ln in text.splitlines():\n"
        "        if ln.lstrip().startswith('assert'):\n"
        "            lead = ln[:len(ln)-len(ln.lstrip())]\n"
        "            out.append(lead + 'pass')\n"
        "        else:\n"
        "            out.append(ln)\n"
        "    return '\n'.join(out) + '\n'\n"
        "def propose_diff(ctx):\n"
        "    files = ctx.get('files') or []\n"
        "    target = None\n"
        "    for f in files:\n"
        "        p = f.get('path') or ''\n"
        "        if p.endswith('m.py'):\n"
        "            target = p\n"
        "            break\n"
        "    if not target and files:\n"
        "        target = files[0].get('path')\n"
        "    if not target:\n"
        "        tdir = Path(ctx.get('_task_dir', '.'))\n"
        "        cand = tdir / 'm.py'\n"
        "        if cand.exists():\n"
        "            target = str(cand)\n"
        "        else:\n"
        "            return ''\n"
        "    text = Path(target).read_text(encoding='utf-8')\n"
        "    new = _disable_asserts(text)\n"
        "    name = Path(target).name\n"
        "    return ''.join(unified_diff(text.splitlines(True), new.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))\n"
    )
    plugin_path.write_text(plugin_code, encoding="utf-8")
    try:
        # Use in-process pytest to avoid subprocess environment issues
        monkeypatch.setenv("NERION_BENCH_USE_LIBPYTEST", "1")
        # Limit iterations to 1; should be sufficient to fix the task
        rc = cli.main(["bench", "repair", "--task", str(task), "--max-iters", "1"])
        assert rc == 0
    finally:
        # Cleanup plugin to avoid polluting further tests
        try:
            plugin_path.unlink()
        except Exception:
            pass


def test_diff_gated_by_style_threshold(monkeypatch, tmp_path: Path):
    # Target file lacking a module docstring (will generate style hints)
    target = _write(Path("tmp/gate_mod.py"), "def f():\n    return 1\n")
    old = target.read_text(encoding="utf-8")
    new = old.replace("return 1", "return 2")
    diff = "".join(
        unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/" + target.as_posix(),
            tofile="b/" + target.as_posix(),
        )
    )
    plan = {"actions": [{"kind": "apply_unified_diff", "payload": {"diff": diff}}]}

    # Gate strictly on style hints: set max to 0 to force block
    monkeypatch.setenv("NERION_REVIEW_STYLE_MAX", "0")
    rc = apply_plan(plan, dry_run=False)
    # No modifications should have been applied
    assert target.read_text(encoding="utf-8") == old

    # Relax threshold and allow apply
    monkeypatch.setenv("NERION_REVIEW_STYLE_MAX", "9999")
    rc = apply_plan(plan, dry_run=False)
    assert target.read_text(encoding="utf-8") == new


def test_diff_rollback_on_postcondition_failure(tmp_path: Path, monkeypatch):
    # Prepare a file and a diff that introduces an unresolved import
    target = _write(Path("tmp/rollback_mod.py"), "def f():\n    return 1\n")
    before = target.read_text(encoding="utf-8")
    after = "import not_a_real_module\n" + before
    diff = "".join(
        unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/" + target.as_posix(),
            tofile="b/" + target.as_posix(),
        )
    )
    plan = {
        "actions": [{"kind": "apply_unified_diff", "payload": {"diff": diff}}],
        "postconditions": ["no_unresolved_imports"],
    }
    # Apply (real) — postconditions should fail and trigger rollback
    rc = apply_plan(plan, dry_run=False)
    # File content should remain unchanged due to rollback
    assert target.read_text(encoding="utf-8") == before


def test_diff_security_gate_blocks_critical(tmp_path: Path):
    # Prepare a benign file and a diff that adds a private key header (critical)
    target = _write(Path("tmp/security_block_mod.py"), "def f():\n    return 1\n")
    before = target.read_text(encoding="utf-8")
    dangerous = "-----BEGIN RSA PRIVATE KEY-----\n" + before
    diff = "".join(
        unified_diff(
            before.splitlines(True),
            dangerous.splitlines(True),
            fromfile="a/" + target.as_posix(),
            tofile="b/" + target.as_posix(),
        )
    )
    plan = {"actions": [{"kind": "apply_unified_diff", "payload": {"diff": diff}}]}
    apply_plan(plan, dry_run=False)
    # Security gate should block writing the dangerous content
    assert target.read_text(encoding="utf-8") == before
