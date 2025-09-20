from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import pytest

from selfcoder import cli
from selfcoder.orchestrator import apply_plan


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def test_scoped_env_apply_preserves_existing_env(tmp_path: Path, monkeypatch):
    # Pre-set a stricter env
    monkeypatch.setenv('NERION_POLICY', 'balanced')
    target = _write(Path('tmp/profile_scope_mod.py'), 'def f():\n    return 1\n')
    plan = {
        'actions': [
            {'kind': 'add_module_docstring', 'payload': {'doc': 'X'}},
        ],
        'target_file': target.as_posix(),
    }
    apply_plan(plan, dry_run=False)
    # Env must remain unchanged (scoped application should not relax/override pre-set values)
    assert os.getenv('NERION_POLICY') == 'balanced'


def test_bench_auto_profile_logs_and_feedback(tmp_path: Path, capsys, monkeypatch):
    task = tmp_path / 'bench_auto_demo'
    _write(task / 'm.py', 'def add(a,b):\n    assert a + b == 3\n    return a+b\n')
    _write(task / 'tests' / 'test_add.py', 'from m import add\n\n\ndef test_add():\n    add(1,1)\n')
    # Install a minimal proposer that disables assert lines
    plugin_path = Path('plugins/repair_diff.py')
    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_code = (
        "from difflib import unified_diff\n"
        "from pathlib import Path\n"
        "def propose_diff(ctx):\n"
        "    files = ctx.get('files') or []\n"
        "    if not files: return ''\n"
        "    target = files[0].get('path')\n"
        "    text = Path(target).read_text(encoding='utf-8')\n"
        "    def _disable_asserts(s: str) -> str:\n"
        "        out = []\n"
        "        for ln in s.splitlines():\n"
        "            if ln.lstrip().startswith('assert'):\n"
        "                lead = ln[:len(ln)-len(ln.lstrip())]\n"
        "                out.append(lead + 'pass')\n"
        "            else:\n"
        "                out.append(ln)\n"
        "        return '\\n'.join(out) + '\\n'\n"
        "    new = _disable_asserts(text)\n"
        "    name = Path(target).name\n"
        "    return ''.join(unified_diff(text.splitlines(True), new.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))\n"
    )
    plugin_path.write_text(plugin_code, encoding='utf-8')
    try:
        monkeypatch.setenv('NERION_BENCH_USE_LIBPYTEST', '1')
        rc = cli.main(['bench', 'repair', '--task', str(task), '--max-iters', '1'])
        out = capsys.readouterr().out
        assert '[bench] profile:' in out
        assert rc in (0,)
        # Feedback file should exist
        p = Path('out/learning/prefs.json')
        assert p.exists()
    finally:
        try:
            plugin_path.unlink()
        except Exception:
            pass


def test_cli_profile_sticky_override_round_trip(monkeypatch, capsys):
    # set sticky override
    rc = cli.main(['profile', 'set', '--task', 'bench_repair', '--profile', 'fast'])
    assert rc == 0
    # show
    rc = cli.main(['profile', 'show'])
    out = capsys.readouterr().out
    assert 'sticky_overrides' in out
    # explain should return a decision
    rc = cli.main(['profile', 'explain', '--task', 'bench_repair'])
    out = capsys.readouterr().out
    assert 'decision' in out
    # clear
    rc = cli.main(['profile', 'clear', '--task', 'bench_repair'])
    assert rc == 0


def test_docs_site_query_profile_hint(monkeypatch, capsys):
    # Network likely disabled; we only rely on the hint printed prior to gate
    monkeypatch.setenv('NERION_ALLOW_NETWORK', '0')
    rc = cli.main(['docs', 'site-query', '--url', 'https://example.com', '--query', 'hello'])
    out = capsys.readouterr().out
    assert '[profile] hint:' in out
    assert rc in (0,1)
