from __future__ import annotations

from pathlib import Path
from difflib import unified_diff

import pytest

from selfcoder.actions.text_patch import preview_unified_diff
from selfcoder.orchestrator import apply_plan


_RELAXED_POLICY = Path(__file__).parent / "fixtures" / "policy_relaxed.yaml"


@pytest.fixture(autouse=True)
def _relaxed_policy(monkeypatch):
    monkeypatch.setenv("NERION_POLICY_FILE", str(_RELAXED_POLICY))


def _make_file(rel: str, text: str) -> Path:
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_preview_unified_diff_basic(tmp_path):
    # Work under repo-relative path (repo jail expects inside repo)
    target = _make_file("tmp/text_patch_target.py", 'def foo():\n    return 1\n')
    old = target.read_text()
    new = old.replace("return 1", "return 2")
    diff = "".join(
        unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/" + target.as_posix(),
            tofile="b/" + target.as_posix(),
        )
    )
    previews, errs = preview_unified_diff(diff, Path(".").resolve())
    assert not errs
    assert target.resolve() in previews
    before, after = previews[target.resolve()]
    assert "return 1" in before
    assert "return 2" in after


def test_apply_plan_with_unified_diff_dryrun():
    target = _make_file("tmp/text_patch_target2.py", 'def foo():\n    return 10\n')
    old = target.read_text()
    new = old.replace("return 10", "return 11")
    diff = "".join(
        unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/" + target.as_posix(),
            tofile="b/" + target.as_posix(),
        )
    )
    plan = {"actions": [{"kind": "apply_unified_diff", "payload": {"diff": diff}}]}
    modified = apply_plan(plan, dry_run=True)
    # Should report the file as would-be modified
    assert target.resolve() in [p.resolve() for p in modified]
