from __future__ import annotations

import os
import json
from pathlib import Path
import pytest

from selfcoder.planner.utils import sanitize_plan, repo_fingerprint, load_plan_cache, save_plan_cache


def test_sanitize_drops_disallowed_and_normalizes(tmp_path: Path):
    raw = {
        "actions": [
            {"kind": "create_file", "payload": {"path": "a.py", "content": "print(1)"}},
            {"kind": "unknown_action", "payload": {}},
        ],
        "metadata": {"foo": "bar"},
    }
    clean = sanitize_plan(raw)
    assert isinstance(clean, dict)
    acts = clean.get("actions")
    assert isinstance(acts, list) and len(acts) == 1
    a0 = acts[0]
    assert a0.get("kind") == "create_file"
    assert "payload" in a0 and a0["payload"].get("path") == "a.py"


def test_repo_fingerprint_changes_with_file(tmp_path: Path):
    (tmp_path / "x.py").write_text("x=1\n", encoding="utf-8")
    f1 = repo_fingerprint(tmp_path)
    (tmp_path / "x.py").write_text("x=2\n", encoding="utf-8")
    f2 = repo_fingerprint(tmp_path)
    assert f1 != f2


def test_cache_roundtrip(tmp_path: Path):
    p = tmp_path / "cache.json"
    data = {"k": {"actions": []}}
    save_plan_cache(p, data)
    got = load_plan_cache(p)
    assert got == data


@pytest.mark.skipif(bool(os.environ.get("NERION_CODER_BASE_URL")), reason="LLM may be available")
def test_llm_strict_raises_when_unavailable():
    os.environ["NERION_LLM_STRICT"] = "1"
    from selfcoder.planner.llm_planner import plan_with_llm
    with pytest.raises(RuntimeError):
        plan_with_llm("add docstring", None)
    os.environ.pop("NERION_LLM_STRICT", None)

