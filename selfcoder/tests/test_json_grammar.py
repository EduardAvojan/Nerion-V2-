from __future__ import annotations

import os
import importlib
import types
import pytest


class DummyCoder:
    def __init__(self, *a, **k):
        pass
    def complete_json(self, prompt: str, system: str | None = None):
        return "not json"


def test_json_grammar_strict_enforced(monkeypatch):
    # Patch the coder to return non-JSON and enable grammar/strict
    mod = importlib.import_module("app.parent.coder_v2")
    monkeypatch.setattr(mod, "DeepSeekCoderV2", DummyCoder)
    os.environ["NERION_JSON_GRAMMAR"] = "1"
    os.environ["NERION_LLM_STRICT"] = "1"
    from selfcoder.planner.llm_planner import plan_with_llm
    with pytest.raises(RuntimeError):
        plan_with_llm("add a docstring", None)
    os.environ.pop("NERION_JSON_GRAMMAR", None)
    os.environ.pop("NERION_LLM_STRICT", None)

