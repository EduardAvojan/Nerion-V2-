from __future__ import annotations

import json
import os
from pathlib import Path


def _build_parser():
    import selfcoder.cli as cli
    return cli._build_parser()


def test_prefs_schema_version_upgrade_and_reset(tmp_path, monkeypatch, capsys):
    # Seed an old-style prefs without schema_version
    outp = Path('out/learning'); outp.mkdir(parents=True, exist_ok=True)
    (outp / 'prefs.json').write_text(json.dumps({'tool_success_rate': {'a': 0.5}}), encoding='utf-8')
    # Add a minimal log so review_outcomes produces fields
    exp = Path('out/experience'); exp.mkdir(parents=True, exist_ok=True)
    (exp / 'log.jsonl').write_text(json.dumps({'outcome_success': True, 'action_taken': {'steps': [{'tool': 'a'}]}})+"\n", encoding='utf-8')
    from selfcoder.learning.continuous import review_outcomes, load_prefs
    _ = review_outcomes()
    prefs = load_prefs()
    assert isinstance(prefs.get('schema_version'), (int, float))

    # Reset clears file, then review recreates with version
    parser = _build_parser()
    ns = parser.parse_args(['learn', 'reset'])
    rc = ns.func(ns)
    assert rc == 0
    assert not (outp / 'prefs.json').exists()
    _ = review_outcomes()
    prefs2 = load_prefs()
    assert isinstance(prefs2.get('schema_version'), (int, float))


def test_policy_interplay_safe_vs_fast(monkeypatch):
    # Same underlying stats: samples=5 delta=0.08 => safe softens, fast allows
    prefs = {
        'tool_success_rate_by_intent': {'web.search': {'a': 0.80, 'b': 0.72}},
        'tool_sample_weight_by_intent': {'web.search': {'a': 5.0, 'b': 5.0}},
    }
    import app.parent.driver as drv
    monkeypatch.setattr(drv, '_load_prefs', lambda: prefs, raising=False)
    from app.parent.driver import ParentDriver, ParentLLM
    from app.parent.tools_manifest import ToolsManifest
    # SAFE should soften
    os.environ['NERION_POLICY'] = 'safe'
    os.environ['NERION_INTENT_HINT'] = 'web.search'
    class _Stub(ParentLLM):
        def complete(self, messages):
            sys = messages[0]['content']
            assert 'softened (low confidence)' in sys
            return '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'
    ParentDriver(llm=_Stub(), tools=ToolsManifest(tools=[])).plan_and_route('q?')
    # FAST should not soften
    os.environ['NERION_POLICY'] = 'fast'
    class _Stub2(ParentLLM):
        def complete(self, messages):
            sys = messages[0]['content']
            assert 'softened' not in sys
            return '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'
    ParentDriver(llm=_Stub2(), tools=ToolsManifest(tools=[])).plan_and_route('q?')

