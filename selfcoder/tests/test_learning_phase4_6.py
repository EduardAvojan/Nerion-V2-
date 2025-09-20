from __future__ import annotations

import json
import os
from pathlib import Path


def _build_parser():
    import selfcoder.cli as cli
    return cli._build_parser()


def test_parent_bias_softened_low_confidence(monkeypatch):
    # Low samples + overlapping CIs should soften per-intent bias
    prefs = {
        'tool_success_rate_by_intent': {'web.search': {'a': 0.70, 'b': 0.69}},
        'tool_sample_weight_by_intent': {'web.search': {'a': 2.0, 'b': 2.0}},
        'tool_success_rate': {'a': 0.70, 'b': 0.69},
        'tool_sample_weight': {'a': 10.0, 'b': 10.0},
    }
    import app.parent.driver as drv
    monkeypatch.setattr(drv, "_load_prefs", lambda: prefs, raising=False)
    os.environ['NERION_INTENT_HINT'] = 'web.search'
    from app.parent.driver import ParentDriver, ParentLLM
    from app.parent.tools_manifest import ToolsManifest
    class _Stub(ParentLLM):
        def complete(self, messages):
            sys = messages[0]['content']
            assert 'LEARNED BIASES' in sys
            assert 'softened (low confidence)' in sys
            return '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'
    ParentDriver(llm=_Stub(), tools=ToolsManifest(tools=[])).plan_and_route('q?')


def test_parent_bias_confident(monkeypatch):
    # High samples + clear delta should not soften
    prefs = {
        'tool_success_rate_by_intent': {'web.search': {'a': 0.90, 'b': 0.70}},
        'tool_sample_weight_by_intent': {'web.search': {'a': 20.0, 'b': 20.0}},
    }
    import app.parent.driver as drv
    monkeypatch.setattr(drv, "_load_prefs", lambda: prefs, raising=False)
    os.environ['NERION_INTENT_HINT'] = 'web.search'
    from app.parent.driver import ParentDriver, ParentLLM
    from app.parent.tools_manifest import ToolsManifest
    class _Stub(ParentLLM):
        def complete(self, messages):
            sys = messages[0]['content']
            assert 'LEARNED BIASES' in sys
            assert 'softened' not in sys
            return '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'
    ParentDriver(llm=_Stub(), tools=ToolsManifest(tools=[])).plan_and_route('q?')


def test_learn_show_intent_explain(tmp_path, capsys):
    # Seed prefs with per-intent rates and samples
    outp = Path('out/learning'); outp.mkdir(parents=True, exist_ok=True)
    prefs = {
        'tool_success_rate_by_intent': {'web.search': {'x': 0.8, 'y': 0.6}},
        'tool_sample_weight_by_intent': {'web.search': {'x': 10.0, 'y': 8.0}},
    }
    (outp / 'prefs.json').write_text(json.dumps(prefs), encoding='utf-8')
    parser = _build_parser()
    ns = parser.parse_args(['learn', 'show', '--intent', 'web.search', '--explain'])
    rc = ns.func(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert 'CI95%' in out and 'delta=' in out


def test_health_dashboard_json_contains_new_sections(tmp_path, capsys):
    outp = Path('out/learning'); outp.mkdir(parents=True, exist_ok=True)
    prefs = {
        'tool_success_rate': {'a': 0.9},
        'tool_success_rate_by_intent': {'web.search': {'a': 0.9, 'b': 0.7}},
        'experiments': {'eval1': {'control': {'success_rate': 0.5, 'n': 2, 'avg_latency_ms': None}}},
    }
    (outp / 'prefs.json').write_text(json.dumps(prefs), encoding='utf-8')
    parser = _build_parser()
    ns = parser.parse_args(['health', 'dashboard', '--json'])
    rc = ns.func(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert 'per_intent' in data and 'experiments' in data


def test_self_learn_fine_tune_writes_artifacts(tmp_path, capsys):
    # Add a couple of experience lines so dataset is non-empty
    log_dir = Path('out/experience'); log_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ts": 1, "user_query": "q1", "parent_decision": {"intent": "web.search"}, "action_taken": {"steps": [{"tool": "w"}]}, "outcome_success": True},
        {"ts": 2, "user_query": "q2", "parent_decision": {"intent": "local.get_current_time"}, "action_taken": {"steps": [{"tool": "t"}]}, "outcome_success": False},
    ]
    with (log_dir / 'log.jsonl').open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    parser = _build_parser()
    ns = parser.parse_args(['self-learn', 'fine-tune', '--data-since', '30d', '--domain', 'all'])
    rc = ns.func(ns)
    assert rc == 0
    root = Path('out/self_learn')
    assert (root / 'dataset.jsonl').exists()
    assert (root / 'train_config.json').exists()
    ds_lines = (root / 'dataset.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(ds_lines) >= 1

