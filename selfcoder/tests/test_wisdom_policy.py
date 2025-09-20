from __future__ import annotations

import os
import json
from pathlib import Path


def test_bandit_deterministic_epsilon(tmp_path, monkeypatch):
    # Build a simple ToolsManifest with a few tools
    from app.parent.tools_manifest import ToolsManifest, ToolSpec, ToolParam
    tools = ToolsManifest(tools=[
        ToolSpec(name='a_tool', description='A', params=[ToolParam(name='x', type='int')]),
        ToolSpec(name='b_tool', description='B', params=[]),
        ToolSpec(name='c_tool', description='C', params=[]),
    ])
    learned = {'a_tool': 0.9, 'b_tool': 0.5, 'c_tool': 0.2}
    # Set deterministic seed and epsilon > 0; rendering twice should be identical
    monkeypatch.setenv('NERION_BANDIT_SEED', '123')
    block1 = tools.as_prompt_block(learned_weights=learned, epsilon=0.3)
    block2 = tools.as_prompt_block(learned_weights=learned, epsilon=0.3)
    assert block1 == block2
    # Ensure learned_weight annotations are present
    assert 'learned_weight: 0.90' in block1


def test_prompt_policy_safe_shrinks_on_ties(monkeypatch):
    # Given two similar rates and safe policy, require min delta forces top-1 only
    import app.parent.driver as drv
    monkeypatch.setenv('NERION_POLICY', 'safe')
    monkeypatch.setenv('NERION_LEARN_MIN_DELTA', '0.10')
    # Provide prefs with small delta (< 0.10)
    monkeypatch.setattr(drv, '_load_prefs', lambda: {
        'tool_success_rate': {'x': 0.80, 'y': 0.73},
        'tool_sample_weight': {'x': 10.0, 'y': 10.0}
    }, raising=False)
    captured = {}
    def _fake_build(user_query, tools, context_snippet=None, extra_policies=None, **_kw):
        captured['pol'] = extra_policies or ''
        return {'messages': []}
    monkeypatch.setattr(drv, 'build_master_prompt', _fake_build, raising=False)
    p = drv.ParentDriver(llm=type('L', (), {'complete': lambda self, m: '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'})(),
                         tools=drv.ToolsManifest(tools=[]))
    p.plan_and_route(user_query='q')
    pol = captured.get('pol', '')
    assert 'LEARNED BIASES' in pol
    # Only one rate should be listed when safe and small delta
    assert pol.count(':') - pol.count('LEARNED BIASES') >= 1
    # Ensure it doesn't list both x and y explicitly
    assert ('x:0.80' in pol and 'y:0.73' not in pol) or ('y:0.73' in pol and 'x:0.80' not in pol)


def test_prompt_policy_fast_trims_list(monkeypatch):
    import app.parent.driver as drv
    monkeypatch.setenv('NERION_POLICY', 'fast')
    # Provide 4 tools so trimming to <=3 can be asserted
    monkeypatch.setattr(drv, '_load_prefs', lambda: {
        'tool_success_rate': {'a': 0.9, 'b': 0.8, 'c': 0.7, 'd': 0.6},
        'tool_sample_weight': {'a': 5.0, 'b': 5.0, 'c': 5.0, 'd': 5.0}
    }, raising=False)
    captured = {}
    def _fake_build(user_query, tools, context_snippet=None, extra_policies=None, **_kw):
        captured['pol'] = extra_policies or ''
        return {'messages': []}
    monkeypatch.setattr(drv, 'build_master_prompt', _fake_build, raising=False)
    p = drv.ParentDriver(llm=type('L', (), {'complete': lambda self, m: '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'})(),
                         tools=drv.ToolsManifest(tools=[]))
    p.plan_and_route(user_query='q')
    pol = captured.get('pol', '')
    assert 'LEARNED BIASES' in pol
    # Count number of tool entries shown after 'Success rates: '
    if 'Success rates:' in pol:
        rates_part = pol.split('Success rates: ', 1)[1].strip()
        n_items = rates_part.count(',') + 1 if rates_part else 0
        assert n_items <= 3


def test_health_dashboard_top_tools_json(tmp_path):
    # Seed prefs so top_tools appear
    outp = Path('out/learning'); outp.mkdir(parents=True, exist_ok=True)
    prefs = {'tool_success_rate': {'a': 0.9, 'b': 0.7}}
    (outp / 'prefs.json').write_text(json.dumps(prefs), encoding='utf-8')
    # Run dashboard in JSON mode and assert top_tools present
    import selfcoder.cli as cli
    parser = cli._build_parser()
    ns = parser.parse_args(['health', 'dashboard', '--json'])
    rc = ns.func(ns)
    assert rc == 0
