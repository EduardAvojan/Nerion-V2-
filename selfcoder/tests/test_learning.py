from __future__ import annotations

import json
from pathlib import Path


def test_learn_review_updates_prefs(tmp_path, monkeypatch):
    # Create synthetic experience log with two tools and mixed outcomes
    out_dir = Path('out/experience'); out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'log.jsonl'
    rows = [
        {"ts": 1, "outcome_success": True,  "action_taken": {"steps": [{"tool": "get_current_time", "duration_ms": 10}]}},
        {"ts": 2, "outcome_success": True,  "action_taken": {"steps": [{"tool": "get_current_time", "duration_ms": 12}]}},
        {"ts": 3, "outcome_success": False, "action_taken": {"steps": [{"tool": "web_search", "duration_ms": 80}]}},
        {"ts": 4, "outcome_success": True,  "action_taken": {"steps": [{"tool": "web_search", "duration_ms": 70}]}}
    ]
    with log.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # Relax min-samples for this small synthetic set
    monkeypatch.setenv('NERION_LEARN_MIN_SAMPLES', '1')
    # Run learn review
    from selfcoder.learning.continuous import review_outcomes, load_prefs
    out = review_outcomes()
    assert isinstance(out, dict)
    p = load_prefs()
    tsr = (p.get('tool_success_rate') or {})
    assert isinstance(tsr, dict) and tsr, 'tool_success_rate empty'
    # Expect both tools present with rates 0..1
    assert 0.0 <= tsr.get('get_current_time', 0.0) <= 1.0
    assert 0.0 <= tsr.get('web_search', 0.0) <= 1.0


def test_learn_acceptance_mixed_counts(tmp_path, monkeypatch):
    # Build 10 rows for web_search (7 ok, 3 fail) and 9 ok for get_current_time
    out_dir = Path('out/experience'); out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'log.jsonl'
    rows = []
    for i in range(7):
        rows.append({"ts": i+1, "outcome_success": True,  "action_taken": {"steps": [{"tool": "web_search"}]}})
    for i in range(3):
        rows.append({"ts": 8+i, "outcome_success": False, "action_taken": {"steps": [{"tool": "web_search"}]}})
    for i in range(9):
        rows.append({"ts": 20+i, "outcome_success": True,  "action_taken": {"steps": [{"tool": "get_current_time"}]}})
    with log.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    # Ensure min samples default (3) passes; run
    from selfcoder.learning.continuous import review_outcomes, load_prefs
    out = review_outcomes()
    p = load_prefs()
    tsr = p.get('tool_success_rate') or {}
    # web_search ~ between 0.6 and 0.85 (smoothing + no decay)
    ws = tsr.get('web_search', 0.0)
    assert 0.6 <= ws <= 0.85
    # get_current_time ~ 0.90 (10/11) within [0.85, 0.95]
    gct = tsr.get('get_current_time', 0.0)
    assert 0.85 <= gct <= 0.95
def test_parent_prompt_injects_bias_block(tmp_path, monkeypatch):
    # Write prefs with tool_success_rate and samples
    outp = Path('out/learning'); outp.mkdir(parents=True, exist_ok=True)
    prefs = {
        'tool_success_rate': {'get_current_time': 0.98, 'web_search': 0.76},
        'tool_sample_weight': {'get_current_time': 5.0, 'web_search': 5.0},
    }
    (outp / 'prefs.json').write_text(json.dumps(prefs), encoding='utf-8')

    # Stub LLM to capture messages and return a small JSON decision
    from app.parent.driver import ParentDriver, ParentLLM
    from app.parent.tools_manifest import ToolsManifest
    class _StubLLM(ParentLLM):
        def complete(self, messages):
            # Assert our bias block is included in the system prompt
            sys = messages[0]['content']
            assert 'LEARNED BIASES' in sys
            assert 'Success rates:' in sys
            return '{"intent":"clarify","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":null}'
    drv = ParentDriver(llm=_StubLLM(), tools=ToolsManifest(tools=[]))
    out = drv.plan_and_route(user_query='time?', context_snippet=None, extra_policies=None)
    assert out.intent in {'clarify'}


def test_learn_per_intent_rates(tmp_path, monkeypatch):
    # Create mixed logs across two intents with different tools
    out_dir = Path('out/experience'); out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'log.jsonl'
    rows = [
        # intent: local.get_current_time
        {"ts": 1, "outcome_success": True,  "parent_decision": {"intent": "local.get_current_time"},
         "action_taken": {"steps": [{"tool": "get_current_time"}]}},
        {"ts": 2, "outcome_success": True,  "parent_decision": {"intent": "local.get_current_time"},
         "action_taken": {"steps": [{"tool": "get_current_time"}]}},
        # intent: web.search
        {"ts": 3, "outcome_success": True,  "parent_decision": {"intent": "web.search"},
         "action_taken": {"steps": [{"tool": "web_search"}]}},
        {"ts": 4, "outcome_success": False, "parent_decision": {"intent": "web.search"},
         "action_taken": {"steps": [{"tool": "web_search"}]}},
    ]
    with log.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # Relax sample threshold to 1 so small sets appear
    monkeypatch.setenv('NERION_LEARN_MIN_SAMPLES', '1')
    from selfcoder.learning.continuous import review_outcomes, load_prefs
    _ = review_outcomes()
    p = load_prefs()
    by_int = p.get('tool_success_rate_by_intent') or {}
    assert isinstance(by_int, dict) and by_int, 'tool_success_rate_by_intent empty'
    # Expect both intents present
    assert 'local.get_current_time' in by_int
    assert 'web.search' in by_int
    # Sanity: per-intent tool rates exist
    assert 0.0 <= (by_int['local.get_current_time'].get('get_current_time', 0.0)) <= 1.0
    assert 0.0 <= (by_int['web.search'].get('web_search', 0.0)) <= 1.0


def test_credit_assignment_biases_terminal_tool(tmp_path, monkeypatch):
    # Two-step plan: tool A then tool B; success => B should get higher credited rate
    out_dir = Path('out/experience'); out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'log.jsonl'
    rows = [
        {"ts": 1, "outcome_success": True,  "parent_decision": {"intent": "local.do"},
         "action_taken": {"steps": [
            {"tool": "A", "duration_ms": 50},
            {"tool": "B", "duration_ms": 60}
         ]}},
    ]
    with log.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setenv('NERION_LEARN_MIN_SAMPLES', '1')
    from selfcoder.learning.continuous import review_outcomes, load_prefs
    _ = review_outcomes()
    p = load_prefs()
    tsr = p.get('tool_success_rate') or {}
    assert tsr.get('B', 0.0) > tsr.get('A', 0.0)


def test_ab_experiment_aggregation(tmp_path, monkeypatch):
    # Logs with experiment arms; verify prefs['experiments'] has per-arm stats
    out_dir = Path('out/experience'); out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'log.jsonl'
    rows = [
        {"ts": 1, "outcome_success": True,  "experiment": {"name": "eval1", "arm": "control"}, "action_taken": {"steps": []}},
        {"ts": 2, "outcome_success": False, "experiment": {"name": "eval1", "arm": "treatment"}, "action_taken": {"steps": []}},
        {"ts": 3, "outcome_success": True,  "experiment": {"name": "eval1", "arm": "treatment"}, "action_taken": {"steps": []}},
    ]
    with log.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    from selfcoder.learning.continuous import review_outcomes, load_prefs
    _ = review_outcomes()
    p = load_prefs()
    exps = p.get('experiments') or {}
    assert 'eval1' in exps
    arms = exps['eval1']
    assert 'control' in arms and 'treatment' in arms
    # success_rate present for both arms
    assert arms['control'].get('success_rate') is not None
    assert arms['treatment'].get('success_rate') is not None
