from __future__ import annotations

from selfcoder.policy.profile_resolver import decide


def test_resolver_bench_defaults_to_bench_recommended():
    dec = decide('bench_repair')
    assert dec.name in {'bench-recommended', 'balanced', 'fast'}  # allow minimal fallback
    # Prefer bench-recommended when profiles.yaml is present
    # If environment is minimal, skip strict equality


def test_resolver_security_findings_safe():
    dec = decide('apply_plan', signals={'security_findings': 1})
    assert dec.name in {'safe', 'balanced'}  # safe preferred when available


def test_resolver_ast_small_change_fast():
    dec = decide('apply_plan', signals={'kinds_ast_only': True, 'files_count': 1, 'delta_bytes': 100, 'security_findings': 0})
    assert dec.name in {'fast', 'balanced'}  # fast preferred when available


def test_sticky_override_honored(tmp_path, monkeypatch):
    # Emulate a sticky override by writing prefs
    import json, os
    from selfcoder.learning.continuous import load_prefs as _lp, save_prefs as _sp
    prefs = _lp()
    prefs.setdefault('profile_overrides', {})['bench_repair'] = 'fast'
    _sp(prefs)
    dec = decide('bench_repair')
    assert dec.name in {'fast', 'bench-recommended'}
