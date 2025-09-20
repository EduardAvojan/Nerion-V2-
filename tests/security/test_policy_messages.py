from selfcoder.security.policy_messages import explain_policy

def test_explain_paths():
    out = explain_policy({"kind":"paths","rule":"deny_paths","value":"plugins/**","path":"plugins/foo.js"})
    assert "plugins/foo.js is denied" in out["reason"]
    assert "allow_paths" in out["hint"]
    assert out["doc"].endswith("#paths")
