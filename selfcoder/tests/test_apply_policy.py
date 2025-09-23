from selfcoder.planner.apply_policy import apply_allowed, evaluate_apply_policy


def test_apply_policy_defaults_to_review_without_metadata():
    plan = {"actions": []}
    decision = evaluate_apply_policy(plan)
    assert decision.decision == "review"
    assert any("No architect" in r for r in decision.reasons)
    assert not apply_allowed(decision)


def test_apply_policy_blocks_on_risk_threshold():
    plan = {
        "metadata": {
            "architect_brief": {
                "decision": "auto",
                "policy": "balanced",
                "gating": {
                    "risk": {
                        "score": 12.0,
                        "block": 10.0,
                        "review": 6.0,
                    }
                },
            }
        }
    }
    decision = evaluate_apply_policy(plan)
    assert decision.decision == "block"
    assert not apply_allowed(decision)


def test_apply_policy_allows_force_override():
    plan = {"metadata": {"architect_brief": {"decision": "review"}}}
    decision = evaluate_apply_policy(plan)
    assert decision.decision == "review"
    assert apply_allowed(decision, force=True)
