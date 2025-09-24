import torch
import pytest

import nerion_digital_physicist.agent.policy as agent_module
from nerion_digital_physicist.agent.policy import AgentV2, Action


def test_epsilon_greedy_triggers_random(monkeypatch):
    agent = AgentV2(learning_rate=0.01, epsilon=1.0)
    predictions = {
        Action.RENAME_LOCAL_VARIABLE_IN_ADD: torch.tensor([0.6, 0.4]),
        Action.CHANGE_OPERATOR_MULTIPLY_TO_ADD: torch.tensor([0.4, 0.6]),
        Action.IMPLEMENT_MULTIPLY_DOCSTRING: torch.tensor([0.2, 0.8]),
    }

    monkeypatch.setattr(agent_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(agent_module.random, "choice", lambda seq: seq[-1])

    decision = agent._select_action(predictions, verbose=False)

    assert decision.mode == "epsilon"
    assert decision.action == Action.IMPLEMENT_MULTIPLY_DOCSTRING
    assert decision.epsilon == pytest.approx(1.0)


def test_visit_counts_break_ties(monkeypatch):
    agent = AgentV2(learning_rate=0.01, epsilon=0.0)
    predictions = {
        Action.RENAME_LOCAL_VARIABLE_IN_ADD: torch.tensor([0.5, 0.5]),
        Action.CHANGE_OPERATOR_MULTIPLY_TO_ADD: torch.tensor([0.5, 0.5]),
        Action.IMPLEMENT_MULTIPLY_DOCSTRING: torch.tensor([0.5, 0.5]),
    }

    agent._action_visit_counts[Action.RENAME_LOCAL_VARIABLE_IN_ADD] = 5
    agent._action_visit_counts[Action.CHANGE_OPERATOR_MULTIPLY_TO_ADD] = 1
    agent._action_visit_counts[Action.IMPLEMENT_MULTIPLY_DOCSTRING] = 0

    monkeypatch.setattr(agent_module.random, "choice", lambda seq: seq[0])

    decision = agent._select_action(predictions, verbose=False)

    assert decision.mode == "curiosity"
    assert decision.action == Action.IMPLEMENT_MULTIPLY_DOCSTRING
    assert decision.uncertainty == pytest.approx(1.0)
    assert decision.entropy == pytest.approx(0.6931, rel=1e-3)
    assert decision.visit_count == 0


def test_forced_action_marks_scheduled():
    agent = AgentV2(learning_rate=0.01, epsilon=0.0)
    result = agent.run_episode(verbose=False, forced_action=Action.IMPLEMENT_MULTIPLY_DOCSTRING)

    assert result.action == Action.IMPLEMENT_MULTIPLY_DOCSTRING
    assert result.policy_mode == "scheduled"
    assert result.policy_entropy_bonus == agent.entropy_bonus


def test_adaptive_epsilon_updates():
    agent = AgentV2(
        learning_rate=0.01,
        epsilon=0.2,
        epsilon_min=0.05,
        epsilon_max=0.5,
        epsilon_decay=0.5,
        epsilon_step=0.1,
        adaptive_surprise_target=0.3,
        adaptive_epsilon=True,
    )

    # High surprise should increase epsilon up to epsilon_max
    start = agent.epsilon
    updated_high = agent._update_adaptive_parameters(0.8)
    assert updated_high > start
    assert updated_high <= agent.epsilon_max

    # Low surprise should decay epsilon but not below epsilon_min
    updated_low = agent._update_adaptive_parameters(0.0)
    assert updated_low <= updated_high
    assert updated_low >= agent.epsilon_min


def test_entropy_bonus_logged_in_episode_result():
    agent = AgentV2(learning_rate=0.01, epsilon=0.0, entropy_bonus=0.25)
    result = agent.run_episode(verbose=False, forced_action=Action.RENAME_LOCAL_VARIABLE_IN_ADD)
    assert result.policy_entropy_bonus == 0.25
