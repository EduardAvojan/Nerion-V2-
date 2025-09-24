"""Tests for the experiment automation harness."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import nerion_digital_physicist.experiments.harness as runner
from nerion_digital_physicist.infrastructure.telemetry import TELEMETRY_FILENAME
from nerion_digital_physicist.infrastructure.memory import REPLAY_FILENAME
from nerion_digital_physicist.infrastructure.registry import ManifestRegistry, TaskManifest


def _load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_run_batch_records_replay_and_telemetry(tmp_path):
    agent = runner.AgentV2(epsilon=0.0)
    telemetry = runner.TelemetryLogger(tmp_path)
    replay = runner.ReplayStore(tmp_path)
    task_root = tmp_path / "tasks"
    registry = ManifestRegistry(task_root)

    artifacts_path = task_root / "alg" / "task-1"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    manifest = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=1,
        parameters={"length": 3},
        artifacts_path=artifacts_path,
        checksum="abc123",
    )
    registry.append(manifest)

    summary = runner.run_batch(
        agent,
        replay,
        telemetry,
        registry,
        batch_index=1,
        episodes_per_batch=1,
        quiet=True,
        starting_episode=1,
        replay_epochs=0,
        replay_batch_size=2,
        replay_learning_rate=0.01,
        generative_per_batch=0,
    )

    assert summary.episodes == 1
    experiences = list(replay.load())
    assert len(experiences) == 1
    assert experiences[0].status in {"solved", "failed"}

    events = _load_events(telemetry.file_path)
    event_types = {event["event_type"] for event in events}
    assert {"episode_completed", "batch_completed"}.issubset(event_types)

    final_status = registry.list_by_status("solved", "archived")
    assert len(final_status) == 1


def test_run_experiment_logs_completion(tmp_path):
    output_root = tmp_path / "experiment"
    task_root = tmp_path / "tasks"
    registry = ManifestRegistry(task_root)
    artifacts_path = task_root / "alg" / "task-1"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    manifest = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=1,
        parameters={"length": 3},
        artifacts_path=artifacts_path,
        checksum="abc123",
    )
    registry.append(manifest)
    args = Namespace(
        batches=1,
        episodes_per_batch=1,
        output_root=str(output_root),
        task_root=str(task_root),
        checkpoint_path=str(output_root / "brain.pth"),
        load_checkpoint=False,
        seed=0,
        quiet=True,
        policy_epsilon=0.0,
        replay_epochs=0,
        replay_batch_size=2,
        replay_learning_rate=0.01,
        generative_per_batch=0,
        entropy_bonus=0.0,
        adaptive_epsilon=False,
        epsilon_min=0.0,
        epsilon_max=None,
        epsilon_decay=1.0,
        epsilon_step=0.0,
        surprise_target=0.0,
        no_auto_generate=True,
    )

    runner.run_experiment(args)

    telemetry_events = _load_events(output_root / TELEMETRY_FILENAME)
    assert any(event["event_type"] == "experiment_completed" for event in telemetry_events)

    replay_path = output_root / REPLAY_FILENAME
    assert replay_path.exists()
    lines = [line for line in replay_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_replay_finetune_logs_event(tmp_path):
    agent = runner.AgentV2(epsilon=0.0)
    telemetry = runner.TelemetryLogger(tmp_path)
    replay = runner.ReplayStore(tmp_path)
    task_root = tmp_path / "tasks"
    registry = ManifestRegistry(task_root)

    artifacts_path = task_root / "alg" / "task-2"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    manifest = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=2,
        parameters={"length": 3},
        artifacts_path=artifacts_path,
        checksum="def456",
    )
    registry.append(manifest)

    runner.run_batch(
        agent,
        replay,
        telemetry,
        registry,
        batch_index=1,
        episodes_per_batch=1,
        quiet=True,
        starting_episode=1,
        replay_epochs=1,
        replay_batch_size=1,
        replay_learning_rate=0.01,
        generative_per_batch=0,
    )

    events = _load_events(telemetry.file_path)
    event_types = {event["event_type"] for event in events}
    assert "replay_finetune" in event_types


def test_run_batch_forces_generative_action(tmp_path):
    agent = runner.AgentV2(epsilon=0.0)
    telemetry = runner.TelemetryLogger(tmp_path)
    replay = runner.ReplayStore(tmp_path)
    task_root = tmp_path / "tasks"
    registry = ManifestRegistry(task_root)

    artifacts_path = task_root / "alg" / "task-3"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    manifest = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=3,
        parameters={"length": 3},
        artifacts_path=artifacts_path,
        checksum="ghi789",
    )
    registry.append(manifest)

    runner.run_batch(
        agent,
        replay,
        telemetry,
        registry,
        batch_index=1,
        episodes_per_batch=1,
        quiet=True,
        starting_episode=1,
        replay_epochs=0,
        replay_batch_size=1,
        replay_learning_rate=0.01,
        generative_per_batch=1,
    )

    experiences = list(replay.load())
    assert experiences, "Expected replay entry for scheduled episode"
    metadata = experiences[0].metadata
    assert metadata.get("policy_mode") == "scheduled"
    assert metadata.get("action") == "IMPLEMENT_MULTIPLY_DOCSTRING"
    assert "policy_entropy_bonus" in metadata
    assert "policy_epsilon_next" in metadata
