from pathlib import Path

from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.outcomes import log_outcome
from nerion_digital_physicist.infrastructure.telemetry import TelemetryLogger


def test_log_outcome_updates_replay(tmp_path: Path) -> None:
    replay = ReplayStore(tmp_path)
    telemetry = TelemetryLogger(tmp_path)
    experience = replay.append(task_id="task1", template_id="tpl")

    log_outcome(
        replay,
        telemetry,
        experience_id=experience.experience_id,
        status="solved",
        surprise=0.8,
        extra_metadata={"latency_ms": 12},
    )

    reloaded = replay.find_by_task("task1")
    assert reloaded.status == "solved"
    assert reloaded.surprise == 0.8
    assert reloaded.metadata == {"latency_ms": 12}

    telemetry_lines = (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any("experience_updated" in line for line in telemetry_lines)


def test_log_outcome_resolves_by_task_id(tmp_path: Path) -> None:
    replay = ReplayStore(tmp_path)
    replay.append(task_id="task2", template_id="tpl")

    log_outcome(
        replay,
        telemetry=None,
        experience_id=None,
        task_id="task2",
        status="failed",
        surprise=1.2,
    )

    reloaded = replay.find_by_task("task2")
    assert reloaded.status == "failed"
    assert reloaded.priority > 1.0
