import json
from pathlib import Path

from nerion_digital_physicist.generation.curriculum import (
    collect_template_stats,
    compute_curriculum_weights,
)
from nerion_digital_physicist.infrastructure.telemetry import TELEMETRY_FILENAME


def write_event(path: Path, template_id: str, duration: float) -> None:
    entry = {
        "event_type": "task_generated",
        "payload": {
            "template_id": template_id,
            "duration_seconds": duration,
        },
        "created_at": "2025-09-24T00:00:00Z",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry))
        fh.write("\n")


def test_collect_template_stats(tmp_path: Path) -> None:
    telemetry_path = tmp_path / TELEMETRY_FILENAME
    telemetry_path.touch()

    write_event(telemetry_path, "a", 1.0)
    write_event(telemetry_path, "a", 3.0)
    write_event(telemetry_path, "b", 2.0)

    stats = collect_template_stats(tmp_path)
    assert stats["a"].count == 2
    assert stats["a"].avg_duration == 2.0
    assert stats["b"].count == 1


def test_compute_curriculum_weights(tmp_path: Path) -> None:
    telemetry_path = tmp_path / TELEMETRY_FILENAME
    telemetry_path.touch()

    # template a: frequent but fast
    for _ in range(4):
        write_event(telemetry_path, "a", 0.5)
    # template b: rare and slow
    write_event(telemetry_path, "b", 4.0)

    weights = compute_curriculum_weights(tmp_path)
    assert weights["b"] > weights["a"]
    assert 0.5 <= weights["a"] <= 3.0
