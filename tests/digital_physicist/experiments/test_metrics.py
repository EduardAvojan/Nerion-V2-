import json
from pathlib import Path

from nerion_digital_physicist.experiments.metrics import summarize_metrics
from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.registry import CATALOG_FILENAME
from nerion_digital_physicist.infrastructure.telemetry import TELEMETRY_FILENAME


def _write_jsonl(path: Path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_summarize_metrics(tmp_path: Path) -> None:
    # Manifests
    catalog_entries = [
        {
            "task_id": "a",
            "template_id": "tpl1",
        },
        {
            "task_id": "b",
            "template_id": "tpl2",
        },
        {
            "task_id": "c",
            "template_id": "tpl1",
        },
    ]
    _write_jsonl(tmp_path / CATALOG_FILENAME, catalog_entries)

    # Replay
    store = ReplayStore(tmp_path)
    exp1 = store.append(task_id="a", template_id="tpl1", status="solved", surprise=0.2)
    exp2 = store.append(task_id="b", template_id="tpl2", status="failed", surprise=1.1)
    store.update(exp1.experience_id, status="solved", surprise=0.2)
    store.update(exp2.experience_id, status="failed", surprise=1.1)

    # Telemetry
    telemetry_entries = [
        {
            "event_type": "generation_run_complete",
            "payload": {"duration_seconds": 0.5},
            "created_at": "2025-09-24T00:00:00Z",
        },
        {
            "event_type": "generation_run_complete",
            "payload": {"duration_seconds": 0.7},
            "created_at": "2025-09-24T00:01:00Z",
        },
    ]
    _write_jsonl(tmp_path / TELEMETRY_FILENAME, telemetry_entries)

    summary = summarize_metrics(tmp_path)
    assert summary.total_tasks == 3
    assert summary.templates["tpl1"] == 2
    assert summary.replay_status_counts["solved"] == 1
    assert summary.replay_status_counts["failed"] == 1
    assert summary.average_surprise == (0.2 + 1.1) / 2
    assert summary.average_duration == (0.5 + 0.7) / 2
    assert summary.generation_runs == 2
