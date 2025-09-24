import json
import subprocess
import sys
from pathlib import Path

from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.registry import CATALOG_FILENAME
from nerion_digital_physicist.infrastructure.telemetry import TELEMETRY_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS_REPORT = REPO_ROOT / "nerion_digital_physicist" / "experiments" / "analysis.py"


def _write_jsonl(path: Path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_metrics_report_cli(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / CATALOG_FILENAME,
        [
            {"task_id": "1", "template_id": "tpl"},
        ],
    )

    store = ReplayStore(tmp_path)
    store.append(task_id="1", template_id="tpl", status="pending", surprise=0.3)

    _write_jsonl(
        tmp_path / TELEMETRY_FILENAME,
        [
            {
                "event_type": "generation_run_complete",
                "payload": {"duration_seconds": 1.2},
                "created_at": "2025-09-24T00:00:00Z",
            }
        ],
    )

    output_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        str(METRICS_REPORT),
        str(tmp_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["total_tasks"] == 1
    assert report["generation_runs"] == 1
