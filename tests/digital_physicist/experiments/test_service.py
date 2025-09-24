import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATH = REPO_ROOT / "nerion_digital_physicist" / "generation" / "service.py"


def test_service_cli_generates_tasks(tmp_path: Path) -> None:
    output_dir = tmp_path / "tasks"
    cmd = [
        sys.executable,
        str(SERVICE_PATH),
        "2",
        "--seed",
        "7",
        "--output",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)

    cmd_curriculum = cmd + ["--curriculum"]
    subprocess.run(cmd_curriculum, check=True)

    catalog = output_dir / "task_catalog.jsonl"
    assert catalog.exists()

    lines = catalog.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # two runs

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 2
    paths = {entry["artifacts_path"] for entry in summary}
    for path in paths:
        src_file = Path(path) / "src" / "module.py"
        assert src_file.exists()

    telemetry_lines = (output_dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any("task_generated" in line for line in telemetry_lines)
    assert any("generation_run_complete" in line for line in telemetry_lines)

    replay_lines = (output_dir / "replay.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(replay_lines) == 4
