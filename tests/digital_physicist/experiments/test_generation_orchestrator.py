import json
import subprocess
import sys
from pathlib import Path

from nerion_digital_physicist.generation.worker import enqueue_request
from nerion_digital_physicist.infrastructure.registry import CATALOG_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATOR = REPO_ROOT / "nerion_digital_physicist" / "generation" / "orchestrator.py"


def test_orchestrator_runs_queue(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_root = tmp_path / "output"
    enqueue_request(queue_root, count=1)
    enqueue_request(queue_root, count=1)

    cmd = [
        sys.executable,
        str(ORCHESTRATOR),
        str(queue_root),
        str(output_root),
        "--seed",
        "0",
    ]
    subprocess.run(cmd, check=True)

    catalog_path = output_root / CATALOG_FILENAME
    entries = catalog_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(entries) == 2
