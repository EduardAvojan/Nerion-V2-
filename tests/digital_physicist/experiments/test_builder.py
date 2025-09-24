from pathlib import Path

from nerion_digital_physicist.generation.builder import TaskBuilder
from nerion_digital_physicist.infrastructure.registry import ManifestRegistry
from nerion_digital_physicist.infrastructure.telemetry import TelemetryLogger, TELEMETRY_FILENAME
from nerion_digital_physicist.infrastructure.memory import ReplayStore, REPLAY_FILENAME


def test_task_builder_generates_unique_directories(tmp_path: Path) -> None:
    registry = ManifestRegistry(tmp_path)
    telemetry = TelemetryLogger(tmp_path)
    replay_store = ReplayStore(tmp_path)
    builder = TaskBuilder(tmp_path, registry, telemetry=telemetry, replay=replay_store)

    manifest1 = builder.build_task("alg_arithmetic_pipeline", seed=0)
    manifest2 = builder.build_task("alg_arithmetic_pipeline", seed=1)

    assert manifest1.task_id != manifest2.task_id
    assert manifest1.artifacts_path != manifest2.artifacts_path

    artifacts1 = Path(manifest1.artifacts_path)
    artifacts2 = Path(manifest2.artifacts_path)
    assert artifacts1.exists() and artifacts2.exists()
    assert (artifacts1 / "src" / "module.py").exists()
    assert (artifacts2 / "tests" / "test_module.py").exists()

    catalog_entries = list(registry.load())
    assert len(catalog_entries) == 2

    telemetry_file = tmp_path / TELEMETRY_FILENAME
    lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all("task_generated" in line for line in lines)

    replay_file = tmp_path / REPLAY_FILENAME
    replay_lines = replay_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(replay_lines) == 2


def test_task_builder_overrides_parameters(tmp_path: Path) -> None:
    registry = ManifestRegistry(tmp_path)
    builder = TaskBuilder(tmp_path, registry)

    manifest = builder.build_task(
        "bug_off_by_one",
        seed=5,
        parameters={"bug_type": "missing_first"},
    )

    data = (Path(manifest.artifacts_path) / "metadata.json").read_text(encoding="utf-8")
    assert "missing_first" in data
