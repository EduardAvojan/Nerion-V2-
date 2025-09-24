import json
from pathlib import Path

import pytest

from nerion_digital_physicist.infrastructure.registry import (
    ManifestRegistry,
    TaskManifest,
    ALLOWED_STATUSES,
)


def test_append_and_load_manifest(tmp_path: Path) -> None:
    registry = ManifestRegistry(tmp_path)
    manifest = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=123,
        parameters={"length": 5, "operations": ["add", "multiply"]},
        artifacts_path=tmp_path / "task-1",
        checksum="deadbeef",
    )

    registry.append(manifest)
    loaded = list(registry.load())

    assert len(loaded) == 1
    assert loaded[0] == manifest


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError):
        TaskManifest.new(
            template_id="alg_arithmetic_pipeline",
            seed=99,
            parameters={},
            artifacts_path=Path("/tmp/task"),
            checksum="badc0de",
            status="unknown",
        )


def test_registry_rejects_malformed_payload(tmp_path: Path) -> None:
    registry = ManifestRegistry(tmp_path)

    # Manually inject a malformed record (missing required fields)
    bad_entry = json.dumps({"foo": "bar"})
    catalog_path = tmp_path / "task_catalog.jsonl"
    catalog_path.write_text(bad_entry + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        list(registry.load())


def test_allowed_statuses_documented() -> None:
    # Sanity check to ensure expected lifecycle statuses remain available.
    expected = {"generated", "invalid", "claimed", "solved", "archived"}
    assert expected == ALLOWED_STATUSES


def test_set_status_updates_manifest(tmp_path: Path) -> None:
    registry = ManifestRegistry(tmp_path)
    manifest = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=1,
        parameters={},
        artifacts_path=tmp_path / "task-1",
        checksum="abc123",
    )
    registry.append(manifest)

    updated = registry.set_status(manifest.task_id, "claimed")
    assert updated.status == "claimed"

    loaded = list(registry.load())
    assert loaded[0].status == "claimed"


def test_list_by_status_filters(tmp_path: Path) -> None:
    registry = ManifestRegistry(tmp_path)
    manifest_a = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=1,
        parameters={},
        artifacts_path=tmp_path / "task-a",
        checksum="aaa",
    )
    manifest_b = TaskManifest.new(
        template_id="alg_arithmetic_pipeline",
        seed=2,
        parameters={},
        artifacts_path=tmp_path / "task-b",
        checksum="bbb",
        status="archived",
    )
    registry.append(manifest_a)
    registry.append(manifest_b)

    archived = registry.list_by_status("archived")
    assert len(archived) == 1
    assert archived[0].task_id == manifest_b.task_id
