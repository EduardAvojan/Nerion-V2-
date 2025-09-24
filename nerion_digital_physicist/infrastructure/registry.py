"""Registry utilities for environment generator manifests."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import json
import uuid
from json import JSONDecodeError

ALLOWED_STATUSES = {
    "generated",
    "invalid",
    "claimed",
    "solved",
    "archived",
}

CATALOG_FILENAME = "task_catalog.jsonl"


@dataclass
class TaskManifest:
    task_id: str
    template_id: str
    seed: int
    parameters: Dict[str, Any]
    artifacts_path: str
    checksum: str
    status: str
    created_at: str

    def __post_init__(self) -> None:  # pragma: no cover - exercised via tests
        self.validate()

    @classmethod
    def new(
        cls,
        template_id: str,
        seed: int,
        parameters: Dict[str, Any],
        artifacts_path: Path,
        checksum: str,
        status: str = "generated",
        created_at: Optional[str] = None,
    ) -> "TaskManifest":
        manifest_id = uuid.uuid4().hex
        timestamp = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(
            task_id=manifest_id,
            template_id=template_id,
            seed=seed,
            parameters=parameters,
            artifacts_path=str(artifacts_path),
            checksum=checksum,
            status=status,
            created_at=timestamp,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def validate(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be a non-empty string")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dictionary")
        if not isinstance(self.artifacts_path, str) or not self.artifacts_path:
            raise ValueError("artifacts_path must be a non-empty string")
        if not isinstance(self.checksum, str) or not self.checksum:
            raise ValueError("checksum must be a non-empty string")
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"status '{self.status}' is not permitted")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be an ISO8601 string")


class ManifestRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.catalog_path = root / CATALOG_FILENAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path.touch(exist_ok=True)

    def append(self, manifest: TaskManifest) -> None:
        manifest.validate()
        with self.catalog_path.open("a", encoding="utf-8") as fh:
            fh.write(manifest.to_json())
            fh.write("\n")

    def load(self) -> Iterator[TaskManifest]:
        with self.catalog_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    manifest = TaskManifest(**data)
                except (JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError("Invalid manifest entry") from exc
                else:
                    yield manifest

    def find_by_template(self, template_id: str) -> Iterable[TaskManifest]:
        return (m for m in self.load() if m.template_id == template_id)

    def latest(self, limit: int = 10) -> Iterable[TaskManifest]:
        items = list(self.load())
        return items[-limit:]

    def _load_all(self) -> List[TaskManifest]:
        return list(self.load())

    def _write_all(self, manifests: Iterable[TaskManifest]) -> None:
        with self.catalog_path.open("w", encoding="utf-8") as fh:
            for manifest in manifests:
                manifest.validate()
                fh.write(manifest.to_json())
                fh.write("\n")

    def _update_fields(self, task_id: str, fields: Dict[str, Any]) -> TaskManifest:
        manifests = self._load_all()
        updated: Optional[TaskManifest] = None

        for index, manifest in enumerate(manifests):
            if manifest.task_id != task_id:
                continue
            data = asdict(manifest)
            data.update(fields)
            updated = TaskManifest(**data)
            manifests[index] = updated
            break

        if updated is None:
            raise KeyError(f"Task '{task_id}' not found in registry")

        self._write_all(manifests)
        return updated

    def list_by_status(self, *statuses: str) -> List[TaskManifest]:
        if not statuses:
            return self._load_all()
        status_set = set(statuses)
        invalid = status_set - ALLOWED_STATUSES
        if invalid:
            raise ValueError(f"Unknown statuses requested: {sorted(invalid)}")
        return [manifest for manifest in self.load() if manifest.status in status_set]

    def set_status(self, task_id: str, status: str) -> TaskManifest:
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"status '{status}' is not permitted")
        return self._update_fields(task_id, {"status": status})

    def claim_next(self, template_id: Optional[str] = None) -> Optional[TaskManifest]:
        manifests = self._load_all()
        for manifest in manifests:
            if manifest.status != "generated":
                continue
            if template_id and manifest.template_id != template_id:
                continue
            return self.set_status(manifest.task_id, "claimed")
        return None
