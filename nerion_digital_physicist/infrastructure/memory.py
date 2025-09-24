"""Persistent replay store for Phase 3."""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPLAY_FILENAME = "replay.jsonl"
DEFAULT_PRIORITY = 1.0
VALID_STATUSES = {"pending", "solved", "failed"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Experience:
    experience_id: str
    task_id: str
    template_id: str
    status: str
    surprise: Optional[float] = None
    priority: float = DEFAULT_PRIORITY
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class ReplayStore:
    """JSONL-backed replay buffer with priority sampling."""

    def __init__(self, root: Path):
        self.root = root
        self.file_path = root / REPLAY_FILENAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_path.touch(exist_ok=True)

    def _load_all(self) -> List[Experience]:
        experiences: List[Experience] = []
        with self.file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                experiences.append(Experience(**data))
        return experiences

    def _write_all(self, experiences: Iterable[Experience]) -> None:
        with self.file_path.open("w", encoding="utf-8") as fh:
            for exp in experiences:
                fh.write(exp.to_json())
                fh.write("\n")

    def append(
        self,
        task_id: str,
        template_id: str,
        status: str = "pending",
        surprise: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
    ) -> Experience:
        if status not in VALID_STATUSES:
            raise ValueError(f"status '{status}' is not valid")
        exp = Experience(
            experience_id=uuid.uuid4().hex,
            task_id=task_id,
            template_id=template_id,
            status=status,
            surprise=surprise,
            priority=priority if priority is not None else self._default_priority(surprise),
            metadata=metadata or {},
        )
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(exp.to_json())
            fh.write("\n")
        return exp

    def _default_priority(self, surprise: Optional[float]) -> float:
        if surprise is None:
            return DEFAULT_PRIORITY
        return max(0.1, min(10.0, 1.0 + float(surprise)))

    def update(self, experience_id: str, **fields: Any) -> Experience:
        experiences = self._load_all()
        updated: Optional[Experience] = None
        for idx, exp in enumerate(experiences):
            if exp.experience_id != experience_id:
                continue

            merged_fields = dict(fields)
            if "metadata" in merged_fields:
                metadata_field = merged_fields["metadata"]
                if metadata_field is None:
                    merged_fields["metadata"] = {}
                else:
                    merged_metadata = dict(exp.metadata)
                    merged_metadata.update(metadata_field)
                    merged_fields["metadata"] = merged_metadata

            data = asdict(exp)
            data.update(merged_fields)
            data["updated_at"] = utc_timestamp()
            if "status" in merged_fields and data["status"] not in VALID_STATUSES:
                raise ValueError(f"status '{data['status']}' is not valid")

            updated = Experience(**data)
            experiences[idx] = updated
            break
        if updated is None:
            raise KeyError(f"Experience {experience_id} not found")
        self._write_all(experiences)
        return updated

    def load(self) -> Iterable[Experience]:
        return self._load_all()

    def sample(self, k: int, strategy: str = "priority") -> List[Experience]:
        experiences = self._load_all()
        if not experiences or k <= 0:
            return []
        k = min(k, len(experiences))
        if strategy == "random":
            return random.sample(experiences, k)
        weights = [max(exp.priority, 0.001) for exp in experiences]
        chosen = random.choices(experiences, weights=weights, k=k)
        # random.choices can repeat entries; ensure uniqueness preserving order
        seen = set()
        unique: List[Experience] = []
        for exp in chosen:
            if exp.experience_id in seen:
                continue
            seen.add(exp.experience_id)
            unique.append(exp)
            if len(unique) == k:
                break
        if len(unique) < k:
            for exp in experiences:
                if exp.experience_id in seen:
                    continue
                unique.append(exp)
                if len(unique) == k:
                    break
        return unique

    def find_by_task(self, task_id: str) -> Experience | None:
        for exp in self._load_all():
            if exp.task_id == task_id:
                return exp
        return None
