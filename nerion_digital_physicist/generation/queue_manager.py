"""In-memory queue manager for distributed task generation."""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, Optional

QUEUE_STATE_FILE = "generation_queue.json"
_GLOBAL_LOCK = threading.Lock()


@dataclass
class QueueItem:
    request_id: str
    count: int
    template_weights: Optional[Dict[str, float]] = None


class GenerationQueue:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / QUEUE_STATE_FILE
        self._lock = threading.Lock()
        self._queue: Deque[QueueItem] = deque()
        self._load_state()

    def enqueue(self, item: QueueItem) -> None:
        with _GLOBAL_LOCK:
            self._load_state()
            self._queue.append(item)
            self._save_state()

    def dequeue(self) -> Optional[QueueItem]:
        with _GLOBAL_LOCK:
            self._load_state()
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._save_state()
            return item

    def peek(self) -> Optional[QueueItem]:
        with self._lock:
            return self._queue[0] if self._queue else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._queue = deque()
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        items = [QueueItem(**entry) for entry in data.get("items", [])]
        self._queue = deque(items)

    def _save_state(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {"items": [asdict(item) for item in self._queue]}
        self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
