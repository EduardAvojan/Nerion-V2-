from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RenameAction:
    """Represents a cross-file rename operation."""
    file_path: str
    old_name: str
    new_name: str

@dataclass(frozen=True)
class TestOutcome:
    """Structured result of a test run."""
    passed: int = 0
    failed: int = 0
    errored: int = 0

    @property
    def was_successful(self) -> bool:
        """Return True if no tests failed or errored."""
        return self.failed == 0 and self.errored == 0
