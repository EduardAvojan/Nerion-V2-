"""
This module contains the BugFixingEnvironment, which is responsible for running bug-fixing lessons.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nerion_digital_physicist.environment.types import TestOutcome


def _run_test_in_sandbox(code: str, test_code: str) -> subprocess.CompletedProcess[str]:
    """Runs the given code and test code in a sandboxed environment."""
    with open("sandbox.py", "w") as f:
        f.write(code)
    with open("test_sandbox.py", "w") as f:
        f.write(test_code)

    return subprocess.run(
        [sys.executable, "-m", "pytest", "test_sandbox.py"],
        capture_output=True,
        text=True,
    )

class BugFixingEnvironment:
    """An environment for running bug-fixing lessons."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def step(self, buggy_code: str, test_code: str, fixed_code: str) -> tuple[TestOutcome, TestOutcome]:
        """
        Runs the test code against the buggy code and the fixed code.

        Returns:
            A tuple containing the TestOutcome for the buggy code and the fixed code.
        """
        buggy_proc = _run_test_in_sandbox(buggy_code, test_code)
        fixed_proc = _run_test_in_sandbox(fixed_code, test_code)

        buggy_outcome = TestOutcome(
            passed=1 if buggy_proc.returncode == 0 else 0,
            failed=1 if buggy_proc.returncode != 0 else 0,
        )

        fixed_outcome = TestOutcome(
            passed=1 if fixed_proc.returncode == 0 else 0,
            failed=1 if fixed_proc.returncode != 0 else 0,
        )

        return buggy_outcome, fixed_outcome
