"""
This module contains the FeatureImplementationEnvironment, which is responsible for running feature implementation lessons.
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

class FeatureImplementationEnvironment:
    """An environment for running feature implementation lessons."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def step(self, initial_code: str, test_code: str, final_code: str) -> tuple[TestOutcome, TestOutcome]:
        """
        Runs the test code against the initial code and the final code.

        Returns:
            A tuple containing the TestOutcome for the initial code and the final code.
        """
        initial_proc = _run_test_in_sandbox(initial_code, test_code)
        final_proc = _run_test_in_sandbox(final_code, test_code)

        initial_outcome = TestOutcome(
            passed=1 if initial_proc.returncode == 0 else 0,
            failed=1 if initial_proc.returncode != 0 else 0,
        )

        final_outcome = TestOutcome(
            passed=1 if final_proc.returncode == 0 else 0,
            failed=1 if final_proc.returncode != 0 else 0,
        )

        return initial_outcome, final_outcome
