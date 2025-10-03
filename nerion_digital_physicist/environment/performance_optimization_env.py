"""
This module contains the PerformanceOptimizationEnvironment, which is responsible for running performance optimization lessons.
"""
from __future__ import annotations

import subprocess
import sys
import timeit
from pathlib import Path

from nerion_digital_physicist.environment.types import TestOutcome


def _run_benchmark_in_sandbox(code: str, test_code: str, number: int = 1000) -> float:
    """Runs the given code and test code in a sandboxed environment and returns the execution time."""
    with open("sandbox.py", "w") as f:
        f.write(code)
    with open("test_sandbox.py", "w") as f:
        f.write(test_code)

    return timeit.timeit(
        stmt="subprocess.run([sys.executable, \"-m\", \"pytest\", \"test_sandbox.py\"])",
        setup="import subprocess; import sys",
        number=number,
    )

class PerformanceOptimizationEnvironment:
    """An environment for running performance optimization lessons."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def step(self, inefficient_code: str, test_code: str, optimized_code: str) -> tuple[float, float]:
        """
        Runs the test code against the inefficient code and the optimized code and returns the execution times.

        Returns:
            A tuple containing the execution time for the inefficient code and the optimized code.
        """
        inefficient_time = _run_benchmark_in_sandbox(inefficient_code, test_code)
        optimized_time = _run_benchmark_in_sandbox(optimized_code, test_code)

        return inefficient_time, optimized_time
