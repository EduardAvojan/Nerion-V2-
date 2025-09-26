"""
Environment for training an agent to perform cross-file refactoring tasks.
"""
from __future__ import annotations

import subprocess
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import time

from selfcoder.vcs import git_ops

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

class RefactorEnvironment:
    """
    An environment that can safely apply a refactoring action, run tests,
    and report the outcome.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def step(self, action: RenameAction) -> TestOutcome:
        """
        Applies a rename action, runs tests, and reverts the changes.

        Returns:
            A TestOutcome object detailing the results of the test run.
        """
        snapshot_id = None
        original_safe_mode = os.getenv("SELFCODER_SAFE_MODE", "1")
        try:
            # 1. Create a snapshot of the current state (forcing real mode)
            print("Creating project snapshot...")
            os.environ["SELFCODER_SAFE_MODE"] = "0"
            snapshot_id = git_ops.snapshot(
                f"pre-refactor-{action.old_name}-{action.new_name}"
            )
            os.environ["SELFCODER_SAFE_MODE"] = original_safe_mode # Restore

            if not snapshot_id or not isinstance(snapshot_id, str):
                raise RuntimeError(f"Failed to create a valid snapshot ID: {snapshot_id}")

            # 2. Apply the rename action using the existing CLI
            print(f"Applying rename: {action.old_name} -> {action.new_name}")
            self._apply_rename(action)

            # 3. Run the project's test suite
            print("Running test suite...")
            outcome = self._run_tests()
            print(f"Outcome: {outcome.passed} passed, {outcome.failed} failed, {outcome.errored} errored.")

            return outcome

        finally:
            # 4. Always restore the snapshot to ensure a clean state
            os.environ["SELFCODER_SAFE_MODE"] = original_safe_mode # Restore before final op
            if snapshot_id:
                print(f"Restoring snapshot {snapshot_id}...")
                git_ops.restore_snapshot(snapshot_ts=snapshot_id, verbose=False)
                print("Restore complete.")

    def _apply_rename(self, action: RenameAction):
        """
        Calls the `nerion rename` CLI command as a subprocess.
        """
        command = [
            "python",
            "-m",
            "selfcoder.cli",
            "rename",
            "--old",
            action.old_name,
            "--new",
            action.new_name,
            "--apply", # Ensure the change is actually applied
            str(self.project_root / action.file_path),
        ]
        try:
            subprocess.run(
                command, 
                cwd=self.project_root, 
                check=True, 
                capture_output=True, 
                text=True
            )
        except subprocess.CalledProcessError as e:
            print("--- RENAME COMMAND FAILED ---")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            print("---------------------------")
            raise

    def _parse_pytest_output(self, output: str) -> TestOutcome:
        """Parses the summary line from pytest output to extract test counts."""
        passed = 0
        failed = 0
        errored = 0
        
        # Robustly find the summary line, which contains keywords and a time duration.
        match = re.search(r"^(.*(passed|failed|errored|skipped|warnings).* in \d+\.\d+s.*)$", output, re.MULTILINE)

        if match:
            summary_text = match.group(1)
            
            passed_match = re.search(r"(\d+) passed", summary_text)
            if passed_match:
                passed = int(passed_match.group(1))
                
            failed_match = re.search(r"(\d+) failed", summary_text)
            if failed_match:
                failed = int(failed_match.group(1))
                
            errored_match = re.search(r"(\d+) errored", summary_text)
            if errored_match:
                errored = int(errored_match.group(1))

        return TestOutcome(passed=passed, failed=failed, errored=errored)

    def _run_tests(self) -> TestOutcome:
        """
        Runs the pytest suite in a clean, isolated subprocess.
        """
        # Create a clean environment configuration
        env = os.environ.copy()
        # Force Python not to write .pyc files, preventing caching issues
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        # Optional: Clean up PYTHONPATH if it might be contaminated
        if 'PYTHONPATH' in env:
            del env['PYTHONPATH']

        command = [
            sys.executable, "-m", "pytest",
            "-p", "no:cacheprovider",
        ]

        proc = subprocess.run(
            command, 
            cwd=self.project_root, 
            env=env,
            capture_output=True, 
            text=True
        )
        
        # Pass both stdout and stderr to the parser, as errors can be in either
        output_text = proc.stdout + proc.stderr
        return self._parse_pytest_output(output_text)

def main():
    """
    Demonstrates a single step of the environment.
    """
    print("Initializing refactor environment...")
    env = RefactorEnvironment(".")

    # Define a sample rename action
    action = RenameAction(
        file_path="selfcoder/planner/planner.py",
        old_name="plan_edits_from_nl",
        new_name="plan_edits_from_natural_language",
    )

    print(f"Performing dry run of action: {action}")
    outcome = env.step(action)

    print(f"\n--- Environment Step Complete ---")
    print(f"Action: rename '{action.old_name}' to '{action.new_name}'")
    print(f"Outcome: {outcome}")
    print(f"Success: {outcome.was_successful}")
    print("---------------------------------")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    main()
