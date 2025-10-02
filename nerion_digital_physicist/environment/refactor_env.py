from __future__ import annotations

import subprocess
import os
import re
import sys
# from dataclasses import dataclass # Removed unused import
from pathlib import Path

# ProjectParser is no longer needed for Plan D
# from nerion_digital_physicist.agent.project_graph import ProjectParser
from selfcoder.vcs import git_ops
from .types import RenameAction, TestOutcome


class RefactorEnvironment:
    """
    An environment that can safely apply a refactoring action, run tests,
    and report the outcome.
    """

    # Updated __init__ to remove ProjectParser dependency
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        # self.parser is removed
        print("RefactorEnvironment initialized. Using Plan D: Heuristic-Based Localized Test Runner.")

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

            # 3. Run the localized test suite
            print("Running localized test suite (Plan D)...")
            affected_tests = self._get_affected_tests(action.file_path)
            outcome = self._run_tests(affected_tests)
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
        # Updated regex to handle optional decimals in time (e.g., 5s or 5.12s)
        match = re.search(r"^(.*(passed|failed|errored|skipped|warnings).* in \d+(\.\d+)?s.*)$", output, re.MULTILINE)

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

    def _get_affected_tests(self, modified_file: str) -> set[str]:
        """
        Implements Plan D: Finds tests in the same directory and subdirectories as the modified file.
        Replaces the dependency analysis logic.
        """
        
        # 1. Resolve the absolute path of the modified file
        try:
            # Convert input to Path object
            input_path = Path(modified_file)
            
            if input_path.is_absolute():
                resolved_path = input_path.resolve()
            else:
                # Assume relative to project root
                resolved_path = (self.project_root / input_path).resolve()

        except Exception as e:
            print(f"Error resolving path for {modified_file}: {e}")
            return set()

        # 2. Security Validation: Ensure the path is within the project boundaries
        try:
            # If the path is not relative to the project root (e.g., "../../etc/passwd"), 
            # this raises a ValueError.
            relative_path_str = str(resolved_path.relative_to(self.project_root))
        except ValueError:
            print(f"Security Violation: Modified file path is outside project root: {resolved_path}")
            return set()

        # 3. Existence Check
        if not resolved_path.exists():
            # If the file doesn't exist after the action, we can't determine localized tests reliably.
            print(f"Warning: Modified file not found: {resolved_path}. Skipping tests.")
            return set()

        # 4. Determine the search base directory
        # If it's a directory, search within it; otherwise, use the parent directory of the file.
        if resolved_path.is_dir():
            search_base_dir = resolved_path
        else:
            search_base_dir = resolved_path.parent
        
        affected_tests = set()

        # 5. Use Recursive Glob (rglob) to find tests in the directory and subdirectories

        # Find test_*.py files
        for test_file in search_base_dir.rglob('test_*.py'):
            if test_file.is_file() and ".venv" not in str(test_file) and "backups" not in str(test_file):
                # Store as string paths, as the test runner expects strings
                affected_tests.add(str(test_file))

        # Find *_test.py files
        for test_file in search_base_dir.rglob('*_test.py'):
            if test_file.is_file() and ".venv" not in str(test_file) and "backups" not in str(test_file):
                affected_tests.add(str(test_file))

        # 6. Ensure the modified file itself is included if it is a test
        if resolved_path.is_file():
            basename = resolved_path.name
            if basename.startswith("test_") or basename.endswith("_test.py"):
                affected_tests.add(str(resolved_path))

        # 7. Logging and return
        if affected_tests:
            print(f"Plan D TIA: Found {len(affected_tests)} localized tests for {relative_path_str}.")
            return affected_tests
        else:
            # CRITICAL: Run nothing if no localized tests are found.
            print(f"Plan D TIA: No localized tests found for {relative_path_str}. Skipping tests.")
            return set()

    def _run_tests(self, test_files: set[str] | None = None) -> TestOutcome:
        """
        Runs the pytest suite in a clean, isolated subprocess.
        """
        
        # CRITICAL: If test_files is None or empty, we must not run pytest.
        # Running pytest without specific targets causes it to run the entire suite.
        if not test_files:
            print("No tests specified. Skipping pytest execution.")
            return TestOutcome(passed=0, failed=0, errored=0)

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

        # Add the specific test files
        command.extend(list(test_files))

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
