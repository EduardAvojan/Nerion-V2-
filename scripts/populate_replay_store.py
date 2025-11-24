#!/usr/bin/env python3
"""Populate ReplayStore with test execution results from curriculum lessons.

This creates the 'immune system memory' by:
1. Executing tests for all curriculum lessons
2. Computing surprise (initially random, later from model predictions)
3. Storing experiences with pass/fail outcomes in ReplayStore
"""

import sqlite3
import subprocess
import tempfile
import sys
from pathlib import Path
from dataclasses import asdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.telemetry import TelemetryLogger
from nerion_digital_physicist.infrastructure.outcomes import log_outcome

def run_test_in_sandbox(source_code: str, test_code: str, timeout: int = 5) -> dict:
    """Execute test code with source in isolated sandbox.

    Returns:
        dict with keys: passed (bool), returncode (int), stdout (str), stderr (str)
    """
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Write source code
        module_path = tmppath / "module.py"
        module_path.write_text(source_code, encoding='utf-8')

        # Write test code
        test_path = tmppath / "test_module.py"
        test_path.write_text(test_code, encoding='utf-8')

        # Run tests with pytest
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v"],
                cwd=str(tmppath),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            passed = result.returncode == 0

            return {
                'passed': passed,
                'returncode': result.returncode,
                'stdout': result.stdout[:500],  # Truncate
                'stderr': result.stderr[:500],
            }

        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test execution timed out after {timeout}s',
            }
        except Exception as e:
            return {
                'passed': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)[:500],
            }

def compute_surprise(before_result: dict, after_result: dict) -> tuple[float, float]:
    """Compute surprise for before and after code execution.

    For now, this is a simple heuristic:
    - Low surprise if outcome matches expectation (before fails, after passes)
    - High surprise if outcome is unexpected (before passes, after fails)

    Later, this will use the GNN model's prediction error.

    Returns:
        (before_surprise, after_surprise)
    """
    # Expected: before fails, after passes
    before_passed = before_result['passed']
    after_passed = after_result['passed']

    # Baseline surprise (random with bias)
    base_surprise = random.uniform(0.3, 0.7)

    # Before code surprise
    if before_passed:
        # Unexpected: before code actually passed!
        before_surprise = base_surprise + random.uniform(0.3, 0.6)
    else:
        # Expected: before code failed
        before_surprise = base_surprise + random.uniform(-0.2, 0.2)

    # After code surprise
    if not after_passed:
        # Unexpected: after code failed!
        after_surprise = base_surprise + random.uniform(0.3, 0.6)
    else:
        # Expected: after code passed
        after_surprise = base_surprise + random.uniform(-0.2, 0.2)

    # Clamp to [0, 2]
    before_surprise = max(0.0, min(2.0, before_surprise))
    after_surprise = max(0.0, min(2.0, after_surprise))

    return before_surprise, after_surprise

def main():
    # Configuration
    DB_PATH = Path("out/learning/curriculum.sqlite")
    REPLAY_ROOT = Path("out/training_runs/replay_immune_system")

    # Initialize systems
    replay = ReplayStore(REPLAY_ROOT)
    telemetry = TelemetryLogger(REPLAY_ROOT)

    # Load lessons
    print("Loading curriculum lessons...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name, before_code, after_code, test_code FROM lessons WHERE test_code IS NOT NULL AND test_code != ''")
    lessons = cursor.fetchall()
    conn.close()

    print(f"Found {len(lessons)} lessons with tests")
    print("=" * 80)

    # Track statistics
    before_passed = 0
    after_passed = 0
    both_passed = 0
    both_failed = 0

    # Process each lesson
    for idx, (name, before_code, after_code, test_code) in enumerate(lessons):
        if (idx + 1) % 50 == 0:
            print(f"\nProgress: {idx + 1}/{len(lessons)}")
            print(f"  Before passed: {before_passed}/{idx+1} ({100*before_passed/(idx+1):.1f}%)")
            print(f"  After passed: {after_passed}/{idx+1} ({100*after_passed/(idx+1):.1f}%)")

        # Execute tests
        before_result = run_test_in_sandbox(before_code, test_code)
        after_result = run_test_in_sandbox(after_code, test_code)

        # Update stats
        if before_result['passed']:
            before_passed += 1
        if after_result['passed']:
            after_passed += 1
        if before_result['passed'] and after_result['passed']:
            both_passed += 1
        if not before_result['passed'] and not after_result['passed']:
            both_failed += 1

        # Compute surprise
        before_surprise, after_surprise = compute_surprise(before_result, after_result)

        # Store BEFORE experience
        before_exp = replay.append(
            task_id=f"{name}_before",
            template_id="curriculum_lesson",
            status="solved" if before_result['passed'] else "failed",
            surprise=before_surprise,
            metadata={
                "lesson_name": name,
                "code_version": "before",
                "source_code": before_code,
                "test_code": test_code,
                "test_result": before_result,
            }
        )

        log_outcome(
            replay=replay,
            telemetry=telemetry,
            experience_id=before_exp.experience_id,
            status=before_exp.status,
            surprise=before_surprise,
        )

        # Store AFTER experience
        after_exp = replay.append(
            task_id=f"{name}_after",
            template_id="curriculum_lesson",
            status="solved" if after_result['passed'] else "failed",
            surprise=after_surprise,
            metadata={
                "lesson_name": name,
                "code_version": "after",
                "source_code": after_code,
                "test_code": test_code,
                "test_result": after_result,
            }
        )

        log_outcome(
            replay=replay,
            telemetry=telemetry,
            experience_id=after_exp.experience_id,
            status=after_exp.status,
            surprise=after_surprise,
        )

    print("\n" + "=" * 80)
    print("EXPERIENCE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total lessons processed: {len(lessons)}")
    print(f"Total experiences stored: {len(lessons) * 2}")
    print()
    print(f"Before code results:")
    print(f"  Passed: {before_passed}/{len(lessons)} ({100*before_passed/len(lessons):.1f}%)")
    print(f"  Failed: {len(lessons)-before_passed}/{len(lessons)} ({100*(len(lessons)-before_passed)/len(lessons):.1f}%)")
    print()
    print(f"After code results:")
    print(f"  Passed: {after_passed}/{len(lessons)} ({100*after_passed/len(lessons):.1f}%)")
    print(f"  Failed: {len(lessons)-after_passed}/{len(lessons)} ({100*(len(lessons)-after_passed)/len(lessons):.1f}%)")
    print()
    print(f"Both passed: {both_passed}/{len(lessons)} ({100*both_passed/len(lessons):.1f}%)")
    print(f"Both failed: {both_failed}/{len(lessons)} ({100*both_failed/len(lessons):.1f}%)")
    print()
    print(f"ReplayStore location: {REPLAY_ROOT / 'replay.jsonl'}")
    print(f"Telemetry location: {REPLAY_ROOT / 'telemetry.jsonl'}")

if __name__ == "__main__":
    main()
