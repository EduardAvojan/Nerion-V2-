#!/usr/bin/env python3
"""
Quality Review and Merge Script for Agent-Generated Lessons

This script:
1. Reviews lessons from agent_generated_curriculum.sqlite
2. Tests each lesson for quality (10/10 standard)
3. Allows manual approval/rejection
4. Merges approved lessons into curriculum.sqlite
5. Generates quality report

Usage:
    python scripts/review_and_merge_lessons.py --review     # Review lessons
    python scripts/review_and_merge_lessons.py --merge      # Merge approved lessons
    python scripts/review_and_merge_lessons.py --stats      # Show statistics
"""
import sys
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
import json
import hashlib

# Database paths
AGENT_DB = Path("out/learning/agent_generated_curriculum.sqlite")
PRODUCTION_DB = Path("out/learning/curriculum.sqlite")
REVIEW_LOG = Path("out/learning/lesson_review_log.json")

# Baseline: All lessons with ID <= 973 are from production (973 lessons, sequential IDs)
# Only review and merge lessons with ID > 973 (new agent-generated lessons start at 974)
PRODUCTION_BASELINE_ID = 973

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class LessonValidator:
    """Validates lesson quality against 10/10 standard (BULLETPROOF version)."""

    def validate(self, before_code: str, after_code: str, test_code: str,
                 name: str = "", description: str = "") -> tuple[bool, list[str]]:
        """
        Returns: (passed: bool, issues: List[str])

        Checks both TECHNICAL validity and SUBJECTIVE quality (10/10 standard).
        """
        issues = []

        # === TECHNICAL VALIDATION ===

        # 1. Syntactic validity
        try:
            compile(before_code, '<before>', 'exec')
        except SyntaxError as e:
            issues.append(f"❌ before_code syntax error: {e}")

        try:
            compile(after_code, '<after>', 'exec')
        except SyntaxError as e:
            issues.append(f"❌ after_code syntax error: {e}")

        try:
            compile(test_code, '<test>', 'exec')
        except SyntaxError as e:
            issues.append(f"❌ test_code syntax error: {e}")

        if issues:
            return False, issues

        # 2. Test framework check
        if 'unittest' not in test_code and 'TestCase' not in test_code:
            issues.append("❌ Test code must use unittest framework")

        # 3. Minimum test count
        test_methods = test_code.count('def test_')
        if test_methods < 2:
            issues.append(f"❌ Must have at least 2 tests (found {test_methods})")

        # 4. Run tests on before_code (should fail)
        before_passed, before_output = self._run_tests(before_code, test_code)
        if before_passed:
            issues.append("❌ Tests PASSED on before_code (should FAIL to demonstrate bug)")

        # 5. Run tests on after_code (should pass)
        after_passed, after_output = self._run_tests(after_code, test_code)
        if not after_passed:
            issues.append(f"❌ Tests FAILED on after_code (should PASS):\n{after_output}")

        # === SUBJECTIVE QUALITY VALIDATION (10/10 Standard) ===

        # 6. Code similarity check (before and after should be similar, not completely different)
        similarity = self._check_code_similarity(before_code, after_code)
        if similarity < 0.3:  # Less than 30% similar
            issues.append(f"⚠️ WARNING: before_code and after_code are very different (similarity: {similarity:.1%}). "
                         "They should be the same code with a single fix, not completely different implementations.")

        # 7. Check for multiple bugs (count TODO, FIXME, BUG comments or multiple issues)
        bug_markers = before_code.count('# Bug:') + before_code.count('# TODO') + before_code.count('# FIXME')
        if bug_markers > 1:
            issues.append(f"⚠️ WARNING: Found {bug_markers} bug markers. Lesson should have ONE clear bug, not multiple.")

        # 8. Check for toy/trivial code (very short code might be toy example)
        before_lines = len([l for l in before_code.split('\n') if l.strip() and not l.strip().startswith('#')])
        if before_lines < 5:
            issues.append(f"⚠️ WARNING: Code is very short ({before_lines} lines). May be toy example, not real-world pattern.")

        # 9. Check test quality (tests should actually test something, not just run code)
        if 'self.assert' not in test_code:
            issues.append("❌ Tests have no assertions! Tests must verify behavior with self.assert* methods.")

        # Count assertions
        assertion_count = test_code.count('self.assert')
        if assertion_count < 2:
            issues.append(f"⚠️ WARNING: Only {assertion_count} assertion(s). Good tests should have multiple assertions.")

        # 10. Check for realistic code patterns (imports, functions, classes)
        has_imports = 'import ' in before_code or 'from ' in before_code
        has_function = 'def ' in before_code
        has_class = 'class ' in before_code

        if not (has_imports or has_function or has_class):
            issues.append("⚠️ WARNING: Code has no imports, functions, or classes. May be too simplistic.")

        # 11. CERF level check (basic pattern matching)
        if name:
            level = name[:2].lower()  # a1, a2, b1, b2, c1, c2
            complexity_indicators = {
                'a1': ['def ', 'print', 'if ', 'for '],  # Basic
                'a2': ['dict', 'list', 'try:', 'except'],  # Elementary
                'b1': ['class ', '@', 'yield', 'with '],  # Intermediate
                'b2': ['async ', 'await', 'metaclass', 'threading'],  # Upper-intermediate
                'c1': ['weakref', 'gc.', 'multiprocessing', '@lru_cache'],  # Professional
                'c2': ['threading.Lock', 'Queue', '__slots__', 'mmap']  # Mastery
            }

            if level in complexity_indicators:
                expected_patterns = complexity_indicators[level]
                found_patterns = [p for p in expected_patterns if p in before_code.lower()]

                if not found_patterns and level in ['b2', 'c1', 'c2']:
                    issues.append(f"⚠️ WARNING: {level.upper()} lesson but no advanced patterns found. "
                                 f"Expected: {', '.join(expected_patterns)}")

        return len(issues) == 0, issues

    def _check_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate rough similarity between two code snippets (0.0 to 1.0)."""
        # Simple approach: count common lines
        lines1 = set(l.strip() for l in code1.split('\n') if l.strip() and not l.strip().startswith('#'))
        lines2 = set(l.strip() for l in code2.split('\n') if l.strip() and not l.strip().startswith('#'))

        if not lines1 or not lines2:
            return 0.0

        common = len(lines1 & lines2)
        total = len(lines1 | lines2)

        return common / total if total > 0 else 0.0

    def _run_tests(self, code: str, test_code: str) -> tuple[bool, str]:
        """Run tests and return (passed, output)."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Inject code and tests
            test_script = f"""
{code}

# Test code
import unittest
import sys

BEFORE_CODE = '''{code}'''
AFTER_CODE = '''{code}'''

{test_code}

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
"""
            f.write(test_script)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(temp_path).unlink()  # Clean up

            # Check if all tests passed
            passed = result.returncode == 0 and 'FAILED' not in result.stderr
            return passed, result.stderr + result.stdout
        except subprocess.TimeoutExpired:
            Path(temp_path).unlink()
            return False, "Test execution timeout (>10s)"
        except Exception as e:
            Path(temp_path).unlink()
            return False, f"Test execution error: {e}"


def show_statistics():
    """Show statistics about both databases."""
    print(f"\n{BLUE}=== DATABASE STATISTICS ==={RESET}\n")

    # Production DB
    if PRODUCTION_DB.exists():
        conn = sqlite3.connect(PRODUCTION_DB)
        total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        print(f"📚 Production DB: {GREEN}{total} lessons{RESET}")

        # By CERF level
        levels = conn.execute("""
            SELECT
                SUBSTR(name, 1, 2) as level,
                COUNT(*) as count
            FROM lessons
            WHERE name LIKE 'a1_%' OR name LIKE 'a2_%'
                OR name LIKE 'b1_%' OR name LIKE 'b2_%'
                OR name LIKE 'c1_%' OR name LIKE 'c2_%'
            GROUP BY level
            ORDER BY level
        """).fetchall()
        for level, count in levels:
            print(f"   {level.upper()}: {count}")
        conn.close()
    else:
        print(f"📚 Production DB: {RED}Not found{RESET}")

    print()

    # Agent DB
    if AGENT_DB.exists():
        conn = sqlite3.connect(AGENT_DB)
        total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        print(f"🤖 Agent-Generated DB: {YELLOW}{total} lessons (pending review){RESET}")

        # By CERF level
        levels = conn.execute("""
            SELECT
                SUBSTR(name, 1, 2) as level,
                COUNT(*) as count
            FROM lessons
            WHERE name LIKE 'a1_%' OR name LIKE 'a2_%'
                OR name LIKE 'b1_%' OR name LIKE 'b2_%'
                OR name LIKE 'c1_%' OR name LIKE 'c2_%'
            GROUP BY level
            ORDER BY level
        """).fetchall()
        for level, count in levels:
            print(f"   {level.upper()}: {count}")
        conn.close()
    else:
        print(f"🤖 Agent-Generated DB: {YELLOW}Not found (no lessons yet){RESET}")

    print()


def review_lessons():
    """Review only NEW lessons in agent-generated database (id > PRODUCTION_BASELINE_ID)."""
    if not AGENT_DB.exists():
        print(f"{RED}❌ No agent-generated database found at {AGENT_DB}{RESET}")
        return

    conn = sqlite3.connect(AGENT_DB)

    # Only review NEW lessons (id > 973), not the 973 production lessons
    cursor = conn.execute("""
        SELECT id, name, description, before_code, after_code, test_code, focus_area
        FROM lessons
        WHERE id > ?
        ORDER BY id
    """, (PRODUCTION_BASELINE_ID,))

    lessons = cursor.fetchall()
    conn.close()

    if not lessons:
        print(f"{YELLOW}No lessons to review{RESET}")
        return

    print(f"\n{BLUE}=== REVIEWING {len(lessons)} LESSONS ==={RESET}\n")

    validator = LessonValidator()
    results = {
        "timestamp": datetime.now().isoformat(),
        "total": len(lessons),
        "passed": 0,
        "failed": 0,
        "lessons": []
    }

    for lesson_id, name, description, before_code, after_code, test_code, focus_area in lessons:
        print(f"\n{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
        print(f"Lesson: {YELLOW}{name}{RESET}")
        print(f"Description: {description}")
        print(f"Focus Area: {focus_area}")
        print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

        # Validate (with name and description for subjective quality checks)
        passed, issues = validator.validate(before_code, after_code, test_code, name, description)

        if passed:
            print(f"{GREEN}✅ PASSED - Quality 10/10{RESET}")
            results["passed"] += 1
            status = "approved"
        else:
            print(f"{RED}❌ FAILED - Issues found:{RESET}")
            for issue in issues:
                print(f"   {issue}")
            results["failed"] += 1
            status = "rejected"

        results["lessons"].append({
            "id": lesson_id,
            "name": name,
            "status": status,
            "issues": issues if not passed else []
        })

    # Save review log
    REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_LOG, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{BLUE}=== REVIEW SUMMARY ==={RESET}")
    print(f"Total: {len(lessons)}")
    print(f"{GREEN}Approved: {results['passed']}{RESET}")
    print(f"{RED}Rejected: {results['failed']}{RESET}")
    print(f"\nReview log saved to: {REVIEW_LOG}")

    if results["passed"] > 0:
        print(f"\n{YELLOW}To merge approved lessons into production:{RESET}")
        print(f"   python scripts/review_and_merge_lessons.py --merge")


def merge_approved_lessons():
    """Merge approved lessons from agent DB to production DB."""
    if not REVIEW_LOG.exists():
        print(f"{RED}❌ No review log found. Run --review first.{RESET}")
        return

    with open(REVIEW_LOG, 'r') as f:
        review_data = json.load(f)

    approved = [l for l in review_data["lessons"] if l["status"] == "approved"]

    if not approved:
        print(f"{YELLOW}No approved lessons to merge{RESET}")
        return

    print(f"\n{BLUE}=== MERGING {len(approved)} APPROVED LESSONS ==={RESET}\n")

    # Use SafeCurriculumDB for production writes
    from nerion_digital_physicist.db.safe_curriculum import SafeCurriculumDB

    agent_conn = sqlite3.connect(AGENT_DB)
    merged_count = 0

    with SafeCurriculumDB(db_path=PRODUCTION_DB) as prod_db:
        for lesson_info in approved:
            lesson_id = lesson_info["id"]

            # Safety check: Only merge NEW lessons (id > PRODUCTION_BASELINE_ID)
            if lesson_id <= PRODUCTION_BASELINE_ID:
                print(f"{RED}❌ Skipped: Lesson ID {lesson_id} is from production baseline{RESET}")
                continue

            # Fetch full lesson from agent DB
            cursor = agent_conn.execute("""
                SELECT name, description, focus_area, before_code, after_code,
                       test_code, category, language, metadata
                FROM lessons WHERE id = ?
            """, (lesson_id,))

            row = cursor.fetchone()
            if not row:
                continue

            name, description, focus_area, before_code, after_code, test_code, category, language, metadata = row

            try:
                # Add to production (SafeCurriculumDB handles duplicates)
                prod_db.add_lesson(
                    name=name,
                    description=description,
                    before_code=before_code,
                    after_code=after_code,
                    test_code=test_code,
                    focus_area=focus_area,
                    category=category,
                    language=language or "python",
                    metadata=metadata
                )
                print(f"{GREEN}✅ Merged: {name}{RESET}")
                merged_count += 1
            except Exception as e:
                print(f"{RED}❌ Failed to merge {name}: {e}{RESET}")

    agent_conn.close()

    print(f"\n{GREEN}Successfully merged {merged_count}/{len(approved)} lessons{RESET}")
    print(f"\n{YELLOW}To clear agent-generated database:{RESET}")
    print(f"   rm {AGENT_DB}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Review and merge agent-generated lessons")
    parser.add_argument('--review', action='store_true', help='Review lessons for quality')
    parser.add_argument('--merge', action='store_true', help='Merge approved lessons into production')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')

    args = parser.parse_args()

    if args.stats or (not args.review and not args.merge):
        show_statistics()

    if args.review:
        review_lessons()

    if args.merge:
        merge_approved_lessons()


if __name__ == "__main__":
    main()
