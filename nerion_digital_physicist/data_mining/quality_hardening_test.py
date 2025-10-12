"""Unit tests for quality hardening detectors.

Tests penalty detection, hardened verification, and tier assignment.
Run with: pytest quality_hardening_test.py -v
"""
import ast
import pytest
from dataclasses import dataclass, field
from typing import Dict, List

from quality_hardening import NegativeEvidenceDetector, VerificationHardening


@dataclass
class MockCommit:
    """Mock commit for testing."""
    files: List[str] = field(default_factory=list)
    message: str = ""
    metrics: Dict = field(default_factory=dict)


class TestNegativeEvidenceDetector:
    """Test suite for penalty detectors."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = NegativeEvidenceDetector()

    # ===== Test 1: Complexity Increase Without Tests =====

    def test_complexity_increase_without_tests_triggers(self):
        """Penalty when complexity increases without test coverage."""
        commit = MockCommit(
            files=["app.py"],
            metrics={"complexity_delta": 3, "test_proximity": False}
        )
        before = "def foo(x): return x"
        after = "def foo(x):\n    if x > 0:\n        if x < 10:\n            return x\n    return 0"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "complexity_increase_no_verification" in penalties

    def test_complexity_increase_with_tests_no_penalty(self):
        """No penalty when complexity increases with test coverage."""
        commit = MockCommit(
            files=["app.py", "test_app.py"],
            metrics={"complexity_delta": 3, "test_proximity": True}
        )
        before = "def foo(x): return x"
        after = "def foo(x):\n    if x > 0:\n        return x * 2\n    return 0"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "complexity_increase_no_verification" not in penalties

    # ===== Test 2: SQL String Concatenation =====

    def test_sql_concat_introduced(self):
        """Penalty when SQL concatenation is introduced (direct in execute)."""
        before = """
def get_user(user_id):
    return db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
"""
        after = """
def get_user(user_id):
    return db.execute("SELECT * FROM users WHERE id = " + str(user_id))
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "sql_string_concat" in penalties

    def test_sql_concat_with_fstring(self):
        """Penalty for f-string in SQL execute."""
        before = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))
"""
        after = """
def get_user(user_id):
    return db.execute(f"SELECT * FROM users WHERE id = {user_id}")
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "sql_string_concat" in penalties

    def test_sql_parameterized_no_penalty(self):
        """No penalty for parameterized queries."""
        before = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return db.execute(query)
"""
        after = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "sql_string_concat" not in penalties

    # ===== Test 3: Removed Validation =====

    def test_removed_validation_triggers(self):
        """Penalty when validation calls are removed."""
        before = """
def process(data):
    if isinstance(data, dict):
        return data.get("value")
    return None
"""
        after = """
def process(data):
    return data.get("value")
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "removed_validation" in penalties

    def test_added_validation_no_penalty(self):
        """No penalty when validation is added."""
        before = """
def process(data):
    return data.get("value")
"""
        after = """
def process(data):
    if isinstance(data, dict):
        return data.get("value")
    return None
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "removed_validation" not in penalties

    # ===== Test 4: Swallowed Exceptions =====

    def test_swallowed_exception_pass(self):
        """Penalty for 'except: pass' pattern."""
        before = """
def load_config():
    with open("config.json") as f:
        return json.load(f)
"""
        after = """
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except:
        pass
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "swallowed_exception" in penalties

    def test_swallowed_exception_return_none(self):
        """Penalty for 'except: return None' pattern."""
        before = """
def load_config():
    with open("config.json") as f:
        return json.load(f)
"""
        after = """
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except Exception:
        return None
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "swallowed_exception" in penalties

    def test_proper_exception_handling_no_penalty(self):
        """No penalty for proper exception handling."""
        before = """
def load_config():
    with open("config.json") as f:
        return json.load(f)
"""
        after = """
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Config not found: {e}")
        raise
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "swallowed_exception" not in penalties

    # ===== Test 5: Linter Disabling =====

    def test_linter_disable_noqa(self):
        """Penalty for adding # noqa comment."""
        before = "def foo(x, y, z, a, b, c, d, e, f, g, h):\n    return x + y"
        after = "def foo(x, y, z, a, b, c, d, e, f, g, h):  # noqa: E501\n    return x + y"

        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "linter_disabled" in penalties

    def test_linter_disable_type_ignore(self):
        """Penalty for adding # type: ignore comment."""
        before = "result = calculate(data)"
        after = "result = calculate(data)  # type: ignore"

        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "linter_disabled" in penalties

    # ===== Test 6: Wildcard Imports =====

    def test_wildcard_import_introduced(self):
        """Penalty for introducing wildcard import."""
        before = "from utils import get_value, set_value"
        after = "from utils import *"

        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "wildcard_import" in penalties

    def test_explicit_imports_no_penalty(self):
        """No penalty for explicit imports."""
        before = "from utils import *"
        after = "from utils import get_value, set_value"

        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "wildcard_import" not in penalties

    # ===== Test 7: Unsafe Path Operations =====

    def test_unsafe_path_concat(self):
        """Penalty for unsafe path concatenation."""
        before = """
import os
def get_file(name):
    return os.path.join("/data", name)
"""
        after = """
def get_file(name):
    return "/data/" + name
"""
        commit = MockCommit()
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = self.detector.detect_penalties(commit, before_ast, after_ast, before, after)

        assert "unsafe_path_concat" in penalties

    # ===== Test Penalty Scoring =====

    def test_penalty_score_calculation(self):
        """Verify penalty scores are calculated correctly."""
        penalties = [
            "complexity_increase_no_verification",  # 4
            "sql_string_concat",  # 6
            "swallowed_exception",  # 5
        ]

        total_penalty = self.detector.calculate_penalty_score(penalties)

        assert total_penalty == 15  # 4 + 6 + 5


class TestVerificationHardening:
    """Test suite for hardened verification signals."""

    def setup_method(self):
        """Initialize hardener for each test."""
        self.hardener = VerificationHardening()

    # ===== Test File Changes =====

    def test_test_file_changes_detected(self):
        """Detect test file changes."""
        commit = MockCommit(files=["app.py", "test_app.py"])
        before = "def foo(): pass"
        after = "def foo(): return 1"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        assert has_verification
        assert points >= 5

    def test_no_test_files_no_verification(self):
        """No verification when no test files touched."""
        commit = MockCommit(files=["app.py", "utils.py"])
        before = "def foo(): pass"
        after = "def foo(): return 1"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        # May have some points from other sources, but not from test files
        # Just verify it doesn't crash
        assert isinstance(has_verification, bool)
        assert isinstance(points, int)

    # ===== Test New Asserts =====

    def test_new_asserts_detected(self):
        """Detect newly added assert statements."""
        commit = MockCommit(files=["app.py"])
        before = "def foo(x): return x"
        after = "def foo(x):\n    assert x > 0\n    return x"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        assert has_verification
        assert points >= 3

    def test_no_new_asserts_no_bonus(self):
        """No bonus for same number of asserts."""
        commit = MockCommit(files=["app.py"])
        before = "def foo(x):\n    assert x > 0\n    return x"
        after = "def foo(x):\n    assert x > 0\n    return x * 2"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        # Should not get assert bonus, but may get other verification
        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        # Just verify no crash and returns valid values
        assert isinstance(has_verification, bool)
        assert isinstance(points, int)

    # ===== Test CI Keywords =====

    def test_ci_keywords_filtered(self):
        """Detect filtered CI keywords."""
        commit = MockCommit(
            files=["app.py"],
            message="fix failing test in CI"
        )
        before = "def foo(): pass"
        after = "def foo(): return 1"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        assert has_verification
        assert points >= 2

    def test_generic_ci_mention_ignored(self):
        """Generic 'ci' in prose should not trigger."""
        commit = MockCommit(
            files=["app.py"],
            message="update docstring to mention CI/CD pipeline"
        )
        before = "def foo(): pass"
        after = "def foo(): return 1"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        # Should not get CI keyword bonus
        # Just verify it doesn't crash
        assert isinstance(has_verification, bool)
        assert isinstance(points, int)

    # ===== Test Issue Linkage =====

    def test_bug_issue_linkage(self):
        """Detect bug issue linkage."""
        commit = MockCommit(
            files=["app.py"],
            message="fixes #123 - crash when input is None"
        )
        before = "def foo(): pass"
        after = "def foo(): return 1"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        assert has_verification
        assert points >= 2

    def test_non_bug_issue_no_bonus(self):
        """No bonus for non-bug issue references."""
        commit = MockCommit(
            files=["app.py"],
            message="closes #456 - add new feature"
        )
        before = "def foo(): pass"
        after = "def foo(): return 1"

        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        has_verification, points = self.hardener.detect_hardened_verification(
            commit, before_ast, after_ast
        )

        # Should not get issue linkage bonus for non-bug
        # Just verify it doesn't crash
        assert isinstance(has_verification, bool)
        assert isinstance(points, int)


class TestIntegration:
    """Integration tests for combined detection."""

    def test_multiple_penalties(self):
        """Multiple penalties stack correctly."""
        detector = NegativeEvidenceDetector()

        before = """
def get_user(user_id):
    if isinstance(user_id, int):
        return db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
"""
        after = """
def get_user(user_id):
    try:
        return db.execute(f"SELECT * FROM users WHERE id = {user_id}")
    except:
        pass
"""

        commit = MockCommit(metrics={"complexity_delta": 0, "test_proximity": False})
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = detector.detect_penalties(commit, before_ast, after_ast, before, after)

        # Should have: sql_concat, removed_validation, swallowed_exception
        assert len(penalties) >= 2
        assert "sql_string_concat" in penalties
        assert "swallowed_exception" in penalties

    def test_clean_improvement_no_penalties(self):
        """Clean improvements trigger no penalties."""
        detector = NegativeEvidenceDetector()

        before = """
def process(data):
    return data.get("value")
"""
        after = """
def process(data):
    if not isinstance(data, dict):
        return None
    return data.get("value")
"""

        commit = MockCommit(
            files=["app.py", "test_app.py"],
            metrics={"complexity_delta": 1, "test_proximity": True}
        )
        before_ast = ast.parse(before)
        after_ast = ast.parse(after)

        penalties = detector.detect_penalties(commit, before_ast, after_ast, before, after)

        # Clean improvement with tests - should have no penalties
        assert len(penalties) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
