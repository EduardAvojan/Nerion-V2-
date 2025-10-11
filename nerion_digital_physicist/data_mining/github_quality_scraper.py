"""Production-grade GitHub commit scraper with multi-stage quality filters.

This scraper fetches Python bug fixes from GitHub and applies rigorous quality
filtering to ensure only high-quality code improvements are saved for GNN training.
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


@dataclass
class ScraperStats:
    """Track scraping statistics across all filter stages."""

    fetched: int = 0
    filtered_message: int = 0
    filtered_file_type: int = 0
    filtered_size: int = 0
    filtered_syntax: int = 0
    filtered_quality: int = 0
    accepted: int = 0
    errors: int = 0

    def acceptance_rate(self) -> float:
        """Calculate overall acceptance rate."""
        if self.fetched == 0:
            return 0.0
        return 100 * self.accepted / self.fetched

    def print_progress(self):
        """Display filtering funnel."""
        print(f"""
╔══════════════════════════════════════╗
║     GitHub Scraping Progress        ║
╠══════════════════════════════════════╣
║ Fetched:              {self.fetched:>6}     ║
║ ├─ Message filter:    {self.filtered_message:>6} ❌  ║
║ ├─ File type filter:  {self.filtered_file_type:>6} ❌  ║
║ ├─ Size filter:       {self.filtered_size:>6} ❌  ║
║ ├─ Syntax filter:     {self.filtered_syntax:>6} ❌  ║
║ ├─ Quality filter:    {self.filtered_quality:>6} ❌  ║
║ └─ ACCEPTED:          {self.accepted:>6} ✅  ║
║                                      ║
║ Acceptance rate: {self.acceptance_rate():>6.1f}%      ║
║ Errors: {self.errors:>6}                   ║
╚══════════════════════════════════════╝
        """)


@dataclass
class CommitData:
    """Structured commit information."""

    sha: str
    repo: str
    message: str
    author: str
    timestamp: str
    files: List[str]
    url: str

    # Extracted code
    before_code: Optional[str] = None
    after_code: Optional[str] = None

    # Quality assessment
    quality_score: int = 0
    metrics: Dict = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class QualityThresholds:
    """Quality filtering thresholds."""

    # Size constraints (relaxed for full file extraction)
    min_lines_changed: int = 2
    max_lines_changed: int = 5000  # Increased from 300 - we extract full files
    max_files_changed: int = 10  # Increased from 5

    # Code quality (relaxed)
    min_code_ratio: float = 0.2  # Reduced from 0.3
    min_additions: int = 1  # Reduced from 2
    min_deletions: int = 1  # Reduced from 2
    max_addition_deletion_ratio: float = 10.0  # Increased from 5.0

    # Quality score
    min_quality_score: int = 60  # Will be overridden by --min-quality flag


class GitHubQualityScraper:
    """Main scraper with multi-stage quality filtering."""

    # Commit message patterns
    REJECT_MESSAGE_PATTERNS = [
        r"^merge",
        r"^bump version",
        r"^update.*readme",
        r"^formatting",
        r"^typo",
        r"^wip\b",
        r"^Revert",
        r"^\d+\.\d+\.\d+$",  # Version numbers
    ]

    ACCEPT_MESSAGE_PATTERNS = [
        r"\bfix\b",
        r"\bbug\b",
        r"\bsecurity\b",
        r"\bvulnerab",
        r"\brefactor\b",
        r"\bimprove\b",
        r"\boptimiz",
    ]

    # File patterns
    REJECT_FILE_PATTERNS = [
        r"\.md$",
        r"\.txt$",
        r"\.json$",
        r"\.yaml$",
        r"\.yml$",
        r"__pycache__",
        r"\.pyc$",
        r"/test_.*\.py$",
        r"^test_.*\.py$",
        r"setup\.py$",
        r"__init__\.py$",
    ]

    ACCEPT_FILE_PATTERNS = [
        r"\.py$",
    ]

    def __init__(
        self,
        db_path: Path,
        github_token: Optional[str] = None,
        thresholds: Optional[QualityThresholds] = None,
    ):
        """Initialize scraper.

        Args:
            db_path: Path to SQLite database for storing lessons
            github_token: Optional GitHub API token for higher rate limits
            thresholds: Quality filtering thresholds
        """
        self.db_path = db_path
        self.github_token = github_token
        self.thresholds = thresholds or QualityThresholds()
        self.stats = ScraperStats()
        self.session = requests.Session()

        if github_token:
            self.session.headers.update({
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            })

        self._init_database()

    def _init_database(self):
        """Initialize database with lessons table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='lessons'
        """)

        if not cursor.fetchone():
            # Create table matching your existing schema
            cursor.execute("""
                CREATE TABLE lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    before_code TEXT NOT NULL,
                    after_code TEXT NOT NULL,
                    test_code TEXT,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            conn.commit()

        conn.close()

    def passes_message_filter(self, commit: CommitData) -> bool:
        """Stage 1: Filter by commit message."""
        message = commit.message.lower()

        # Reject patterns
        for pattern in self.REJECT_MESSAGE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return False

        # Accept patterns (must match at least one)
        for pattern in self.ACCEPT_MESSAGE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return True

        return False

    def passes_file_filter(self, commit: CommitData) -> bool:
        """Stage 2: Filter by file types."""
        if len(commit.files) > self.thresholds.max_files_changed:
            return False

        for file_path in commit.files:
            # Must match accept pattern
            if not any(re.search(p, file_path) for p in self.ACCEPT_FILE_PATTERNS):
                return False

            # Must not match reject pattern
            if any(re.search(p, file_path) for p in self.REJECT_FILE_PATTERNS):
                return False

        return True

    def passes_size_filter(self, before_code: str, after_code: str) -> bool:
        """Stage 3: Filter by diff size."""
        before_lines = [l.strip() for l in before_code.split('\n') if l.strip()]
        after_lines = [l.strip() for l in after_code.split('\n') if l.strip()]

        additions = len([l for l in after_lines if l not in before_lines])
        deletions = len([l for l in before_lines if l not in after_lines])
        total_changes = additions + deletions

        # Check size constraints
        if total_changes < self.thresholds.min_lines_changed:
            return False
        if total_changes > self.thresholds.max_lines_changed:
            return False

        # Must have both additions and deletions
        if additions < self.thresholds.min_additions:
            return False
        if deletions < self.thresholds.min_deletions:
            return False

        # Check addition/deletion ratio
        if additions > 0 and deletions > 0:
            ratio = max(additions, deletions) / min(additions, deletions)
            if ratio > self.thresholds.max_addition_deletion_ratio:
                return False

        # Calculate code ratio (exclude whitespace/comments)
        code_lines = len([l for l in after_lines if l and not l.startswith('#')])
        total_lines = len(after_lines)
        if total_lines > 0:
            code_ratio = code_lines / total_lines
            if code_ratio < self.thresholds.min_code_ratio:
                return False

        return True

    def validate_syntax(self, before_code: str, after_code: str) -> bool:
        """Stage 4: Validate both versions parse as valid Python."""
        try:
            before_ast = ast.parse(before_code)
            after_ast = ast.parse(after_code)

            # Must have meaningful structure (not just imports)
            before_nodes = list(ast.walk(before_ast))
            after_nodes = list(ast.walk(after_ast))

            if len(before_nodes) < 5 or len(after_nodes) < 5:
                return False

            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    def assess_quality(self, commit: CommitData) -> bool:
        """Stage 5: Semantic quality assessment.

        Returns True if quality score >= threshold.
        Updates commit.quality_score and commit.metrics.
        """
        before_code = commit.before_code
        after_code = commit.after_code

        if not before_code or not after_code:
            return False

        metrics = {}
        score = 0

        try:
            # Parse ASTs
            before_ast = ast.parse(before_code)
            after_ast = ast.parse(after_code)

            # Complexity metrics
            before_complexity = self._calculate_complexity(before_ast)
            after_complexity = self._calculate_complexity(after_ast)
            metrics["complexity_delta"] = after_complexity - before_complexity

            # Complexity reduction is good
            if metrics["complexity_delta"] < -2:
                score += 20
            elif metrics["complexity_delta"] < 0:
                score += 10

            # Security improvements
            metrics["removes_eval"] = "eval(" in before_code and "eval(" not in after_code
            metrics["removes_exec"] = "exec(" in before_code and "exec(" not in after_code
            metrics["adds_validation"] = self._count_validations(after_ast) > self._count_validations(before_ast)

            if metrics["removes_eval"]:
                score += 15
            if metrics["removes_exec"]:
                score += 15
            if metrics["adds_validation"]:
                score += 10

            # Quality improvements
            metrics["adds_error_handling"] = self._count_try_except(after_ast) > self._count_try_except(before_ast)
            metrics["adds_type_hints"] = self._count_type_hints(after_ast) > self._count_type_hints(before_ast)
            metrics["adds_docstrings"] = self._count_docstrings(after_ast) > self._count_docstrings(before_ast)

            if metrics["adds_error_handling"]:
                score += 15
            if metrics["adds_type_hints"]:
                score += 10
            if metrics["adds_docstrings"]:
                score += 5

            # Structure improvements
            before_functions = self._count_functions(before_ast)
            after_functions = self._count_functions(after_ast)
            metrics["function_count_delta"] = after_functions - before_functions

            # Modularization is good (more functions from refactoring)
            if metrics["function_count_delta"] > 0 and "refactor" in commit.message.lower():
                score += 10

            commit.quality_score = min(100, score)
            commit.metrics = metrics

            return commit.quality_score >= self.thresholds.min_quality_score

        except Exception as e:
            print(f"  - Quality assessment error: {e}")
            return False

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def _count_try_except(self, tree: ast.AST) -> int:
        """Count try/except blocks."""
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))

    def _count_type_hints(self, tree: ast.AST) -> int:
        """Count type annotations."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns:
                    count += 1
                count += sum(1 for arg in node.args.args if arg.annotation)
        return count

    def _count_docstrings(self, tree: ast.AST) -> int:
        """Count docstrings."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    count += 1
        return count

    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions."""
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))

    def _count_validations(self, tree: ast.AST) -> int:
        """Count input validation patterns (isinstance, type checks, etc.)."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('isinstance', 'issubclass', 'hasattr'):
                        count += 1
        return count

    def infer_category(self, commit: CommitData) -> str:
        """Infer CEFR category from commit characteristics."""
        message = commit.message.lower()
        score = commit.quality_score

        # Security fixes are typically C1/C2
        if "security" in message or "vulnerab" in message:
            return "c2" if score >= 80 else "c1"

        # Refactorings depend on complexity
        if "refactor" in message:
            if commit.metrics.get("complexity_delta", 0) < -5:
                return "c1"  # Significant refactoring
            return "b2"

        # Based on quality score
        if score >= 80:
            return "c1"
        elif score >= 70:
            return "b2"
        elif score >= 60:
            return "b1"
        else:
            return "a2"

    def synthesize_test_code(self, commit: CommitData) -> str:
        """Stage 6: Generate test code for the fix."""
        before_code = commit.before_code
        after_code = commit.after_code
        message = commit.message.lower()

        try:
            tree = ast.parse(after_code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not functions:
                return self._generate_basic_test(after_code)

            # Security test
            if "security" in message or "vulnerab" in message:
                return self._generate_security_test(functions, before_code, after_code, commit)

            # Bug fix test
            if "fix" in message or "bug" in message:
                return self._generate_regression_test(functions, commit)

            # Refactor test
            if "refactor" in message:
                return self._generate_equivalence_test(functions, commit)

            # Default
            return self._generate_smoke_test(functions, commit)

        except Exception as e:
            print(f"  - Test synthesis error: {e}")
            return self._generate_basic_test(after_code)

    def _generate_basic_test(self, code: str) -> str:
        """Generate basic import test."""
        return f'''"""Basic smoke test for generated code."""
import pytest

def test_code_imports():
    """Verify the code can be imported without errors."""
    try:
        # Code should at least parse
        compile({repr(code)}, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Code has syntax errors: {{e}}")
'''

    def _generate_security_test(self, functions: List[str], before_code: str, after_code: str, commit: CommitData) -> str:
        """Generate security-focused test."""
        main_func = functions[0] if functions else "main"

        test = f'''"""Security regression test."""
import pytest

# The fix addressed: {commit.message}

def test_no_eval_in_code():
    """Verify dangerous eval() was removed."""
    assert "eval(" not in {repr(after_code)}, "Code should not contain eval()"

def test_no_exec_in_code():
    """Verify dangerous exec() was removed."""
    assert "exec(" not in {repr(after_code)}, "Code should not contain exec()"
'''

        if "sql" in commit.message.lower() or "injection" in commit.message.lower():
            test += '''
def test_no_string_formatting_in_queries():
    """Verify SQL injection vulnerability was fixed."""
    # Check for parameterized queries instead of string formatting
    assert "?" in after_code or "%s" in after_code, "Should use parameterized queries"
'''

        return test

    def _generate_regression_test(self, functions: List[str], commit: CommitData) -> str:
        """Generate regression test for bug fix."""
        main_func = functions[0] if functions else "main"

        return f'''"""Regression test for bug fix."""
import pytest

# Bug fix: {commit.message}

def test_{main_func}_basic():
    """Test that {main_func} works correctly after fix."""
    # TODO: Add specific test case that would have failed before fix
    pass

def test_{main_func}_edge_cases():
    """Test edge cases that might have caused the bug."""
    # TODO: Add edge case tests
    pass
'''

    def _generate_equivalence_test(self, functions: List[str], commit: CommitData) -> str:
        """Generate test that before/after produce same results."""
        main_func = functions[0] if functions else "main"

        return f'''"""Equivalence test for refactoring."""
import pytest

# Refactoring: {commit.message}

def test_{main_func}_behavior_unchanged():
    """Verify refactoring didn't change behavior."""
    # Before and after should produce same results
    # TODO: Add test cases
    pass

def test_{main_func}_performance():
    """Verify refactoring improved performance or maintainability."""
    # TODO: Add performance or complexity assertions
    pass
'''

    def _generate_smoke_test(self, functions: List[str], commit: CommitData) -> str:
        """Generate basic smoke test."""
        main_func = functions[0] if functions else "main"

        return f'''"""Smoke test for code change."""
import pytest

# Change: {commit.message}

def test_{main_func}_executes():
    """Verify {main_func} can be called without errors."""
    # TODO: Add basic execution test
    pass
'''

    def save_lesson(self, commit: CommitData, test_code: str):
        """Save validated lesson to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        name = f"github_{commit.sha[:8]}_{commit.category}"
        description = f"[GitHub] {commit.message[:100]}"

        metadata = json.dumps({
            "source": "github",
            "repo": commit.repo,
            "commit_sha": commit.sha,
            "commit_message": commit.message,
            "commit_url": commit.url,
            "author": commit.author,
            "timestamp": commit.timestamp,
            "quality_score": commit.quality_score,
            "metrics": commit.metrics,
        })

        try:
            cursor.execute("""
                INSERT INTO lessons (name, description, before_code, after_code, test_code, category, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, description, commit.before_code, commit.after_code, test_code, commit.category, metadata))

            conn.commit()
            self.stats.accepted += 1

        except sqlite3.IntegrityError:
            # Duplicate name, skip
            pass
        except Exception as e:
            print(f"  - Database error: {e}")
            self.stats.errors += 1
        finally:
            conn.close()

    def scrape(self, target_count: int = 10000, max_attempts: int = 100000):
        """Main scraping loop.

        Args:
            target_count: Target number of quality examples to collect
            max_attempts: Maximum commits to fetch before stopping
        """
        print(f"🚀 Starting GitHub Quality Scraper")
        print(f"   Target: {target_count} quality examples")
        print(f"   Max attempts: {max_attempts}")
        print(f"   Database: {self.db_path}")
        print()

        # This will be implemented in the next file (github_api_connector.py)
        # For now, this is the main entry point
        raise NotImplementedError(
            "GitHub API connector not yet implemented. "
            "Next step: implement fetch_commits_from_github() method."
        )
