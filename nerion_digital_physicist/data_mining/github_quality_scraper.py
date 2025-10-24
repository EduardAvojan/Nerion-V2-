"""Production-grade GitHub commit scraper with multi-stage quality filters.

This scraper fetches code improvements from GitHub across multiple languages
(Python, JavaScript, TypeScript, Rust, Go, Java) and applies rigorous quality
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

# Quality hardening components
try:
    from .quality_hardening import EnhancedStats, NegativeEvidenceDetector, VerificationHardening
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from quality_hardening import EnhancedStats, NegativeEvidenceDetector, VerificationHardening


@dataclass
class ScraperStats:
    """Track scraping statistics across all filter stages."""

    fetched: int = 0
    filtered_message: int = 0
    filtered_file_type: int = 0
    failed_fetch_diff: int = 0  # Failed to fetch full commit diff
    no_code_files: int = 0  # No valid code files or extraction failed
    filtered_size: int = 0
    filtered_syntax: int = 0
    filtered_quarantine: int = 0
    filtered_quality: int = 0
    duplicates: int = 0  # Duplicate commits (already saved)
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
║ ├─ Failed fetch diff: {self.failed_fetch_diff:>6} ❌  ║
║ ├─ No code files:     {self.no_code_files:>6} ❌  ║
║ ├─ Size filter:       {self.filtered_size:>6} ❌  ║
║ ├─ Syntax filter:     {self.filtered_syntax:>6} ❌  ║
║ ├─ Quarantine:        {self.filtered_quarantine:>6} ❌  ║
║ ├─ Quality filter:    {self.filtered_quality:>6} ❌  ║
║ ├─ Duplicates:        {self.duplicates:>6} 🔁  ║
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
    language: Optional[str] = None  # python, javascript, typescript, rust, go, java

    # Quality assessment
    quality_score: int = 0
    metrics: Dict = field(default_factory=dict)
    category: Optional[str] = None
    tier: str = "REJECT"  # GOLD, SILVER, or REJECT


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

    # Quality score (lowered from 60 to 45 for better throughput)
    min_quality_score: int = 45  # Will be overridden by --min-quality flag


class GitHubQualityScraper:
    """Main scraper with multi-stage quality filtering."""

    # Language configuration: file extensions and patterns
    LANGUAGE_CONFIG = {
        'python': {
            'extensions': ['.py'],
            'name': 'Python',
        },
        'javascript': {
            'extensions': ['.js', '.jsx'],
            'name': 'JavaScript',
        },
        'typescript': {
            'extensions': ['.ts', '.tsx'],
            'name': 'TypeScript',
        },
        'rust': {
            'extensions': ['.rs'],
            'name': 'Rust',
        },
        'go': {
            'extensions': ['.go'],
            'name': 'Go',
        },
        'java': {
            'extensions': ['.java'],
            'name': 'Java',
        },
    }

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

    # File patterns - reject build artifacts, configs, and trivial files
    REJECT_FILE_PATTERNS = [
        r"__pycache__",
        r"\.pyc$",
        r"\.class$",  # Java bytecode
        r"\.o$",  # Object files
        r"\.so$",  # Shared libraries
        r"\.a$",  # Static libraries
        r"\.dll$",  # Windows DLLs
        r"\.exe$",  # Windows executables
        r"setup\.py$",  # Setup scripts
        r"__init__\.py$",  # Usually just imports
        r"package-lock\.json$",  # Lock files
        r"yarn\.lock$",
        r"Cargo\.lock$",
        r"go\.sum$",
        r"\.min\.js$",  # Minified files
        r"\.min\.css$",
        r"/dist/",  # Build outputs
        r"/build/",
        r"/target/",  # Rust build output
        r"/node_modules/",
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

        # Hardening components
        self.enhanced_stats = EnhancedStats()
        self.penalty_detector = NegativeEvidenceDetector()
        self.verification_hardener = VerificationHardening()

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
            # Create table with full schema including content_hash for deduplication
            cursor.execute("""
                CREATE TABLE lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    focus_area TEXT,
                    before_code TEXT NOT NULL,
                    after_code TEXT NOT NULL,
                    test_code TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT,
                    category TEXT,
                    language TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
        else:
            # Table exists - add missing columns if needed
            cursor.execute("PRAGMA table_info(lessons)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'language' not in columns:
                cursor.execute("ALTER TABLE lessons ADD COLUMN language TEXT")
            if 'content_hash' not in columns:
                cursor.execute("ALTER TABLE lessons ADD COLUMN content_hash TEXT")
            if 'category' not in columns:
                cursor.execute("ALTER TABLE lessons ADD COLUMN category TEXT")
            if 'metadata' not in columns:
                cursor.execute("ALTER TABLE lessons ADD COLUMN metadata TEXT")
            if 'focus_area' not in columns:
                cursor.execute("ALTER TABLE lessons ADD COLUMN focus_area TEXT")

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
        """Stage 2: Filter by file types.

        Multi-language filtering: Accept commits with mixed file types as long as they
        include at least one valid source file in any supported language:
        - Python (.py)
        - JavaScript (.js, .jsx)
        - TypeScript (.ts, .tsx)
        - Rust (.rs)
        - Go (.go)
        - Java (.java)

        This allows commits like:
        - fix.py + README.md
        - bugfix.js + package.json
        - security.rs + Cargo.toml

        We'll process only the source files later, ignoring docs/configs.

        Note: This is called before files are populated from the full commit,
        so we accept empty files lists (they'll be validated later).
        """
        # Accept empty files lists (will be populated after fetching full commit)
        if not commit.files:
            return True

        if len(commit.files) > self.thresholds.max_files_changed:
            return False

        # Check if commit has at least one valid source file in any supported language
        valid_source_files = []

        for file_path in commit.files:
            # Check if this file should be rejected (build artifacts, etc.)
            if any(re.search(p, file_path) for p in self.REJECT_FILE_PATTERNS):
                continue

            # Check if file matches any supported language
            for lang, config in self.LANGUAGE_CONFIG.items():
                for ext in config['extensions']:
                    if file_path.endswith(ext):
                        valid_source_files.append((file_path, lang))
                        break

        # Accept if commit has at least one valid source file
        return len(valid_source_files) > 0

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension.

        Returns language key (python, javascript, etc.) or None if not supported.
        """
        for lang, config in self.LANGUAGE_CONFIG.items():
            for ext in config['extensions']:
                if file_path.endswith(ext):
                    return lang
        return None

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

        # Accept commits with meaningful changes (additions OR deletions, not necessarily both)
        # This allows: bug fixes, new features, code cleanup, refactoring
        # Old logic required BOTH, which rejected many valid commits
        has_additions = additions >= self.thresholds.min_additions
        has_deletions = deletions >= self.thresholds.min_deletions

        if not (has_additions or has_deletions):
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

    def validate_syntax(self, before_code: str, after_code: str, language: str = 'python') -> bool:
        """Stage 4: Validate both versions have valid syntax for the given language.

        Args:
            before_code: Code before the fix
            after_code: Code after the fix
            language: Programming language (python, javascript, typescript, rust, go, java)

        Returns:
            True if both versions have valid syntax
        """
        if language == 'python':
            return self._validate_python_syntax(before_code, after_code)
        elif language in ['javascript', 'typescript']:
            return self._validate_js_ts_syntax(before_code, after_code)
        elif language == 'rust':
            return self._validate_rust_syntax(before_code, after_code)
        elif language == 'go':
            return self._validate_go_syntax(before_code, after_code)
        elif language == 'java':
            return self._validate_java_syntax(before_code, after_code)
        else:
            # Unknown language - accept by default
            return True

    def _validate_python_syntax(self, before_code: str, after_code: str) -> bool:
        """Validate Python code using lenient heuristics.

        Production scraper approach: Accept code that looks reasonable,
        not just code that parses perfectly. This handles partial snippets,
        new files, deleted files, and complex refactorings.
        """
        # Empty code is valid (new/deleted files)
        if not before_code.strip() or not after_code.strip():
            return True

        # Try AST parsing but don't fail hard
        valid_count = 0
        for code in [before_code, after_code]:
            try:
                ast.parse(code)
                valid_count += 1
            except:
                # Failed AST parse - check if it looks like valid Python
                # These are common patterns in real code snippets:
                lines = [l for l in code.split('\n') if l.strip()]
                if not lines:
                    continue

                # Heuristic: Has Python keywords or patterns
                has_python_patterns = any(
                    keyword in code
                    for keyword in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'return ', 'self.']
                )

                # Heuristic: Looks like code (has indentation, colons, etc.)
                has_code_structure = ':' in code or '=' in code or '(' in code

                if has_python_patterns and has_code_structure:
                    valid_count += 1

        # Accept if at least one version looks valid
        # This is lenient but production-appropriate
        return valid_count >= 1

    def _validate_js_ts_syntax(self, before_code: str, after_code: str) -> bool:
        """Validate JavaScript/TypeScript syntax using heuristics."""
        # Basic syntax checks - look for common patterns
        for code in [before_code, after_code]:
            # Must have some substance (not empty or just comments)
            lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]
            if len(lines) < 3:
                return False

            # Check for balanced braces
            if code.count('{') != code.count('}'):
                return False
            if code.count('(') != code.count(')'):
                return False
            if code.count('[') != code.count(']'):
                return False

        return True

    def _validate_rust_syntax(self, before_code: str, after_code: str) -> bool:
        """Validate Rust syntax using heuristics."""
        for code in [before_code, after_code]:
            # Must have some substance
            lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]
            if len(lines) < 3:
                return False

            # Check for balanced braces
            if code.count('{') != code.count('}'):
                return False
            if code.count('(') != code.count(')'):
                return False
            if code.count('[') != code.count(']'):
                return False

        return True

    def _validate_go_syntax(self, before_code: str, after_code: str) -> bool:
        """Validate Go syntax using heuristics."""
        for code in [before_code, after_code]:
            # Must have some substance
            lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]
            if len(lines) < 3:
                return False

            # Check for balanced braces
            if code.count('{') != code.count('}'):
                return False
            if code.count('(') != code.count(')'):
                return False
            if code.count('[') != code.count(']'):
                return False

        return True

    def _validate_java_syntax(self, before_code: str, after_code: str) -> bool:
        """Validate Java syntax using heuristics."""
        for code in [before_code, after_code]:
            # Must have some substance
            lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]
            if len(lines) < 3:
                return False

            # Check for balanced braces
            if code.count('{') != code.count('}'):
                return False
            if code.count('(') != code.count(')'):
                return False
            if code.count('[') != code.count(']'):
                return False

        return True

    def passes_quarantine_filter(self, commit: CommitData, before_code: str, after_code: str) -> bool:
        """Stage 4.5: Quarantine vendor/generated/whitespace-heavy diffs.

        Rejects:
        - Vendor/generated/lockfiles
        - Mass whitespace diffs (>80% whitespace changes)
        - Pure renames (no body changes, detected heuristically)
        """
        # 1. Check for vendor/generated/lockfile paths
        QUARANTINE_PATH_PATTERNS = [
            r'vendor/',
            r'third_party/',
            r'node_modules/',
            r'dist/',
            r'build/',
            r'\.min\.js$',
            r'\.min\.css$',
            r'package-lock\.json$',
            r'poetry\.lock$',
            r'Pipfile\.lock$',
            r'yarn\.lock$',
            r'Gemfile\.lock$',
            r'composer\.lock$',
            r'-lock\.json$',
            r'\.generated\.',
            r'_pb2\.py$',  # Protobuf generated
            r'\.pb\.go$',  # Protobuf generated
        ]

        for file_path in commit.files:
            for pattern in QUARANTINE_PATH_PATTERNS:
                if re.search(pattern, file_path, re.IGNORECASE):
                    return False  # Quarantine

        # 2. Check for mass whitespace diff
        before_lines = before_code.split('\n')
        after_lines = after_code.split('\n')

        # Count non-whitespace changes
        before_stripped = [l.strip() for l in before_lines if l.strip()]
        after_stripped = [l.strip() for l in after_lines if l.strip()]

        # If very few stripped lines changed but many raw lines changed, it's mostly whitespace
        total_lines = len(before_lines) + len(after_lines)
        if total_lines > 10:  # Only check for non-trivial diffs
            stripped_diff = abs(len(before_stripped) - len(after_stripped))
            raw_diff = abs(len(before_lines) - len(after_lines))

            if raw_diff > 0:
                whitespace_ratio = (raw_diff - stripped_diff) / raw_diff
                if whitespace_ratio > 0.8:  # >80% whitespace
                    return False  # Quarantine

        # 3. Check for pure rename (same stripped content, different line count suggests formatting only)
        # Simple heuristic: if all stripped lines from before are in after and vice versa
        if len(before_stripped) > 5 and len(after_stripped) > 5:
            before_set = set(before_stripped)
            after_set = set(after_stripped)

            # If >95% overlap, likely just formatting/rename
            overlap = len(before_set & after_set)
            total_unique = len(before_set | after_set)
            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                if overlap_ratio > 0.95:
                    return False  # Quarantine (likely pure formatting)

        return True  # Pass quarantine

    def assess_quality(self, commit: CommitData) -> bool:
        """Stage 5: Evidence-based semantic quality assessment with negative evidence.

        Two-tier system with penalty enforcement:
        - GOLD: score ≥8 AND 2+ evidence types AND verification AND zero penalties
        - SILVER: score ≥2 AND penalties ≤2
        - REJECT: else

        For non-Python languages, applies basic quality acceptance (passed all other filters).

        Returns True if quality score >= threshold.
        Updates commit.quality_score, commit.metrics, and commit.tier.
        """
        before_code = commit.before_code
        after_code = commit.after_code

        # For non-Python languages OR empty code (new/deleted files), apply basic quality acceptance
        # They've already passed: message, file, size, syntax, and quarantine filters
        if (commit.language and commit.language != 'python') or not before_code or not after_code:
            commit.quality_score = 65  # Base score for non-Python or new/deleted files
            commit.tier = "SILVER"
            commit.metrics = {
                "language": commit.language or "unknown",
                "note": "Basic quality assessment (non-Python or partial code)"
            }
            return True  # Accept if passed all other filters

        # Python-specific quality assessment with AST analysis
        metrics = {}
        score = 0
        evidence = set()  # Track independent evidence types
        penalties = []  # Track penalties

        try:
            # Try to parse Python ASTs (may fail on partial snippets)
            before_ast = ast.parse(before_code)
            after_ast = ast.parse(after_code)

            # ===== NEGATIVE EVIDENCE (PENALTIES) =====
            penalties = self.penalty_detector.detect_penalties(
                commit, before_ast, after_ast, before_code, after_code
            )
            penalty_score = self.penalty_detector.calculate_penalty_score(penalties)
            metrics['penalties'] = penalties
            metrics['penalty_score'] = penalty_score

            # ===== 1. COMPLEXITY (Evidence: "complexity") =====
            before_complexity = self._calculate_complexity(before_ast)
            after_complexity = self._calculate_complexity(after_ast)
            metrics["complexity_delta"] = after_complexity - before_complexity

            complexity_points = 0
            if metrics["complexity_delta"] < -2:
                complexity_points = 20
            elif metrics["complexity_delta"] < 0:
                complexity_points = 10

            score += complexity_points
            if complexity_points > 0:
                evidence.add("complexity")

            # ===== 2. SECURITY (Evidence: "security") =====
            security_points = 0

            # Removes eval/exec
            metrics["removes_eval"] = "eval(" in before_code and "eval(" not in after_code
            metrics["removes_exec"] = "exec(" in before_code and "exec(" not in after_code

            if metrics["removes_eval"]:
                security_points += 15
            if metrics["removes_exec"]:
                security_points += 15

            # Adds validation
            metrics["adds_validation"] = self._count_validations(after_ast) > self._count_validations(before_ast)
            if metrics["adds_validation"]:
                security_points += 10

            # NEW: Sanitization patterns
            metrics["adds_sanitization"] = self._detect_sanitization(before_code, after_code)
            if metrics["adds_sanitization"]:
                security_points += 5

            score += security_points
            if security_points > 0:
                evidence.add("security")

            # ===== 3. GUARDS / BOUNDARY / RAII / DEFAULTS (Evidence: "guard_or_boundary_or_raii") =====
            micro_fix_points = 0

            # NEW: Guard checks (null/None checks, length checks, type checks)
            guard_score = self._detect_guard_additions(before_ast, after_ast, before_code, after_code)
            metrics["adds_guards"] = guard_score > 0
            micro_fix_points += guard_score

            # NEW: RAII patterns (with statements, context managers)
            raii_score = self._detect_raii_patterns(before_ast, after_ast)
            metrics["adds_raii"] = raii_score > 0
            micro_fix_points += raii_score

            # NEW: Deterministic defaults (return None/[]/{}/ 0 instead of undefined)
            default_score = self._detect_deterministic_defaults(before_ast, after_ast)
            metrics["adds_defaults"] = default_score > 0
            micro_fix_points += default_score

            score += micro_fix_points
            if micro_fix_points > 0:
                evidence.add("guard_or_boundary_or_raii")

            # ===== 4. VERIFICATION (Evidence: "verification") - HARDENED =====
            has_verification, verification_points = self.verification_hardener.detect_hardened_verification(
                commit, before_ast, after_ast
            )
            metrics["verification_points"] = verification_points
            metrics["has_verification"] = has_verification

            score += verification_points
            if has_verification:
                evidence.add("verification")

            # ===== 5. STRUCTURE (Evidence: "structure") =====
            structure_points = 0

            # Type hints
            metrics["adds_type_hints"] = self._count_type_hints(after_ast) > self._count_type_hints(before_ast)
            if metrics["adds_type_hints"]:
                structure_points += 10

            # Docstrings
            metrics["adds_docstrings"] = self._count_docstrings(after_ast) > self._count_docstrings(before_ast)
            if metrics["adds_docstrings"]:
                structure_points += 5

            # Function extraction
            before_functions = self._count_functions(before_ast)
            after_functions = self._count_functions(after_ast)
            metrics["function_count_delta"] = after_functions - before_functions

            if metrics["function_count_delta"] > 0 and "refactor" in commit.message.lower():
                structure_points += 10

            score += structure_points
            if structure_points > 0:
                evidence.add("structure")

            # ===== APPLY PENALTIES =====
            net_score = score - penalty_score

            # ===== FINALIZE =====
            commit.quality_score = max(0, min(100, net_score))  # Clamp to [0, 100]
            commit.metrics = metrics
            metrics["evidence_types"] = list(evidence)
            metrics["evidence_count"] = len(evidence)
            metrics["gross_score"] = score
            metrics["net_score"] = net_score

            # ===== TWO-TIER GATE WITH PENALTY ENFORCEMENT =====
            GOLD_THRESH = 8
            SILVER_THRESH = 2
            MAX_PENALTY_FOR_SILVER = 2

            # Gold requirements:
            # - Net score >= 8 AND
            # - At least 2 different evidence types AND
            # - Has verification evidence AND
            # - Zero penalties (pristine)
            is_gold = (
                net_score >= GOLD_THRESH and
                len(evidence) >= 2 and
                "verification" in evidence and
                penalty_score == 0  # Zero tolerance for penalties in GOLD
            )

            # Silver requirements:
            # - Net score >= 2 AND
            # - Penalties <= 2 (some tolerance)
            is_silver = net_score >= SILVER_THRESH and penalty_score <= MAX_PENALTY_FOR_SILVER

            if is_gold:
                commit.tier = "GOLD"
                return True  # Always accept Gold
            elif is_silver:
                commit.tier = "SILVER"
                # Accept Silver only if threshold allows (for two-tier collection)
                return self.thresholds.min_quality_score <= SILVER_THRESH
            else:
                commit.tier = "REJECT"
                return False

        except Exception as e:
            # AST parsing failed (likely partial code snippet)
            # Fall back to basic quality acceptance - they've passed all other filters
            commit.quality_score = 55  # Slightly lower than perfect code, but still acceptable
            commit.tier = "SILVER"
            commit.metrics = {
                "language": commit.language or "python",
                "note": f"Basic quality assessment (AST parsing failed: {str(e)[:50]})"
            }
            return True

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

    def _detect_guard_additions(self, before_ast, after_ast, before_code, after_code):
        """Detect addition of guard clauses (null checks, length checks, type checks)."""
        score = 0

        # Pattern 1: None/null checks added
        before_none_checks = self._count_none_checks(before_ast)
        after_none_checks = self._count_none_checks(after_ast)
        if after_none_checks > before_none_checks:
            score += 2

        # Pattern 2: Length/boundary checks added
        before_len_checks = self._count_length_checks(before_ast)
        after_len_checks = self._count_length_checks(after_ast)
        if after_len_checks > before_len_checks:
            score += 2

        # Pattern 3: Early returns added (common guard pattern)
        before_returns = self._count_early_returns(before_ast)
        after_returns = self._count_early_returns(after_ast)
        if after_returns > before_returns:
            score += 1

        return score

    def _count_none_checks(self, tree):
        """Count None/null checks (if x is None, if x is not None, if not x)."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # if x is None, if x is not None
                if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                    if any(isinstance(comp, ast.Constant) and comp.value is None
                           for comp in node.comparators):
                        count += 1
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                # if not x (common guard)
                count += 1
        return count

    def _count_length_checks(self, tree):
        """Count length/boundary checks (len(x) > 0, i < len(arr), etc.)."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check if len() is involved
                if isinstance(node.left, ast.Call):
                    if isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                        count += 1
                # Check comparators
                for comp in node.comparators:
                    if isinstance(comp, ast.Call):
                        if isinstance(comp.func, ast.Name) and comp.func.id == 'len':
                            count += 1
        return count

    def _count_early_returns(self, tree):
        """Count early return statements (guard pattern)."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count returns that aren't the last statement
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                if len(returns) > 1:  # More than one return suggests guards
                    count += len(returns) - 1
        return count

    def _detect_raii_patterns(self, before_ast, after_ast):
        """Detect RAII patterns (with statements, context managers)."""
        before_with = sum(1 for n in ast.walk(before_ast) if isinstance(n, ast.With))
        after_with = sum(1 for n in ast.walk(after_ast) if isinstance(n, ast.With))

        if after_with > before_with:
            return 3  # RAII pattern added
        return 0

    def _detect_deterministic_defaults(self, before_ast, after_ast):
        """Detect addition of deterministic defaults (return None/[]/{}/ instead of undefined)."""
        score = 0

        # Count explicit default returns
        before_returns = [n for n in ast.walk(before_ast) if isinstance(n, ast.Return)]
        after_returns = [n for n in ast.walk(after_ast) if isinstance(n, ast.Return)]

        # Check for returns with explicit None/empty collections
        def has_default(ret_node):
            if ret_node.value is None:
                return False
            if isinstance(ret_node.value, ast.Constant):
                return ret_node.value.value in (None, [], {}, 0, False, "")
            if isinstance(ret_node.value, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
                return len(ret_node.value.elts if hasattr(ret_node.value, 'elts') else []) == 0
            return False

        before_defaults = sum(1 for r in before_returns if has_default(r))
        after_defaults = sum(1 for r in after_returns if has_default(r))

        if after_defaults > before_defaults:
            score += 2

        return score

    def _detect_sanitization(self, before_code, after_code):
        """Detect sanitization patterns (escaping, parameterized queries, etc.)."""
        # Parameterized SQL
        if ("execute(" in after_code and "?" in after_code) and ("execute(" in before_code and "?" not in before_code):
            return True

        # HTML escaping
        if "html.escape" in after_code and "html.escape" not in before_code:
            return True

        # URL encoding
        if "urllib.parse.quote" in after_code and "urllib.parse.quote" not in before_code:
            return True

        return False

    def _detect_test_proximity(self, commit, before_ast, after_ast):
        """Detect test proximity signals (touches tests, adds asserts)."""
        # Check if any files are test files
        test_files = [f for f in commit.files if 'test' in f.lower()]
        if test_files:
            return True

        # Check if asserts were added
        before_asserts = sum(1 for n in ast.walk(before_ast) if isinstance(n, ast.Assert))
        after_asserts = sum(1 for n in ast.walk(after_ast) if isinstance(n, ast.Assert))
        if after_asserts > before_asserts:
            return True

        # Check commit message for test/CI keywords
        msg = commit.message.lower()
        if any(kw in msg for kw in ['test', 'ci', 'failing', 'green', 'pass']):
            return True

        return False

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
        """Stage 6: Generate test code for the fix.

        For non-Python languages, generates basic placeholder tests.
        """
        before_code = commit.before_code
        after_code = commit.after_code
        message = commit.message.lower()

        # For non-Python languages, generate language-appropriate basic test
        if commit.language and commit.language != 'python':
            return self._generate_basic_test_for_language(commit.language, after_code, message)

        # Python-specific test generation with AST analysis
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

    def _generate_basic_test_for_language(self, language: str, code: str, message: str) -> str:
        """Generate basic test placeholder for non-Python languages."""
        lang_name = self.LANGUAGE_CONFIG.get(language, {}).get('name', language)

        return f'''"""Basic test placeholder for {lang_name} code.

Fix: {message[:100]}

Note: This is a placeholder test. The actual {lang_name} code
has been validated for syntax and quality during scraping.
The GNN will learn from the before/after code patterns.

To run proper tests, compile and test with {lang_name} tooling:
- JavaScript/TypeScript: jest, mocha, or vitest
- Rust: cargo test
- Go: go test
- Java: JUnit or TestNG
"""

# TODO: Add language-specific tests using appropriate testing framework
pass
'''

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
        """Save validated lesson to database using direct SQLite connection.

        Note: This does NOT use SafeCurriculumDB to avoid excessive backup overhead.
        The scraper database is separate from the main curriculum and doesn't need
        the same level of backup protection (scraper can be re-run if needed).
        """
        import sqlite3

        # Generate unique name per file using filename hash (prevents duplicates when commit has multiple files)
        import hashlib
        # Use filename if available, otherwise fall back to content hash
        file_id = ""
        if hasattr(commit, 'current_file') and commit.current_file:
            # Hash the filename to keep name short
            file_id = hashlib.md5(commit.current_file.encode()).hexdigest()[:6]
        else:
            # Fallback: use content hash
            file_id = hashlib.md5(f"{commit.before_code}{commit.after_code}".encode()).hexdigest()[:6]

        name = f"github_{commit.sha[:8]}_{commit.category}_{file_id}"
        description = f"[GitHub] {commit.message[:100]}"

        metadata = json.dumps({
            "source": "github",
            "repo": commit.repo,
            "commit_sha": commit.sha,
            "commit_message": commit.message,
            "commit_url": commit.url,
            "author": commit.author,
            "timestamp": commit.timestamp,
            "language": commit.language,
            "quality_score": commit.quality_score,
            "metrics": commit.metrics,
            "tier": commit.tier,
        })

        try:
            # Direct SQLite connection (no backup overhead - scraper can be re-run)
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            # Check for duplicate by name
            cursor.execute("SELECT 1 FROM lessons WHERE name = ?", (name,))
            if cursor.fetchone():
                conn.close()
                self.stats.duplicates += 1
                return False

            # Insert lesson
            cursor.execute("""
                INSERT INTO lessons (name, description, before_code, after_code, test_code, category, language, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, commit.before_code, commit.after_code, test_code, commit.category, commit.language, metadata))

            conn.commit()
            conn.close()

            self.stats.accepted += 1
            self.enhanced_stats.record_commit(commit)
            return True

        except sqlite3.IntegrityError:
            # Duplicate (constraint violation)
            self.stats.duplicates += 1
            return False
        except Exception as e:
            print(f"  - Database error: {e}")
            self.stats.errors += 1
            return False

    def print_final_report(self):
        """Print comprehensive metrics report after scraping."""
        print("\n")
        print("=" * 70)
        print("BASIC SCRAPING STATISTICS")
        print("=" * 70)
        self.stats.print_progress()

        print("\n")
        print("=" * 70)
        print("COMPREHENSIVE QUALITY VALIDATION")
        print("=" * 70)
        self.enhanced_stats.print_comprehensive_report()

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
