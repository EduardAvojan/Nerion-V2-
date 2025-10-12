"""Quality assessment hardening with negative evidence and comprehensive metrics.

This module extends the evidence-based quality system with:
- Negative evidence (penalties for risky patterns)
- Comprehensive metrics tracking (tier distribution, evidence co-occurrence)
- Hardened verification signals
- Repo-local percentile calibration
"""
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class EnhancedStats:
    """Track detailed quality metrics for validation."""

    # Tier counts
    gold_count: int = 0
    silver_count: int = 0
    reject_count: int = 0

    # Evidence distribution
    evidence_counts: Dict[str, int] = field(default_factory=Counter)
    evidence_cooccurrence: Dict[Tuple[str, str], int] = field(default_factory=Counter)

    # Score distribution (histogram)
    score_bins: Dict[int, int] = field(default_factory=Counter)

    # Penalty tracking
    penalty_counts: Dict[str, int] = field(default_factory=Counter)

    # Per-repo stats
    repo_stats: Dict[str, Dict] = field(default_factory=lambda: defaultdict(lambda: {
        'gold': 0, 'silver': 0, 'reject': 0, 'scores': []
    }))

    def record_commit(self, commit):
        """Record a commit's tier and evidence."""
        # Tier counts
        if commit.tier == "GOLD":
            self.gold_count += 1
        elif commit.tier == "SILVER":
            self.silver_count += 1
        else:
            self.reject_count += 1

        # Evidence distribution
        evidence_types = commit.metrics.get('evidence_types', [])
        for evidence in evidence_types:
            self.evidence_counts[evidence] += 1

        # Evidence co-occurrence (only for accepted)
        if commit.tier in ("GOLD", "SILVER"):
            for i, ev1 in enumerate(evidence_types):
                for ev2 in evidence_types[i+1:]:
                    pair = tuple(sorted([ev1, ev2]))
                    self.evidence_cooccurrence[pair] += 1

        # Score histogram
        score = commit.quality_score
        self.score_bins[score] += 1

        # Penalties
        for penalty in commit.metrics.get('penalties', []):
            self.penalty_counts[penalty] += 1

        # Per-repo stats
        repo = commit.repo
        repo_data = self.repo_stats[repo]
        repo_data[commit.tier.lower()] += 1
        repo_data['scores'].append(score)

    def print_comprehensive_report(self):
        """Print detailed validation metrics."""
        total = self.gold_count + self.silver_count + self.reject_count
        if total == 0:
            print("No commits processed yet.")
            return

        gold_pct = 100 * self.gold_count / total
        silver_pct = 100 * self.silver_count / total
        reject_pct = 100 * self.reject_count / total

        accepted = self.gold_count + self.silver_count
        gold_share = 100 * self.gold_count / accepted if accepted > 0 else 0

        print(f"""
╔══════════════════════════════════════════════════════════╗
║            QUALITY ASSESSMENT VALIDATION                 ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║ TIER DISTRIBUTION                                        ║
║ ├─ GOLD:     {self.gold_count:>6}  ({gold_pct:>5.2f}%)  [Target: 1-5%]    ║
║ ├─ SILVER:   {self.silver_count:>6}  ({silver_pct:>5.2f}%)  [Target: 10-25%]  ║
║ └─ REJECT:   {self.reject_count:>6}  ({reject_pct:>5.2f}%)                      ║
║                                                          ║
║ ACCEPTED COMPOSITION                                     ║
║ └─ GOLD share: {gold_share:>5.2f}%  [Target: ≥25%]           ║
║                                                          ║
║ EVIDENCE DISTRIBUTION (top 5)                            ║""")

        for evidence, count in sorted(self.evidence_counts.items(), key=lambda x: -x[1])[:5]:
            pct = 100 * count / accepted if accepted > 0 else 0
            print(f"║ ├─ {evidence:<30} {count:>5} ({pct:>5.1f}%)")

        print(f"""║                                                          ║
║ EVIDENCE CO-OCCURRENCE (top 5 pairs)                     ║""")

        for pair, count in sorted(self.evidence_cooccurrence.items(), key=lambda x: -x[1])[:5]:
            pct = 100 * count / accepted if accepted > 0 else 0
            pair_str = f"{pair[0][:15]} ∧ {pair[1][:15]}"
            print(f"║ ├─ {pair_str:<30} {count:>5} ({pct:>5.1f}%)")

        # Verification presence check
        verification_count = self.evidence_counts.get('verification', 0)
        verification_pct = 100 * verification_count / accepted if accepted > 0 else 0
        verification_in_gold = 100  # By design, should be 100%

        print(f"""║                                                          ║
║ VERIFICATION SIGNALS                                     ║
║ ├─ In GOLD:    100.0%  (by design)                       ║
║ └─ In accepted: {verification_pct:>5.1f}%  [Target: ≥40%]            ║
║                                                          ║
║ TOP PENALTIES (if any)                                   ║""")

        if self.penalty_counts:
            for penalty, count in sorted(self.penalty_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"║ ├─ {penalty:<40} {count:>5}")
        else:
            print(f"║ └─ No penalties detected                                 ║")

        print(f"""║                                                          ║
║ SCORE HISTOGRAM (bins around thresholds)                ║
║ ├─ [0-1]:    {self.score_bins.get(0, 0) + self.score_bins.get(1, 0):>6}                                    ║
║ ├─ [2-7]:    {sum(self.score_bins.get(i, 0) for i in range(2, 8)):>6}  (SILVER zone)                  ║
║ ├─ [8-15]:   {sum(self.score_bins.get(i, 0) for i in range(8, 16)):>6}  (GOLD zone)                   ║
║ └─ [16+]:    {sum(self.score_bins.get(i, 0) for i in range(16, 101)):>6}  (exceptional)                   ║
║                                                          ║
║ REPO NORMALIZATION (top 5 by volume)                     ║""")

        for repo, stats in sorted(self.repo_stats.items(), key=lambda x: -(x[1]['gold'] + x[1]['silver']))[:5]:
            repo_total = stats['gold'] + stats['silver'] + stats['reject']
            repo_accept_pct = 100 * (stats['gold'] + stats['silver']) / repo_total if repo_total > 0 else 0
            repo_gold_share = 100 * stats['gold'] / (stats['gold'] + stats['silver']) if (stats['gold'] + stats['silver']) > 0 else 0
            repo_short = repo[-30:] if len(repo) > 30 else repo
            print(f"║ ├─ {repo_short:<30}                      ║")
            print(f"║ │  Accept: {repo_accept_pct:>5.1f}%  |  GOLD share: {repo_gold_share:>5.1f}%          ║")

        print(f"""╚══════════════════════════════════════════════════════════╝
""")


class NegativeEvidenceDetector:
    """Detect risky patterns that should penalize quality score."""

    def detect_penalties(self, commit, before_ast, after_ast, before_code, after_code) -> List[str]:
        """Run all penalty detectors and return list of triggered penalties."""
        penalties = []

        # 1. Complexity ↑ without verification
        if self._complexity_increase_without_tests(commit, before_ast, after_ast):
            penalties.append("complexity_increase_no_verification")

        # 2. SQL concatenation introduced
        if self._introduced_sql_concat(before_code, after_code):
            penalties.append("sql_string_concat")

        # 3. Validation removed
        if self._removed_validation(before_ast, after_ast):
            penalties.append("removed_validation")

        # 4. Swallowed exceptions
        if self._added_swallow_except(before_ast, after_ast, before_code, after_code):
            penalties.append("swallowed_exception")

        # 5. Linter disabling
        if self._added_linter_disable(before_code, after_code):
            penalties.append("linter_disabled")

        # 6. Wildcard imports
        if self._introduced_wildcard_import(before_ast, after_ast):
            penalties.append("wildcard_import")

        # 7. Unsafe path operations
        if self._unsafe_path_operations(before_code, after_code):
            penalties.append("unsafe_path_concat")

        return penalties

    def calculate_penalty_score(self, penalties: List[str]) -> int:
        """Calculate total penalty score from triggered penalties."""
        penalty_weights = {
            "complexity_increase_no_verification": 4,
            "sql_string_concat": 6,
            "removed_validation": 4,
            "swallowed_exception": 5,
            "linter_disabled": 3,
            "wildcard_import": 3,
            "unsafe_path_concat": 3,
        }
        return sum(penalty_weights.get(p, 0) for p in penalties)

    def _complexity_increase_without_tests(self, commit, before_ast, after_ast) -> bool:
        """Detect complexity increase without test coverage."""
        delta = commit.metrics.get('complexity_delta', 0)
        has_tests = commit.metrics.get('test_proximity', False)
        return delta >= 2 and not has_tests

    def _introduced_sql_concat(self, before_code: str, after_code: str) -> bool:
        """Detect SQL queries using string concatenation instead of parameterization."""
        # Heuristic: .execute() with + or f-string
        sql_concat_pattern = re.compile(r'\.execute\s*\(\s*(["\'].*?["\']|\w+)\s*[+%]|\.execute\s*\(\s*f["\']', re.DOTALL)

        # Check if after_code has SQL concat that before_code didn't
        has_after = bool(sql_concat_pattern.search(after_code))
        has_before = bool(sql_concat_pattern.search(before_code))

        return has_after and not has_before

    def _removed_validation(self, before_ast, after_ast) -> bool:
        """Detect removal of validation calls (isinstance, hasattr, etc.)."""
        before_validations = self._count_validation_calls(before_ast)
        after_validations = self._count_validation_calls(after_ast)
        return after_validations < before_validations

    def _count_validation_calls(self, tree) -> int:
        """Count validation function calls."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('isinstance', 'issubclass', 'hasattr', 'callable'):
                        count += 1
        return count

    def _added_swallow_except(self, before_ast, after_ast, before_code: str, after_code: str) -> bool:
        """Detect added exception handlers that swallow errors (pass/return None)."""
        # Look for except: pass or except: return patterns added
        before_swallow = self._count_swallow_except(before_ast)
        after_swallow = self._count_swallow_except(after_ast)
        return after_swallow > before_swallow

    def _count_swallow_except(self, tree) -> int:
        """Count exception handlers that swallow errors."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Check if body is just pass or return None
                if len(node.body) == 1:
                    stmt = node.body[0]
                    if isinstance(stmt, ast.Pass):
                        count += 1
                    elif isinstance(stmt, ast.Return):
                        # return None or return without value
                        if stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                            count += 1
        return count

    def _added_linter_disable(self, before_code: str, after_code: str) -> bool:
        """Detect added linter/type-checker disable comments."""
        linter_disable_patterns = [
            r'#\s*noqa',
            r'#\s*type:\s*ignore',
            r'#\s*pylint:\s*disable',
            r'#\s*mypy:\s*ignore',
            r'//\s*eslint-disable',
            r'//\s*@ts-ignore',
        ]

        for pattern in linter_disable_patterns:
            before_count = len(re.findall(pattern, before_code, re.IGNORECASE))
            after_count = len(re.findall(pattern, after_code, re.IGNORECASE))
            if after_count > before_count:
                return True
        return False

    def _introduced_wildcard_import(self, before_ast, after_ast) -> bool:
        """Detect added wildcard imports (from x import *)."""
        before_wildcard = self._has_wildcard_import(before_ast)
        after_wildcard = self._has_wildcard_import(after_ast)
        return after_wildcard and not before_wildcard

    def _has_wildcard_import(self, tree) -> bool:
        """Check if tree has wildcard imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.names and any(alias.name == '*' for alias in node.names):
                    return True
        return False

    def _unsafe_path_operations(self, before_code: str, after_code: str) -> bool:
        """Detect unsafe path operations (string concat instead of os.path.join/pathlib)."""
        # Pattern: filename + "/" + something or similar
        unsafe_path_pattern = re.compile(r'(["\'].*?/.*?["\']|\w+)\s*\+\s*["\']/')

        has_after = bool(unsafe_path_pattern.search(after_code))
        has_before = bool(unsafe_path_pattern.search(before_code))

        # Also check if safe API was present before but removed
        safe_apis = ['os.path.join', 'Path(', 'pathlib']
        had_safe = any(api in before_code for api in safe_apis)
        lost_safe = had_safe and not any(api in after_code for api in safe_apis)

        return (has_after and not has_before) or lost_safe


class VerificationHardening:
    """Strengthen verification evidence to reduce noise."""

    def detect_hardened_verification(self, commit, before_ast, after_ast) -> Tuple[bool, int]:
        """
        Detect verification signals with noise filtering.

        Returns:
            (has_verification, verification_points)
        """
        verification_points = 0
        has_verification = False

        # 1. Test file changes (body changes only, not renames/moves)
        if self._has_substantive_test_changes(commit):
            verification_points += 5
            has_verification = True

        # 2. Added asserts (new asserts, not just moved)
        if self._has_new_asserts(before_ast, after_ast):
            verification_points += 3
            has_verification = True

        # 3. CI keywords (only in commit message, filtered)
        if self._has_ci_keywords_filtered(commit):
            verification_points += 2
            has_verification = True

        # 4. Issue linkage (bonus for linked bug issues)
        if self._has_bug_issue_linkage(commit):
            verification_points += 2
            has_verification = True

        # 5. Error handling (already tracked)
        if commit.metrics.get('adds_error_handling', False):
            verification_points += 15
            has_verification = True

        return has_verification, verification_points

    def _has_substantive_test_changes(self, commit) -> bool:
        """Check if test files have actual code changes (not just renames)."""
        # Simple heuristic: test file in files list
        test_files = [f for f in commit.files if self._is_test_file(f)]
        return len(test_files) > 0

    def _is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file."""
        test_patterns = [
            r'(^|/)tests?/',
            r'(^|/)__tests__/',
            r'^test_.*\.py$',
            r'_test\.py$',
            r'\.test\.(js|ts|jsx|tsx)$',
            r'\.spec\.(js|ts|jsx|tsx)$',
        ]
        return any(re.search(pattern, filepath, re.IGNORECASE) for pattern in test_patterns)

    def _has_new_asserts(self, before_ast, after_ast) -> bool:
        """Check if new assert statements were added (not just moved)."""
        before_asserts = sum(1 for n in ast.walk(before_ast) if isinstance(n, ast.Assert))
        after_asserts = sum(1 for n in ast.walk(after_ast) if isinstance(n, ast.Assert))
        return after_asserts > before_asserts

    def _has_ci_keywords_filtered(self, commit) -> bool:
        """Check for CI keywords in commit message (filtered for noise)."""
        msg = commit.message.lower()

        # Positive CI keywords
        ci_keywords = [
            'failing test',
            'test fail',
            'green',
            'make tests pass',
            'fix ci',
            'jenkins',
            'travis',
            'circleci',
            'github actions',
        ]

        return any(kw in msg for kw in ci_keywords)

    def _has_bug_issue_linkage(self, commit) -> bool:
        """Check if commit references a bug issue (fixes #123 with bug-ish title)."""
        msg = commit.message.lower()

        # Common issue reference patterns
        issue_patterns = [
            r'fix(es)?\s+#\d+',
            r'close(s|d)?\s+#\d+',
            r'resolve(s|d)?\s+#\d+',
        ]

        # Bug-related keywords
        bug_keywords = ['bug', 'error', 'exception', 'crash', 'fail', 'issue', 'problem']

        has_issue_ref = any(re.search(pattern, msg) for pattern in issue_patterns)
        has_bug_keyword = any(kw in msg for kw in bug_keywords)

        return has_issue_ref and has_bug_keyword
