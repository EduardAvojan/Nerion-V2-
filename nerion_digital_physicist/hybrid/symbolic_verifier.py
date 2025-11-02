"""
Symbolic Verifier

Verifies code changes before they're applied to ensure they don't
violate symbolic rules or introduce regressions.
"""
from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .rule_engine import (
    SymbolicRuleEngine,
    RuleCheckResult,
    RuleCategory,
    RuleSeverity,
    RuleViolation
)


class VerificationStatus(Enum):
    """Status of code verification"""
    APPROVED = "approved"      # Change is safe and approved
    REJECTED = "rejected"      # Change violates critical rules
    CONDITIONAL = "conditional"  # Change acceptable with warnings
    MANUAL_REVIEW = "manual_review"  # Requires human review


@dataclass
class CodeChange:
    """Represents a code change"""
    before_code: str
    after_code: str
    description: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class VerificationResult:
    """Result of verifying a code change"""
    status: VerificationStatus
    approved: bool
    before_check: RuleCheckResult
    after_check: RuleCheckResult
    new_violations: List[RuleViolation]
    fixed_violations: List[RuleViolation]
    regression_score: float  # 0.0 (no regression) to 1.0 (severe regression)
    explanation: List[str]
    recommendation: str

    @property
    def safety_score(self) -> float:
        """
        Calculate safety score (0.0 to 1.0, higher is safer).

        Considers:
        - New violations introduced
        - Violations fixed
        - Severity of changes
        """
        # Start with perfect score
        score = 1.0

        # Penalize new violations (heavy penalties for critical issues)
        for violation in self.new_violations:
            if violation.severity == RuleSeverity.CRITICAL:
                score -= 0.6  # Critical violations are very dangerous
            elif violation.severity == RuleSeverity.ERROR:
                score -= 0.3
            elif violation.severity == RuleSeverity.WARNING:
                score -= 0.1

        # Reward fixed violations
        for violation in self.fixed_violations:
            if violation.severity == RuleSeverity.CRITICAL:
                score += 0.3
            elif violation.severity == RuleSeverity.ERROR:
                score += 0.2
            elif violation.severity == RuleSeverity.WARNING:
                score += 0.05

        return max(0.0, min(1.0, score))


class SymbolicVerifier:
    """
    Verifies code changes using symbolic rules.

    Ensures changes don't introduce new violations and preferably
    fix existing ones.

    Usage:
        >>> verifier = SymbolicVerifier()
        >>> change = CodeChange(
        ...     before_code="x = eval('1 + 1')",
        ...     after_code="x = 1 + 1"
        ... )
        >>> result = verifier.verify_change(change)
        >>> assert result.approved
    """

    def __init__(
        self,
        rule_engine: Optional[SymbolicRuleEngine] = None,
        strict_mode: bool = False
    ):
        """
        Initialize verifier.

        Args:
            rule_engine: Rule engine to use (creates default if None)
            strict_mode: If True, rejects changes with ANY new violations
        """
        self.rule_engine = rule_engine or SymbolicRuleEngine()
        self.strict_mode = strict_mode

    def verify_change(
        self,
        change: CodeChange,
        allow_warnings: bool = True,
        require_fixes: bool = False
    ) -> VerificationResult:
        """
        Verify a code change.

        Args:
            change: Code change to verify
            allow_warnings: Allow changes that introduce warnings
            require_fixes: Require change to fix at least one violation

        Returns:
            Verification result with approval status
        """
        # Check both versions
        before_check = self.rule_engine.check_code(
            change.before_code,
            context=change.context
        )
        after_check = self.rule_engine.check_code(
            change.after_code,
            context=change.context
        )

        # Identify new violations
        new_violations = self._find_new_violations(
            before_check,
            after_check
        )

        # Identify fixed violations
        fixed_violations = self._find_fixed_violations(
            before_check,
            after_check
        )

        # Calculate regression score
        regression_score = self._calculate_regression_score(
            new_violations,
            fixed_violations
        )

        # Determine status and approval
        status, approved, explanation = self._determine_status(
            before_check=before_check,
            after_check=after_check,
            new_violations=new_violations,
            fixed_violations=fixed_violations,
            regression_score=regression_score,
            allow_warnings=allow_warnings,
            require_fixes=require_fixes
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            status=status,
            new_violations=new_violations,
            fixed_violations=fixed_violations,
            regression_score=regression_score
        )

        return VerificationResult(
            status=status,
            approved=approved,
            before_check=before_check,
            after_check=after_check,
            new_violations=new_violations,
            fixed_violations=fixed_violations,
            regression_score=regression_score,
            explanation=explanation,
            recommendation=recommendation
        )

    def verify_batch(
        self,
        changes: List[CodeChange],
        stop_on_rejection: bool = True
    ) -> List[VerificationResult]:
        """
        Verify multiple code changes.

        Args:
            changes: List of changes to verify
            stop_on_rejection: Stop if any change is rejected

        Returns:
            List of verification results
        """
        results = []

        for change in changes:
            result = self.verify_change(change)
            results.append(result)

            if stop_on_rejection and result.status == VerificationStatus.REJECTED:
                break

        return results

    def _find_new_violations(
        self,
        before_check: RuleCheckResult,
        after_check: RuleCheckResult
    ) -> List[RuleViolation]:
        """Find violations introduced by the change"""
        # Get all violations from both checks
        before_violations = set(
            (v.rule_name, v.message, v.location)
            for v in (before_check.violations + before_check.warnings)
        )

        new_violations = []
        for v in (after_check.violations + after_check.warnings):
            violation_key = (v.rule_name, v.message, v.location)
            if violation_key not in before_violations:
                new_violations.append(v)

        return new_violations

    def _find_fixed_violations(
        self,
        before_check: RuleCheckResult,
        after_check: RuleCheckResult
    ) -> List[RuleViolation]:
        """Find violations fixed by the change"""
        # Get all violations from both checks
        after_violations = set(
            (v.rule_name, v.message, v.location)
            for v in (after_check.violations + after_check.warnings)
        )

        fixed_violations = []
        for v in (before_check.violations + before_check.warnings):
            violation_key = (v.rule_name, v.message, v.location)
            if violation_key not in after_violations:
                fixed_violations.append(v)

        return fixed_violations

    def _calculate_regression_score(
        self,
        new_violations: List[RuleViolation],
        fixed_violations: List[RuleViolation]
    ) -> float:
        """
        Calculate regression score (0.0 to 1.0).

        Higher score = more regression
        """
        if not new_violations:
            return 0.0

        # Weight violations by severity
        severity_weights = {
            RuleSeverity.CRITICAL: 1.0,
            RuleSeverity.ERROR: 0.7,
            RuleSeverity.WARNING: 0.3,
            RuleSeverity.INFO: 0.1
        }

        # Calculate weighted new violations
        new_score = sum(
            severity_weights[v.severity]
            for v in new_violations
        )

        # Calculate weighted fixed violations
        fixed_score = sum(
            severity_weights[v.severity]
            for v in fixed_violations
        )

        # Normalize to [0, 1]
        # More new violations = higher regression
        # More fixed violations = lower regression
        max_possible = max(new_score, 1.0)
        regression = (new_score - fixed_score) / max_possible

        return max(0.0, min(1.0, regression))

    def _determine_status(
        self,
        before_check: RuleCheckResult,
        after_check: RuleCheckResult,
        new_violations: List[RuleViolation],
        fixed_violations: List[RuleViolation],
        regression_score: float,
        allow_warnings: bool,
        require_fixes: bool
    ) -> Tuple[VerificationStatus, bool, List[str]]:
        """Determine verification status and approval"""
        explanation = []

        # Check for critical violations in after_code
        if after_check.has_critical_violations:
            explanation.append("REJECTED: After-code contains critical violations")
            explanation.append(f"  - {len([v for v in after_check.violations if v.severity == RuleSeverity.CRITICAL])} critical violations")
            return VerificationStatus.REJECTED, False, explanation

        # Check for new critical violations
        new_critical = [v for v in new_violations if v.severity == RuleSeverity.CRITICAL]
        if new_critical:
            explanation.append(f"REJECTED: Introduces {len(new_critical)} new critical violations")
            for v in new_critical:
                explanation.append(f"  - {v.message}")
            return VerificationStatus.REJECTED, False, explanation

        # Check for new errors
        new_errors = [v for v in new_violations if v.severity == RuleSeverity.ERROR]
        if new_errors:
            if self.strict_mode:
                explanation.append(f"REJECTED (strict mode): Introduces {len(new_errors)} new errors")
                return VerificationStatus.REJECTED, False, explanation
            else:
                explanation.append(f"WARNING: Introduces {len(new_errors)} new errors")

        # Check for new warnings
        new_warnings = [v for v in new_violations if v.severity == RuleSeverity.WARNING]
        if new_warnings and not allow_warnings:
            explanation.append(f"REJECTED: Introduces {len(new_warnings)} new warnings (warnings not allowed)")
            return VerificationStatus.REJECTED, False, explanation

        # Check if fixes are required
        if require_fixes and not fixed_violations:
            explanation.append("REJECTED: No violations fixed (fixes required)")
            return VerificationStatus.REJECTED, False, explanation

        # Determine final status
        if new_violations and not fixed_violations:
            # New violations but no fixes
            explanation.append(f"CONDITIONAL: {len(new_violations)} new violations, 0 fixes")
            return VerificationStatus.CONDITIONAL, True, explanation

        elif new_violations and fixed_violations:
            # Both new violations and fixes
            if len(fixed_violations) > len(new_violations):
                explanation.append(f"APPROVED: {len(fixed_violations)} fixes > {len(new_violations)} new violations")
                return VerificationStatus.APPROVED, True, explanation
            else:
                explanation.append(f"CONDITIONAL: {len(new_violations)} new violations, {len(fixed_violations)} fixes")
                return VerificationStatus.CONDITIONAL, True, explanation

        elif fixed_violations:
            # Only fixes, no new violations
            explanation.append(f"APPROVED: {len(fixed_violations)} violations fixed, 0 new violations")
            return VerificationStatus.APPROVED, True, explanation

        else:
            # No changes in violations
            explanation.append("APPROVED: No change in rule violations")
            return VerificationStatus.APPROVED, True, explanation

    def _generate_recommendation(
        self,
        status: VerificationStatus,
        new_violations: List[RuleViolation],
        fixed_violations: List[RuleViolation],
        regression_score: float
    ) -> str:
        """Generate recommendation based on verification result"""
        if status == VerificationStatus.REJECTED:
            if new_violations:
                critical = [v for v in new_violations if v.severity == RuleSeverity.CRITICAL]
                if critical:
                    return f"REJECT: Fix {len(critical)} critical violations before proceeding"
                else:
                    return f"REJECT: Fix {len(new_violations)} violations before proceeding"
            else:
                return "REJECT: Code does not pass symbolic verification"

        elif status == VerificationStatus.APPROVED:
            if fixed_violations:
                return f"APPROVE: Change fixes {len(fixed_violations)} violations and introduces no new issues"
            else:
                return "APPROVE: Change passes all symbolic checks"

        elif status == VerificationStatus.CONDITIONAL:
            if regression_score > 0.5:
                return f"REVIEW: Moderate regression detected (score: {regression_score:.2f}). Consider alternative approach."
            else:
                return f"APPROVE with caution: {len(new_violations)} new violations but acceptable"

        else:  # MANUAL_REVIEW
            return "Requires manual review by expert"

    def compare_code_quality(
        self,
        code1: str,
        code2: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare code quality between two versions.

        Args:
            code1: First version
            code2: Second version
            context: Optional context

        Returns:
            Comparison with quality metrics
        """
        check1 = self.rule_engine.check_code(code1, context)
        check2 = self.rule_engine.check_code(code2, context)

        return {
            'code1': {
                'passed': check1.passed,
                'total_violations': len(check1.violations),
                'critical': len([v for v in check1.violations if v.severity == RuleSeverity.CRITICAL]),
                'errors': len([v for v in check1.violations if v.severity == RuleSeverity.ERROR]),
                'warnings': len(check1.warnings),
            },
            'code2': {
                'passed': check2.passed,
                'total_violations': len(check2.violations),
                'critical': len([v for v in check2.violations if v.severity == RuleSeverity.CRITICAL]),
                'errors': len([v for v in check2.violations if v.severity == RuleSeverity.ERROR]),
                'warnings': len(check2.warnings),
            },
            'winner': 'code1' if len(check1.violations) < len(check2.violations) else 'code2',
            'improvement': abs(len(check1.violations) - len(check2.violations))
        }


# Example usage
if __name__ == "__main__":
    verifier = SymbolicVerifier()

    # Test 1: Fix security violation
    print("=== Test 1: Fixing Security Violation ===")
    change1 = CodeChange(
        before_code="x = eval('1 + 1')",
        after_code="x = 1 + 1",
        description="Replace eval with direct computation"
    )
    result1 = verifier.verify_change(change1)
    print(f"Status: {result1.status.value}")
    print(f"Approved: {result1.approved}")
    print(f"Safety Score: {result1.safety_score:.2f}")
    print(f"Fixed violations: {len(result1.fixed_violations)}")
    print(f"Recommendation: {result1.recommendation}")

    # Test 2: Introduce security violation
    print("\n=== Test 2: Introducing Security Violation ===")
    change2 = CodeChange(
        before_code="x = 1 + 1",
        after_code="x = eval('1 + 1')",
        description="Use eval (BAD)"
    )
    result2 = verifier.verify_change(change2)
    print(f"Status: {result2.status.value}")
    print(f"Approved: {result2.approved}")
    print(f"Safety Score: {result2.safety_score:.2f}")
    print(f"New violations: {len(result2.new_violations)}")
    print(f"Recommendation: {result2.recommendation}")

    # Test 3: Neutral change
    print("\n=== Test 3: Neutral Change ===")
    change3 = CodeChange(
        before_code="x = 1 + 1",
        after_code="x = 2",
        description="Simplify calculation"
    )
    result3 = verifier.verify_change(change3)
    print(f"Status: {result3.status.value}")
    print(f"Approved: {result3.approved}")
    print(f"Safety Score: {result3.safety_score:.2f}")
    print(f"Recommendation: {result3.recommendation}")
