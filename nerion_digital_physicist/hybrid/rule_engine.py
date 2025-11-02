"""
Symbolic Rule Engine

Defines and enforces symbolic rules for code quality, security, and correctness.
These rules provide hard constraints that neural predictions cannot violate.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Set
from abc import ABC, abstractmethod


class RuleCategory(Enum):
    """Categories of symbolic rules"""
    SYNTAX = "syntax"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BEST_PRACTICES = "best_practices"
    TYPE_SAFETY = "type_safety"
    CORRECTNESS = "correctness"
    MAINTAINABILITY = "maintainability"


class RuleSeverity(Enum):
    """Severity levels for rule violations"""
    CRITICAL = "critical"  # Must never be violated
    ERROR = "error"        # Should not be violated
    WARNING = "warning"    # Can be violated with justification
    INFO = "info"          # Informational only


@dataclass
class RuleViolation:
    """Represents a violation of a symbolic rule"""
    rule_name: str
    category: RuleCategory
    severity: RuleSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class RuleCheckResult:
    """Result of checking code against rules"""
    passed: bool
    violations: List[RuleViolation]
    warnings: List[RuleViolation]
    info: List[RuleViolation]

    @property
    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations"""
        return any(v.severity == RuleSeverity.CRITICAL for v in self.violations)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return any(v.severity == RuleSeverity.ERROR for v in self.violations)


class SymbolicRule(ABC):
    """
    Base class for symbolic rules.

    Each rule defines a constraint that code must satisfy.
    """

    def __init__(
        self,
        name: str,
        category: RuleCategory,
        severity: RuleSeverity,
        description: str
    ):
        self.name = name
        self.category = category
        self.severity = severity
        self.description = description

    @abstractmethod
    def check(self, code: str, context: Optional[Dict[str, Any]] = None) -> List[RuleViolation]:
        """
        Check if code violates this rule.

        Args:
            code: Source code to check
            context: Optional context (AST, metadata, etc.)

        Returns:
            List of violations (empty if code passes)
        """
        pass


class ASTRule(SymbolicRule):
    """Base class for rules that analyze AST"""

    def check(self, code: str, context: Optional[Dict[str, Any]] = None) -> List[RuleViolation]:
        """Check code using AST analysis"""
        try:
            tree = ast.parse(code)
            return self.check_ast(tree, code, context)
        except SyntaxError as e:
            # Syntax error - return violation
            return [RuleViolation(
                rule_name=self.name,
                category=self.category,
                severity=RuleSeverity.CRITICAL,
                message=f"Syntax error: {str(e)}",
                location=f"Line {e.lineno}" if hasattr(e, 'lineno') else None
            )]

    @abstractmethod
    def check_ast(
        self,
        tree: ast.AST,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[RuleViolation]:
        """Check AST for violations"""
        pass


class RegexRule(SymbolicRule):
    """Base class for rules using regex patterns"""

    def __init__(
        self,
        name: str,
        category: RuleCategory,
        severity: RuleSeverity,
        description: str,
        pattern: str,
        violation_message: str
    ):
        super().__init__(name, category, severity, description)
        self.pattern = re.compile(pattern)
        self.violation_message = violation_message

    def check(self, code: str, context: Optional[Dict[str, Any]] = None) -> List[RuleViolation]:
        """Check code using regex pattern"""
        violations = []

        for match in self.pattern.finditer(code):
            violations.append(RuleViolation(
                rule_name=self.name,
                category=self.category,
                severity=self.severity,
                message=self.violation_message,
                location=f"Position {match.start()}-{match.end()}",
                evidence=[match.group()]
            ))

        return violations


# ============================================================================
# Concrete Rule Implementations
# ============================================================================

class NoEvalRule(RegexRule):
    """Prohibits use of eval() - critical security risk"""

    def __init__(self):
        super().__init__(
            name="no-eval",
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.CRITICAL,
            description="Prohibits use of eval() function",
            pattern=r'\beval\s*\(',
            violation_message="Use of eval() is prohibited - critical security risk"
        )


class NoExecRule(RegexRule):
    """Prohibits use of exec() - critical security risk"""

    def __init__(self):
        super().__init__(
            name="no-exec",
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.CRITICAL,
            description="Prohibits use of exec() function",
            pattern=r'\bexec\s*\(',
            violation_message="Use of exec() is prohibited - critical security risk"
        )


class NoSQLInjectionRule(RegexRule):
    """Detects potential SQL injection vulnerabilities"""

    def __init__(self):
        super().__init__(
            name="no-sql-injection",
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.CRITICAL,
            description="Detects potential SQL injection",
            pattern=r'(execute|cursor\.execute)\s*\([^)]*%s[^)]*%',
            violation_message="Potential SQL injection - use parameterized queries"
        )


class NoHardcodedCredentialsRule(RegexRule):
    """Detects hardcoded credentials"""

    def __init__(self):
        super().__init__(
            name="no-hardcoded-credentials",
            category=RuleCategory.SECURITY,
            severity=RuleSeverity.ERROR,
            description="Detects hardcoded passwords/keys",
            pattern=r'(password|api_key|secret|token)\s*=\s*["\'][^"\']{8,}["\']',
            violation_message="Hardcoded credentials detected - use environment variables"
        )


class RequireExceptionHandlingRule(ASTRule):
    """Requires exception handling for risky operations"""

    def __init__(self):
        super().__init__(
            name="require-exception-handling",
            category=RuleCategory.CORRECTNESS,
            severity=RuleSeverity.WARNING,
            description="Requires try/except for file I/O and network operations"
        )

    def check_ast(
        self,
        tree: ast.AST,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[RuleViolation]:
        """Check for unhandled risky operations"""
        violations = []

        # Find all function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's a risky operation
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'urlopen', 'request']:
                        # Check if inside try block
                        if not self._is_in_try_block(node, tree):
                            violations.append(RuleViolation(
                                rule_name=self.name,
                                category=self.category,
                                severity=self.severity,
                                message=f"Risky operation '{node.func.id}' should be in try/except block",
                                location=f"Line {node.lineno}" if hasattr(node, 'lineno') else None,
                                suggestion="Wrap in try/except block"
                            ))

        return violations

    def _is_in_try_block(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is inside a try block"""
        # Simplified check - would need proper parent tracking
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Try):
                if any(node is child for child in ast.walk(parent)):
                    return True
        return False


class NoGlobalVariablesRule(ASTRule):
    """Prohibits use of global variables"""

    def __init__(self):
        super().__init__(
            name="no-global-variables",
            category=RuleCategory.BEST_PRACTICES,
            severity=RuleSeverity.WARNING,
            description="Prohibits use of global keyword"
        )

    def check_ast(
        self,
        tree: ast.AST,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[RuleViolation]:
        """Check for global statements"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                violations.append(RuleViolation(
                    rule_name=self.name,
                    category=self.category,
                    severity=self.severity,
                    message=f"Global variable(s) detected: {', '.join(node.names)}",
                    location=f"Line {node.lineno}" if hasattr(node, 'lineno') else None,
                    suggestion="Use function parameters or class attributes instead"
                ))

        return violations


class MaxComplexityRule(ASTRule):
    """Limits cyclomatic complexity"""

    def __init__(self, max_complexity: int = 5):
        super().__init__(
            name="max-complexity",
            category=RuleCategory.MAINTAINABILITY,
            severity=RuleSeverity.WARNING,
            description=f"Limits cyclomatic complexity to {max_complexity}"
        )
        self.max_complexity = max_complexity

    def check_ast(
        self,
        tree: ast.AST,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[RuleViolation]:
        """Check complexity of functions"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)

                if complexity > self.max_complexity:
                    violations.append(RuleViolation(
                        rule_name=self.name,
                        category=self.category,
                        severity=self.severity,
                        message=f"Function '{node.name}' has complexity {complexity} (max: {self.max_complexity})",
                        location=f"Line {node.lineno}" if hasattr(node, 'lineno') else None,
                        suggestion="Consider breaking function into smaller functions"
                    ))

        return violations

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Each decision point adds 1
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


class RequireTypeHintsRule(ASTRule):
    """Requires type hints for function signatures"""

    def __init__(self):
        super().__init__(
            name="require-type-hints",
            category=RuleCategory.TYPE_SAFETY,
            severity=RuleSeverity.INFO,
            description="Requires type hints for function arguments and return values"
        )

    def check_ast(
        self,
        tree: ast.AST,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[RuleViolation]:
        """Check for missing type hints"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check arguments
                missing_args = []
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        missing_args.append(arg.arg)

                # Check return type
                missing_return = node.returns is None

                if missing_args or missing_return:
                    message_parts = []
                    if missing_args:
                        message_parts.append(f"arguments: {', '.join(missing_args)}")
                    if missing_return:
                        message_parts.append("return type")

                    violations.append(RuleViolation(
                        rule_name=self.name,
                        category=self.category,
                        severity=self.severity,
                        message=f"Function '{node.name}' missing type hints for {' and '.join(message_parts)}",
                        location=f"Line {node.lineno}" if hasattr(node, 'lineno') else None,
                        suggestion="Add type hints for better type safety"
                    ))

        return violations


# ============================================================================
# Rule Engine
# ============================================================================

class SymbolicRuleEngine:
    """
    Symbolic rule engine that enforces hard constraints on code.

    Usage:
        >>> engine = SymbolicRuleEngine()
        >>> engine.add_rule(NoEvalRule())
        >>> result = engine.check_code("x = eval('1 + 1')")
        >>> assert not result.passed
    """

    def __init__(self):
        self.rules: Dict[str, SymbolicRule] = {}
        self.rule_categories: Dict[RuleCategory, List[str]] = {
            cat: [] for cat in RuleCategory
        }

        # Load default rules
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default security and quality rules"""
        default_rules = [
            # Security (CRITICAL)
            NoEvalRule(),
            NoExecRule(),
            NoSQLInjectionRule(),

            # Security (ERROR)
            NoHardcodedCredentialsRule(),

            # Correctness (WARNING)
            RequireExceptionHandlingRule(),

            # Best Practices (WARNING)
            NoGlobalVariablesRule(),
            MaxComplexityRule(max_complexity=5),

            # Type Safety (INFO)
            RequireTypeHintsRule(),
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: SymbolicRule):
        """Add a rule to the engine"""
        self.rules[rule.name] = rule
        self.rule_categories[rule.category].append(rule.name)

    def remove_rule(self, rule_name: str):
        """Remove a rule from the engine"""
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            del self.rules[rule_name]
            self.rule_categories[rule.category].remove(rule_name)

    def check_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        categories: Optional[List[RuleCategory]] = None,
        severity_threshold: RuleSeverity = RuleSeverity.INFO
    ) -> RuleCheckResult:
        """
        Check code against symbolic rules.

        Args:
            code: Source code to check
            context: Optional context for rules
            categories: Only check rules in these categories (None = all)
            severity_threshold: Minimum severity to report

        Returns:
            Result with violations
        """
        all_violations = []

        # Determine which rules to check
        rules_to_check = []
        if categories:
            for category in categories:
                for rule_name in self.rule_categories[category]:
                    rules_to_check.append(self.rules[rule_name])
        else:
            rules_to_check = list(self.rules.values())

        # Check each rule
        for rule in rules_to_check:
            violations = rule.check(code, context)
            all_violations.extend(violations)

        # Filter by severity threshold
        severity_order = {
            RuleSeverity.CRITICAL: 3,
            RuleSeverity.ERROR: 2,
            RuleSeverity.WARNING: 1,
            RuleSeverity.INFO: 0
        }

        threshold_level = severity_order[severity_threshold]
        filtered_violations = [
            v for v in all_violations
            if severity_order[v.severity] >= threshold_level
        ]

        # Categorize violations
        critical_and_errors = [
            v for v in filtered_violations
            if v.severity in [RuleSeverity.CRITICAL, RuleSeverity.ERROR]
        ]
        warnings = [
            v for v in filtered_violations
            if v.severity == RuleSeverity.WARNING
        ]
        info = [
            v for v in filtered_violations
            if v.severity == RuleSeverity.INFO
        ]

        # Code passes if no critical violations or errors
        passed = len(critical_and_errors) == 0

        return RuleCheckResult(
            passed=passed,
            violations=critical_and_errors,
            warnings=warnings,
            info=info
        )

    def get_rules_by_category(self, category: RuleCategory) -> List[SymbolicRule]:
        """Get all rules in a category"""
        return [
            self.rules[rule_name]
            for rule_name in self.rule_categories[category]
        ]

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded rules"""
        return {
            'total_rules': len(self.rules),
            'by_category': {
                cat.value: len(rules)
                for cat, rules in self.rule_categories.items()
            },
            'by_severity': {
                severity.value: sum(
                    1 for rule in self.rules.values()
                    if rule.severity == severity
                )
                for severity in RuleSeverity
            }
        }


# Example usage
if __name__ == "__main__":
    engine = SymbolicRuleEngine()

    # Test 1: Security violation (eval)
    print("=== Test 1: Security Violation ===")
    bad_code = """
x = eval('1 + 1')
y = 2
"""
    result = engine.check_code(bad_code)
    print(f"Passed: {result.passed}")
    print(f"Violations: {len(result.violations)}")
    for violation in result.violations:
        print(f"  - {violation.severity.value}: {violation.message}")

    # Test 2: Good code
    print("\n=== Test 2: Good Code ===")
    good_code = """
def add(x: int, y: int) -> int:
    return x + y
"""
    result = engine.check_code(good_code)
    print(f"Passed: {result.passed}")
    print(f"Violations: {len(result.violations)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Info: {len(result.info)}")

    # Test 3: Statistics
    print("\n=== Test 3: Engine Statistics ===")
    stats = engine.get_rule_statistics()
    print(f"Total rules: {stats['total_rules']}")
    print(f"By category: {stats['by_category']}")
    print(f"By severity: {stats['by_severity']}")
