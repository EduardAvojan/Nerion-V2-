"""
Integration tests for Neuro-Symbolic Hybrid Architecture

Tests the three major modules:
1. Symbolic Rule Engine
2. Symbolic Verifier
3. Neuro-Symbolic Reasoner
"""
import pytest

from nerion_digital_physicist.hybrid import (
    # Rule Engine
    SymbolicRuleEngine,
    RuleCategory,
    RuleSeverity,
    NoEvalRule,
    NoExecRule,
    NoSQLInjectionRule,
    MaxComplexityRule,
    RequireTypeHintsRule,
    # Symbolic Verifier
    SymbolicVerifier,
    CodeChange,
    VerificationStatus,
    # Neuro-Symbolic Reasoner
    NeuroSymbolicReasoner,
    ReasoningMode,
)


class TestSymbolicRuleEngine:
    """Test Symbolic Rule Engine functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.engine = SymbolicRuleEngine()

    def test_engine_creation(self):
        """Test rule engine creation with default rules"""
        assert self.engine is not None
        stats = self.engine.get_rule_statistics()
        assert stats['total_rules'] == 8  # Default rules loaded
        assert stats['by_severity'][RuleSeverity.CRITICAL.value] == 3
        assert stats['by_severity'][RuleSeverity.ERROR.value] == 1
        assert stats['by_severity'][RuleSeverity.WARNING.value] == 3
        assert stats['by_severity'][RuleSeverity.INFO.value] == 1

    def test_detect_eval_violation(self):
        """Test detection of eval() usage"""
        bad_code = "x = eval('1 + 1')"
        result = self.engine.check_code(bad_code)

        assert not result.passed
        assert len(result.violations) >= 1
        assert any('eval' in v.message.lower() for v in result.violations)
        assert result.has_critical_violations

    def test_detect_exec_violation(self):
        """Test detection of exec() usage"""
        bad_code = "exec('print(1)')"
        result = self.engine.check_code(bad_code)

        assert not result.passed
        assert any('exec' in v.message.lower() for v in result.violations)

    def test_detect_sql_injection(self):
        """Test detection of SQL injection vulnerability"""
        bad_code = "cursor.execute('SELECT * FROM users WHERE id = %s' % user_id)"
        result = self.engine.check_code(bad_code)

        assert not result.passed
        assert any('sql' in v.message.lower() for v in result.violations)

    def test_detect_hardcoded_credentials(self):
        """Test detection of hardcoded credentials"""
        bad_code = "password = 'my_secret_password_12345'"
        result = self.engine.check_code(bad_code)

        assert not result.passed
        assert any('credential' in v.message.lower() or 'password' in v.message.lower() for v in result.violations)

    def test_good_code_passes(self):
        """Test that good code passes all checks"""
        good_code = """
def add(x: int, y: int) -> int:
    return x + y
"""
        result = self.engine.check_code(good_code)

        assert result.passed  # No critical/error violations
        assert not result.has_critical_violations
        assert not result.has_errors

    def test_detect_missing_type_hints(self):
        """Test detection of missing type hints"""
        code_without_hints = """
def process(data):
    return data * 2
"""
        result = self.engine.check_code(code_without_hints)

        # Should have info-level violations for missing type hints
        assert len(result.info) > 0
        assert any('type hint' in v.message.lower() for v in result.info)

    def test_detect_high_complexity(self):
        """Test detection of high cyclomatic complexity"""
        complex_code = """
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                if x > y:
                    if y > z:
                        if x > z:
                            return 1
                        else:
                            return 2
                    else:
                        return 3
                else:
                    return 4
            else:
                return 5
        else:
            return 6
    else:
        return 7
"""
        result = self.engine.check_code(complex_code)

        # Should have warning about complexity
        assert len(result.warnings) > 0
        assert any('complexity' in v.message.lower() for v in result.warnings)

    def test_add_custom_rule(self):
        """Test adding custom rule"""
        initial_count = self.engine.get_rule_statistics()['total_rules']

        custom_rule = NoEvalRule()  # Use existing rule as example
        custom_rule.name = "custom-no-eval"
        self.engine.add_rule(custom_rule)

        new_count = self.engine.get_rule_statistics()['total_rules']
        assert new_count == initial_count + 1

    def test_remove_rule(self):
        """Test removing rule"""
        initial_count = self.engine.get_rule_statistics()['total_rules']

        self.engine.remove_rule("no-eval")

        new_count = self.engine.get_rule_statistics()['total_rules']
        assert new_count == initial_count - 1

    def test_check_by_category(self):
        """Test checking code with specific categories only"""
        bad_code = "x = eval('1 + 1')"

        # Check only security rules
        result = self.engine.check_code(
            bad_code,
            categories=[RuleCategory.SECURITY]
        )

        assert not result.passed
        assert all(v.category == RuleCategory.SECURITY for v in result.violations)

    def test_severity_threshold(self):
        """Test checking with severity threshold"""
        code = """
global_var = 0

def func():
    global global_var
    global_var += 1
"""
        # Check with WARNING threshold - should include warnings
        result1 = self.engine.check_code(code, severity_threshold=RuleSeverity.WARNING)
        warning_count1 = len(result1.warnings)

        # Check with ERROR threshold - should exclude warnings
        result2 = self.engine.check_code(code, severity_threshold=RuleSeverity.ERROR)
        warning_count2 = len(result2.warnings)

        assert warning_count1 >= warning_count2


class TestSymbolicVerifier:
    """Test Symbolic Verifier functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.verifier = SymbolicVerifier()

    def test_verifier_creation(self):
        """Test verifier creation"""
        assert self.verifier is not None
        assert self.verifier.rule_engine is not None

    def test_approve_fix(self):
        """Test approval of code that fixes violations"""
        change = CodeChange(
            before_code="x = eval('1 + 1')",
            after_code="x = 1 + 1",
            description="Remove eval"
        )

        result = self.verifier.verify_change(change)

        assert result.approved
        assert result.status == VerificationStatus.APPROVED
        assert len(result.fixed_violations) > 0
        assert len(result.new_violations) == 0
        assert result.safety_score > 0.8

    def test_reject_regression(self):
        """Test rejection of code that introduces violations"""
        change = CodeChange(
            before_code="x = 1 + 1",
            after_code="x = eval('1 + 1')",
            description="Add eval (BAD)"
        )

        result = self.verifier.verify_change(change)

        assert not result.approved
        assert result.status == VerificationStatus.REJECTED
        assert len(result.new_violations) > 0
        assert len(result.fixed_violations) == 0
        assert result.safety_score < 0.5

    def test_approve_neutral_change(self):
        """Test approval of neutral change (no violation changes)"""
        change = CodeChange(
            before_code="x = 1 + 1",
            after_code="x = 2",
            description="Simplify"
        )

        result = self.verifier.verify_change(change)

        assert result.approved
        assert result.status == VerificationStatus.APPROVED
        assert len(result.new_violations) == 0
        assert len(result.fixed_violations) == 0

    def test_conditional_approval(self):
        """Test conditional approval (warnings but not critical)"""
        change = CodeChange(
            before_code="def func(): return 42",
            after_code="global x\ndef func(): global x; x = 42",
            description="Add global variable"
        )

        result = self.verifier.verify_change(change, allow_warnings=True)

        # Should be approved with warnings (global is WARNING, not CRITICAL)
        assert result.approved or result.status == VerificationStatus.CONDITIONAL

    def test_strict_mode(self):
        """Test strict mode rejects ANY new violations"""
        verifier_strict = SymbolicVerifier(strict_mode=True)

        change = CodeChange(
            before_code="def func(): return 42",
            after_code="global x\ndef func(): global x; x = 42"
        )

        result = verifier_strict.verify_change(change)

        # Strict mode should reject even warnings
        assert not result.approved or result.status != VerificationStatus.APPROVED

    def test_regression_score_calculation(self):
        """Test regression score calculation"""
        # No regression
        change1 = CodeChange(
            before_code="x = 1",
            after_code="x = 2"
        )
        result1 = self.verifier.verify_change(change1)
        assert result1.regression_score == 0.0

        # High regression
        change2 = CodeChange(
            before_code="x = 1",
            after_code="x = eval('1')"
        )
        result2 = self.verifier.verify_change(change2)
        assert result2.regression_score > 0.5

    def test_batch_verification(self):
        """Test verifying multiple changes"""
        changes = [
            CodeChange(before_code="x = 1", after_code="x = 2"),
            CodeChange(before_code="y = 2", after_code="y = 3"),
            CodeChange(before_code="z = 3", after_code="z = eval('3')")  # Bad
        ]

        results = self.verifier.verify_batch(changes, stop_on_rejection=True)

        # Should stop at the bad change
        assert len(results) == 3
        assert results[0].approved
        assert results[1].approved
        assert not results[2].approved

    def test_compare_code_quality(self):
        """Test comparing code quality"""
        code1 = "x = eval('1 + 1')"
        code2 = "x = 1 + 1"

        comparison = self.verifier.compare_code_quality(code1, code2)

        assert comparison['winner'] == 'code2'
        assert comparison['code1']['total_violations'] > comparison['code2']['total_violations']


class TestNeuroSymbolicReasoner:
    """Test Neuro-Symbolic Reasoner functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.reasoner = NeuroSymbolicReasoner(
            gnn_model=None,  # No GNN in tests
            neural_weight=0.6,
            symbolic_weight=0.4
        )

    def test_reasoner_creation(self):
        """Test reasoner creation"""
        assert self.reasoner is not None
        assert self.reasoner.rule_engine is not None
        assert self.reasoner.symbolic_verifier is not None
        assert self.reasoner.neural_weight == 0.6
        assert self.reasoner.symbolic_weight == 0.4

    def test_reason_good_code(self):
        """Test reasoning about good code"""
        good_code = "def add(x: int, y: int) -> int: return x + y"

        decision = self.reasoner.reason(good_code, mode=ReasoningMode.SYMBOLIC_ONLY)

        assert decision.approved
        assert decision.confidence > 0.5
        assert not decision.symbolic_analysis.check_result.has_critical_violations

    def test_reason_bad_code(self):
        """Test reasoning about bad code"""
        bad_code = "x = eval('1 + 1')"

        decision = self.reasoner.reason(bad_code, mode=ReasoningMode.SYMBOLIC_ONLY)

        assert not decision.approved
        assert decision.symbolic_analysis.check_result.has_critical_violations

    def test_reason_on_change_approve(self):
        """Test reasoning on code change that should be approved"""
        before = "x = eval('1 + 1')"
        after = "x = 1 + 1"

        decision, verification = self.reasoner.reason_on_change(
            before,
            after,
            mode=ReasoningMode.NEURAL_THEN_SYMBOLIC
        )

        assert decision.approved
        assert verification.approved
        assert decision.prediction == 1  # Prefer after

    def test_reason_on_change_reject(self):
        """Test reasoning on code change that should be rejected"""
        before = "x = 1 + 1"
        after = "x = eval('1 + 1')"

        decision, verification = self.reasoner.reason_on_change(
            before,
            after,
            mode=ReasoningMode.NEURAL_THEN_SYMBOLIC
        )

        assert not decision.approved or not verification.approved
        assert len(verification.new_violations) > 0

    def test_neural_only_mode(self):
        """Test neural-only reasoning mode"""
        code = "def test(): return 42"

        decision = self.reasoner.reason(code, mode=ReasoningMode.NEURAL_ONLY)

        # Should have neural prediction but no symbolic verification
        assert decision.neural_prediction is not None
        assert decision.reasoning_mode == ReasoningMode.NEURAL_ONLY

    def test_symbolic_only_mode(self):
        """Test symbolic-only reasoning mode"""
        code = "x = eval('1')"

        decision = self.reasoner.reason(code, mode=ReasoningMode.SYMBOLIC_ONLY)

        # Should use only symbolic analysis
        assert decision.reasoning_mode == ReasoningMode.SYMBOLIC_ONLY
        assert not decision.approved  # eval is critical violation

    def test_hybrid_mode(self):
        """Test hybrid reasoning mode"""
        code = "def add(x, y): return x + y"  # No type hints (info-level)

        decision = self.reasoner.reason(code, mode=ReasoningMode.HYBRID)

        # Should combine neural and symbolic
        assert decision.reasoning_mode == ReasoningMode.HYBRID
        assert 'neural' in decision.confidence_breakdown
        assert 'symbolic' in decision.confidence_breakdown
        assert 'combined' in decision.confidence_breakdown

    def test_trustworthy_decision(self):
        """Test trustworthy decision detection"""
        good_code = "def multiply(x: int, y: int) -> int: return x * y"

        decision = self.reasoner.reason(good_code, mode=ReasoningMode.HYBRID)

        # Good code should produce trustworthy decision
        # (high confidence + no violations + approval)
        assert decision.safety_score > 0.5

    def test_safety_score_calculation(self):
        """Test safety score calculation"""
        # Good code = high safety
        decision1 = self.reasoner.reason(
            "def test(): return 42",
            mode=ReasoningMode.SYMBOLIC_ONLY
        )
        safety1 = decision1.safety_score

        # Bad code = low safety
        decision2 = self.reasoner.reason(
            "x = eval('1')",
            mode=ReasoningMode.SYMBOLIC_ONLY
        )
        safety2 = decision2.safety_score

        assert safety1 > safety2

    def test_explanation_generation(self):
        """Test explanation generation"""
        code = "x = eval('1 + 1')"

        decision = self.reasoner.reason(code, mode=ReasoningMode.HYBRID)

        assert len(decision.explanation) > 0
        assert any(isinstance(line, str) for line in decision.explanation)

    def test_reasoner_statistics(self):
        """Test getting reasoner statistics"""
        stats = self.reasoner.get_statistics()

        assert 'has_gnn' in stats
        assert 'neural_weight' in stats
        assert 'symbolic_weight' in stats
        assert 'rule_engine' in stats
        assert stats['neural_weight'] == 0.6
        assert stats['symbolic_weight'] == 0.4


class TestIntegration:
    """Test integration between components"""

    def test_rule_engine_to_verifier(self):
        """Test rule engine integration with verifier"""
        engine = SymbolicRuleEngine()
        verifier = SymbolicVerifier(rule_engine=engine)

        change = CodeChange(
            before_code="x = eval('1')",
            after_code="x = 1"
        )

        result = verifier.verify_change(change)

        assert result.approved
        assert len(result.fixed_violations) > 0

    def test_verifier_to_reasoner(self):
        """Test verifier integration with reasoner"""
        verifier = SymbolicVerifier()
        reasoner = NeuroSymbolicReasoner(symbolic_verifier=verifier)

        before = "x = eval('1')"
        after = "x = 1"

        decision, verification = reasoner.reason_on_change(before, after)

        assert decision.approved
        assert verification.approved

    def test_end_to_end_workflow(self):
        """Test complete neuro-symbolic workflow"""
        # Create components
        engine = SymbolicRuleEngine()
        verifier = SymbolicVerifier(rule_engine=engine)
        reasoner = NeuroSymbolicReasoner(
            rule_engine=engine,
            symbolic_verifier=verifier
        )

        # Test code with security issue
        bad_code = "password = 'hardcoded123'; user_input = eval(request.data)"
        good_code = "password = os.getenv('PASSWORD'); user_input = json.loads(request.data)"

        # Reason about the change
        decision, verification = reasoner.reason_on_change(
            before_code=bad_code,
            after_code=good_code,
            mode=ReasoningMode.HYBRID
        )

        # Should approve the fix
        assert decision.approved
        assert verification.status in [VerificationStatus.APPROVED, VerificationStatus.CONDITIONAL]
        assert len(verification.fixed_violations) > 0
        assert decision.safety_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
