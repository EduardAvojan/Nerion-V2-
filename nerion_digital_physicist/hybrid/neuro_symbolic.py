"""
Neuro-Symbolic Reasoner

Combines neural pattern matching (GNN) with symbolic reasoning (rules)
for explainable, trustworthy code quality decisions.

Architecture:
1. Neural layer: GNN predicts code quality (pattern recognition)
2. Symbolic layer: Rule engine enforces hard constraints
3. Integration layer: Combines predictions with rule compliance
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .rule_engine import (
    SymbolicRuleEngine,
    RuleCheckResult,
    RuleSeverity,
    RuleViolation
)
from .symbolic_verifier import SymbolicVerifier, CodeChange, VerificationResult


class ReasoningMode(Enum):
    """Reasoning modes for neuro-symbolic integration"""
    NEURAL_ONLY = "neural_only"          # Use only neural predictions
    SYMBOLIC_ONLY = "symbolic_only"      # Use only symbolic rules
    NEURAL_THEN_SYMBOLIC = "neural_then_symbolic"  # Neural predicts, symbolic verifies
    SYMBOLIC_THEN_NEURAL = "symbolic_then_neural"  # Symbolic filters, neural ranks
    HYBRID = "hybrid"                    # Combine both with weights


@dataclass
class NeuralPrediction:
    """Neural network prediction"""
    prediction: int  # 0 = before_code, 1 = after_code
    confidence: float  # 0.0 to 1.0
    logits: Optional[torch.Tensor] = None
    attention_weights: Optional[Dict[str, float]] = None
    embedding: Optional[torch.Tensor] = None


@dataclass
class SymbolicAnalysis:
    """Symbolic rule analysis"""
    check_result: RuleCheckResult
    passed: bool
    violation_count: int
    critical_violations: int
    rule_confidence: float  # Based on rule violation severity


@dataclass
class NeuroSymbolicDecision:
    """
    Combined neuro-symbolic decision.

    Integrates neural predictions with symbolic verification
    for explainable, trustworthy decisions.
    """
    # Final decision
    prediction: int  # 0 or 1
    confidence: float  # 0.0 to 1.0
    approved: bool  # Whether decision is approved by symbolic layer

    # Neural component
    neural_prediction: NeuralPrediction

    # Symbolic component
    symbolic_analysis: SymbolicAnalysis

    # Integration
    reasoning_mode: ReasoningMode
    neural_weight: float
    symbolic_weight: float

    # Explanation
    explanation: List[str]
    rule_violations: List[RuleViolation]
    confidence_breakdown: Dict[str, float]

    @property
    def is_trustworthy(self) -> bool:
        """
        Check if decision is trustworthy.

        Trustworthy = high confidence + no critical violations + neural-symbolic agreement
        """
        neural_symbolic_agree = (
            self.neural_prediction.prediction == (1 if self.symbolic_analysis.passed else 0)
        )

        return (
            self.confidence > 0.7 and
            self.symbolic_analysis.critical_violations == 0 and
            neural_symbolic_agree
        )

    @property
    def safety_score(self) -> float:
        """
        Calculate overall safety score (0.0 to 1.0).

        Combines confidence with rule compliance.
        """
        # Start with confidence
        score = self.confidence

        # Penalize for rule violations
        if self.symbolic_analysis.critical_violations > 0:
            score *= 0.3  # Critical violations drastically reduce safety
        elif self.symbolic_analysis.violation_count > 0:
            score *= 0.7  # Other violations moderately reduce safety

        # Reward approval
        if self.approved:
            score *= 1.1

        return min(1.0, score)


class NeuroSymbolicReasoner:
    """
    Neuro-Symbolic reasoning system.

    Combines GNN (neural) with rule engine (symbolic) for
    explainable, trustworthy code quality decisions.

    Usage:
        >>> reasoner = NeuroSymbolicReasoner(gnn_model, rule_engine)
        >>> decision = reasoner.reason(code, mode=ReasoningMode.HYBRID)
        >>> print(decision.prediction, decision.confidence, decision.approved)
    """

    def __init__(
        self,
        gnn_model: Optional[Any] = None,  # GNN model from brain.py
        rule_engine: Optional[SymbolicRuleEngine] = None,
        symbolic_verifier: Optional[SymbolicVerifier] = None,
        neural_weight: float = 0.6,
        symbolic_weight: float = 0.4,
        device: str = 'cpu'
    ):
        """
        Initialize neuro-symbolic reasoner.

        Args:
            gnn_model: Pre-trained GNN model (optional)
            rule_engine: Symbolic rule engine (creates default if None)
            symbolic_verifier: Symbolic verifier (creates default if None)
            neural_weight: Weight for neural predictions in hybrid mode
            symbolic_weight: Weight for symbolic analysis in hybrid mode
            device: Device for neural computation
        """
        self.gnn_model = gnn_model
        self.rule_engine = rule_engine or SymbolicRuleEngine()
        self.symbolic_verifier = symbolic_verifier or SymbolicVerifier(self.rule_engine)
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.device = device

        # Normalize weights
        total = self.neural_weight + self.symbolic_weight
        if total > 0:
            self.neural_weight /= total
            self.symbolic_weight /= total

    def reason(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        mode: ReasoningMode = ReasoningMode.HYBRID
    ) -> NeuroSymbolicDecision:
        """
        Make a neuro-symbolic decision about code quality.

        Args:
            code: Source code to analyze
            context: Optional context (graph, metadata)
            mode: Reasoning mode

        Returns:
            Neuro-symbolic decision with explanation
        """
        # Get neural prediction
        neural_pred = self._get_neural_prediction(code, context)

        # Get symbolic analysis
        symbolic_analysis = self._get_symbolic_analysis(code, context)

        # Integrate predictions based on mode
        decision = self._integrate_predictions(
            neural_pred=neural_pred,
            symbolic_analysis=symbolic_analysis,
            mode=mode
        )

        return decision

    def reason_on_change(
        self,
        before_code: str,
        after_code: str,
        context: Optional[Dict[str, Any]] = None,
        mode: ReasoningMode = ReasoningMode.NEURAL_THEN_SYMBOLIC
    ) -> Tuple[NeuroSymbolicDecision, VerificationResult]:
        """
        Reason about a code change.

        Args:
            before_code: Code before change
            after_code: Code after change
            context: Optional context
            mode: Reasoning mode

        Returns:
            (Decision, Verification result)
        """
        # Get neural predictions for both versions
        neural_before = self._get_neural_prediction(before_code, context)
        neural_after = self._get_neural_prediction(after_code, context)

        # Verify change symbolically
        change = CodeChange(
            before_code=before_code,
            after_code=after_code,
            context=context
        )
        verification = self.symbolic_verifier.verify_change(change)

        # Combine neural preference with symbolic verification
        # Neural prefers after_code if confidence is higher
        neural_prefers_after = neural_after.confidence > neural_before.confidence

        # Symbolic approves if verification passes
        symbolic_approves = verification.approved

        # Pre-compute scores for all modes
        neural_score = neural_after.confidence if neural_prefers_after else neural_before.confidence
        symbolic_score = verification.safety_score

        # Final decision
        if mode == ReasoningMode.NEURAL_THEN_SYMBOLIC:
            # Neural decides, symbolic vetos
            prediction = 1 if neural_prefers_after else 0
            approved = symbolic_approves if neural_prefers_after else True
            confidence = neural_after.confidence if neural_prefers_after else neural_before.confidence

            # Override if symbolic rejects
            if not symbolic_approves and neural_prefers_after:
                approved = False
                confidence *= 0.5  # Reduce confidence due to symbolic rejection

        elif mode == ReasoningMode.SYMBOLIC_THEN_NEURAL:
            # Symbolic filters, neural ranks
            if not symbolic_approves:
                # Symbolic rejects after_code
                prediction = 0
                approved = False
                confidence = 1.0 - verification.regression_score
            else:
                # Symbolic approves, use neural ranking
                prediction = 1 if neural_prefers_after else 0
                approved = True
                confidence = neural_after.confidence if neural_prefers_after else neural_before.confidence

        else:  # HYBRID
            # Weighted combination
            combined_score = (
                self.neural_weight * neural_score +
                self.symbolic_weight * symbolic_score
            )

            prediction = 1 if neural_prefers_after and symbolic_approves else 0
            approved = symbolic_approves
            confidence = combined_score

        # Get symbolic analysis for chosen code
        chosen_code = after_code if prediction == 1 else before_code
        symbolic_analysis = self._get_symbolic_analysis(chosen_code, context)

        # Create decision
        chosen_neural = neural_after if prediction == 1 else neural_before

        explanation = [
            f"Neural prediction: {'after' if neural_prefers_after else 'before'} (conf: {neural_after.confidence:.2f} vs {neural_before.confidence:.2f})",
            f"Symbolic verification: {verification.status.value}",
            f"Final decision: {'after' if prediction == 1 else 'before'} (approved: {approved})"
        ]

        if not symbolic_approves:
            explanation.append(f"Symbolic rejection: {verification.recommendation}")

        decision = NeuroSymbolicDecision(
            prediction=prediction,
            confidence=confidence,
            approved=approved,
            neural_prediction=chosen_neural,
            symbolic_analysis=symbolic_analysis,
            reasoning_mode=mode,
            neural_weight=self.neural_weight,
            symbolic_weight=self.symbolic_weight,
            explanation=explanation,
            rule_violations=verification.new_violations,
            confidence_breakdown={
                'neural': chosen_neural.confidence,
                'symbolic': symbolic_score,
                'combined': confidence
            }
        )

        return decision, verification

    def _get_neural_prediction(
        self,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> NeuralPrediction:
        """Get prediction from neural network (GNN)"""
        if self.gnn_model is None:
            # No model available - use heuristic-based quality score
            # This is a fallback when GNN isn't loaded
            quality_score = self._heuristic_quality_score(code)
            return NeuralPrediction(
                prediction=1 if quality_score > 0.5 else 0,
                confidence=quality_score,
                logits=None,
                attention_weights=None,
                embedding=None
            )

        # In practice, would:
        # 1. Convert code to graph
        # 2. Pass through GNN
        # 3. Get prediction + confidence

        # Placeholder for actual GNN inference
        # This would integrate with nerion_digital_physicist/agent/brain.py
        return NeuralPrediction(
            prediction=1,
            confidence=0.8,
            logits=None,
            attention_weights=None,
            embedding=None
        )

    def _heuristic_quality_score(self, code: str) -> float:
        """
        Calculate heuristic-based quality score for code.
        Used as fallback when GNN is not available.

        Returns:
            Quality score between 0.0 (very bad) and 1.0 (very good)
        """
        score = 0.7  # Start with neutral-positive score

        # Negative indicators (reduce score)
        bad_patterns = [
            ('eval(', 0.3),
            ('exec(', 0.3),
            ("password = '", 0.2),
            ("password='", 0.2),
            ('% user_id', 0.15),  # SQL injection pattern
            ('global ', 0.05),
        ]

        for pattern, penalty in bad_patterns:
            if pattern in code:
                score -= penalty

        # Positive indicators (increase score)
        good_patterns = [
            ('->', 0.05),  # Type hints
            ('def ', 0.05),  # Function definition
            ('os.getenv', 0.1),  # Environment variables
            ('json.loads', 0.1),  # Safe parsing
        ]

        for pattern, bonus in good_patterns:
            if pattern in code:
                score += bonus

        return max(0.0, min(1.0, score))

    def _get_symbolic_analysis(
        self,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> SymbolicAnalysis:
        """Get analysis from symbolic rule engine"""
        check_result = self.rule_engine.check_code(code, context)

        # Count violations
        violation_count = len(check_result.violations) + len(check_result.warnings)
        critical_violations = len([
            v for v in check_result.violations
            if v.severity == RuleSeverity.CRITICAL
        ])

        # Calculate rule confidence based on violations
        if critical_violations > 0:
            rule_confidence = 0.0  # Critical violations = no confidence
        elif violation_count == 0:
            rule_confidence = 1.0  # No violations = full confidence
        else:
            # Gradual reduction based on violation severity
            penalty = sum(
                0.3 if v.severity == RuleSeverity.ERROR else 0.1
                for v in (check_result.violations + check_result.warnings)
            )
            rule_confidence = max(0.0, 1.0 - penalty)

        return SymbolicAnalysis(
            check_result=check_result,
            passed=check_result.passed,
            violation_count=violation_count,
            critical_violations=critical_violations,
            rule_confidence=rule_confidence
        )

    def _integrate_predictions(
        self,
        neural_pred: NeuralPrediction,
        symbolic_analysis: SymbolicAnalysis,
        mode: ReasoningMode
    ) -> NeuroSymbolicDecision:
        """Integrate neural and symbolic predictions"""
        if mode == ReasoningMode.NEURAL_ONLY:
            # Use only neural prediction
            prediction = neural_pred.prediction
            confidence = neural_pred.confidence
            approved = True  # No symbolic verification
            explanation = [f"Neural-only mode: prediction={prediction}, confidence={confidence:.2f}"]

        elif mode == ReasoningMode.SYMBOLIC_ONLY:
            # Use only symbolic rules
            prediction = 1 if symbolic_analysis.passed else 0
            confidence = symbolic_analysis.rule_confidence
            approved = symbolic_analysis.passed
            explanation = [
                f"Symbolic-only mode: passed={symbolic_analysis.passed}",
                f"Violations: {symbolic_analysis.violation_count} (critical: {symbolic_analysis.critical_violations})"
            ]

        elif mode == ReasoningMode.NEURAL_THEN_SYMBOLIC:
            # Neural decides, symbolic verifies
            prediction = neural_pred.prediction
            confidence = neural_pred.confidence

            # Symbolic veto for critical violations
            if symbolic_analysis.critical_violations > 0:
                approved = False
                confidence *= 0.3  # Drastically reduce confidence
                explanation = [
                    f"Neural prediction: {prediction} (conf: {neural_pred.confidence:.2f})",
                    f"Symbolic VETO: {symbolic_analysis.critical_violations} critical violations",
                    f"Final confidence reduced: {confidence:.2f}"
                ]
            else:
                approved = symbolic_analysis.passed
                if not approved:
                    confidence *= 0.7  # Moderately reduce confidence
                explanation = [
                    f"Neural prediction: {prediction} (conf: {neural_pred.confidence:.2f})",
                    f"Symbolic check: {'PASS' if approved else 'FAIL'} ({symbolic_analysis.violation_count} violations)"
                ]

        elif mode == ReasoningMode.SYMBOLIC_THEN_NEURAL:
            # Symbolic filters, neural ranks
            if not symbolic_analysis.passed:
                # Symbolic rejects
                prediction = 0
                confidence = 0.0
                approved = False
                explanation = [
                    f"Symbolic filter: REJECT ({symbolic_analysis.violation_count} violations)",
                    "Neural prediction not consulted"
                ]
            else:
                # Symbolic approves, use neural
                prediction = neural_pred.prediction
                confidence = neural_pred.confidence
                approved = True
                explanation = [
                    "Symbolic filter: PASS",
                    f"Neural prediction: {prediction} (conf: {confidence:.2f})"
                ]

        else:  # HYBRID
            # Weighted combination
            neural_score = neural_pred.confidence if neural_pred.prediction == 1 else (1.0 - neural_pred.confidence)
            symbolic_score = symbolic_analysis.rule_confidence

            combined_score = (
                self.neural_weight * neural_score +
                self.symbolic_weight * symbolic_score
            )

            # Final prediction based on combined score
            prediction = 1 if combined_score > 0.5 else 0
            confidence = combined_score

            # Approval requires symbolic pass
            approved = symbolic_analysis.passed

            explanation = [
                f"Neural: {neural_pred.prediction} (conf: {neural_pred.confidence:.2f}, weight: {self.neural_weight:.2f})",
                f"Symbolic: {'PASS' if symbolic_analysis.passed else 'FAIL'} (conf: {symbolic_score:.2f}, weight: {self.symbolic_weight:.2f})",
                f"Combined: prediction={prediction}, confidence={combined_score:.2f}"
            ]

        return NeuroSymbolicDecision(
            prediction=prediction,
            confidence=confidence,
            approved=approved,
            neural_prediction=neural_pred,
            symbolic_analysis=symbolic_analysis,
            reasoning_mode=mode,
            neural_weight=self.neural_weight,
            symbolic_weight=self.symbolic_weight,
            explanation=explanation,
            rule_violations=symbolic_analysis.check_result.violations,
            confidence_breakdown={
                'neural': neural_pred.confidence,
                'symbolic': symbolic_analysis.rule_confidence,
                'combined': confidence
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoner"""
        rule_stats = self.rule_engine.get_rule_statistics()

        return {
            'has_gnn': self.gnn_model is not None,
            'neural_weight': self.neural_weight,
            'symbolic_weight': self.symbolic_weight,
            'rule_engine': rule_stats,
            'device': self.device
        }


# Example usage
if __name__ == "__main__":
    # Create reasoner
    reasoner = NeuroSymbolicReasoner(
        gnn_model=None,  # Would use actual GNN in production
        neural_weight=0.6,
        symbolic_weight=0.4
    )

    # Test 1: Reason about code quality
    print("=== Test 1: Code Quality Reasoning ===")
    good_code = "def add(x: int, y: int) -> int: return x + y"
    decision1 = reasoner.reason(good_code, mode=ReasoningMode.HYBRID)
    print(f"Prediction: {decision1.prediction}")
    print(f"Confidence: {decision1.confidence:.2f}")
    print(f"Approved: {decision1.approved}")
    print(f"Trustworthy: {decision1.is_trustworthy}")
    print(f"Safety Score: {decision1.safety_score:.2f}")

    # Test 2: Reason about code change
    print("\n=== Test 2: Code Change Reasoning ===")
    before = "x = eval('1 + 1')"
    after = "x = 1 + 1"
    decision2, verification = reasoner.reason_on_change(before, after, mode=ReasoningMode.NEURAL_THEN_SYMBOLIC)
    print(f"Decision: {'after' if decision2.prediction == 1 else 'before'}")
    print(f"Confidence: {decision2.confidence:.2f}")
    print(f"Approved: {decision2.approved}")
    print(f"Verification: {verification.status.value}")
    print("\nExplanation:")
    for line in decision2.explanation:
        print(f"  {line}")

    # Test 3: Statistics
    print("\n=== Test 3: Reasoner Statistics ===")
    stats = reasoner.get_statistics()
    print(f"Has GNN: {stats['has_gnn']}")
    print(f"Neural weight: {stats['neural_weight']:.2f}")
    print(f"Symbolic weight: {stats['symbolic_weight']:.2f}")
    print(f"Total rules: {stats['rule_engine']['total_rules']}")
