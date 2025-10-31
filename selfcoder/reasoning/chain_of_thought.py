"""
Chain-of-Thought Reasoning for Code Decisions

Enables explicit, step-by-step reasoning for code modification decisions.
Based on "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)

Key Concepts:
- System 2 thinking: Slow, deliberate, logical
- Explicit reasoning traces: Each decision has auditable steps
- Confidence calibration: State uncertainty explicitly
- Self-correction: Detect and fix reasoning errors

Integration with Nerion:
- Wraps planner decisions with reasoning traces
- Provides explainability for user trust
- Enables meta-reasoning (reasoning about reasoning)
- Stores reasoning history for learning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class ReasoningStep(Enum):
    """Steps in chain-of-thought reasoning"""
    PROBLEM_UNDERSTANDING = "problem_understanding"
    CONTEXT_ANALYSIS = "context_analysis"
    APPROACH_GENERATION = "approach_generation"
    CONSEQUENCE_PREDICTION = "consequence_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION = "decision"


@dataclass
class ThoughtTrace:
    """A single thought in the reasoning chain"""
    step: ReasoningStep
    content: str
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningResult:
    """Result of chain-of-thought reasoning"""
    decision: str
    reasoning_chain: List[ThoughtTrace]
    overall_confidence: float
    risks_identified: List[str]
    fallback_plan: Optional[str] = None
    execution_approved: bool = False
    user_explanation: str = ""


class ChainOfThoughtReasoner:
    """
    Chain-of-thought reasoner for code modification decisions.

    Implements 6-step reasoning pipeline:
    1. Problem Understanding: What are we trying to achieve?
    2. Context Analysis: What's the current state?
    3. Approach Generation: What are the possible approaches?
    4. Consequence Prediction: What will happen if we do this?
    5. Risk Assessment: What could go wrong?
    6. Decision: What should we do?

    Usage:
        >>> reasoner = ChainOfThoughtReasoner()
        >>> result = reasoner.reason_about_modification(
        ...     task="Fix authentication bug",
        ...     context={"file": "app/auth.py", "lines": "45-67"},
        ...     proposed_change="Add null check before token validation"
        ... )
        >>> if result.execution_approved:
        ...     execute_modification(result.decision)
    """

    def __init__(
        self,
        min_confidence_for_execution: float = 0.75,
        min_confidence_for_flagging: float = 0.60
    ):
        """
        Initialize chain-of-thought reasoner.

        Args:
            min_confidence_for_execution: Minimum confidence for autonomous execution
            min_confidence_for_flagging: Minimum confidence threshold (below = abort)
        """
        self.min_confidence_for_execution = min_confidence_for_execution
        self.min_confidence_for_flagging = min_confidence_for_flagging

        # Reasoning history for learning
        self.reasoning_history: List[ReasoningResult] = []

    def reason_about_modification(
        self,
        task: str,
        context: Dict[str, Any],
        proposed_change: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Reason about a proposed code modification.

        Args:
            task: High-level task description
            context: Code context (file, lines, AST, etc.)
            proposed_change: Proposed modification
            additional_info: Additional context (test results, user feedback, etc.)

        Returns:
            ReasoningResult with decision and reasoning chain
        """
        print(f"[ChainOfThought] Reasoning about: {task}")

        reasoning_chain: List[ThoughtTrace] = []

        # Step 1: Problem Understanding
        problem_trace = self._understand_problem(task, context)
        reasoning_chain.append(problem_trace)

        # Step 2: Context Analysis
        context_trace = self._analyze_context(context, additional_info)
        reasoning_chain.append(context_trace)

        # Step 3: Approach Generation
        approach_trace = self._generate_approaches(proposed_change, context)
        reasoning_chain.append(approach_trace)

        # Step 4: Consequence Prediction
        consequence_trace = self._predict_consequences(proposed_change, context)
        reasoning_chain.append(consequence_trace)

        # Step 5: Risk Assessment
        risk_trace = self._assess_risks(proposed_change, context, consequence_trace)
        reasoning_chain.append(risk_trace)

        # Step 6: Decision
        decision_trace = self._make_decision(
            task,
            proposed_change,
            reasoning_chain
        )
        reasoning_chain.append(decision_trace)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(reasoning_chain)

        # Determine execution approval
        execution_approved = overall_confidence >= self.min_confidence_for_execution

        if overall_confidence < self.min_confidence_for_flagging:
            execution_approved = False
            decision = "ABORT: Confidence too low"
        else:
            decision = proposed_change

        # Generate user explanation
        user_explanation = self._generate_user_explanation(
            reasoning_chain,
            overall_confidence,
            execution_approved
        )

        # Extract risks
        risks_identified = risk_trace.evidence

        # Create result
        result = ReasoningResult(
            decision=decision,
            reasoning_chain=reasoning_chain,
            overall_confidence=overall_confidence,
            risks_identified=risks_identified,
            fallback_plan=self._generate_fallback_plan(risk_trace),
            execution_approved=execution_approved,
            user_explanation=user_explanation
        )

        # Store in history
        self.reasoning_history.append(result)

        print(f"[ChainOfThought] Decision: {decision}")
        print(f"[ChainOfThought] Confidence: {overall_confidence:.2f}")
        print(f"[ChainOfThought] Execution approved: {execution_approved}")

        return result

    def _understand_problem(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> ThoughtTrace:
        """
        Step 1: Understand the problem we're solving.

        Args:
            task: Task description
            context: Code context

        Returns:
            ThoughtTrace for problem understanding
        """
        # Extract key aspects of the problem
        file_path = context.get('file', 'unknown')
        lines = context.get('lines', 'unknown')

        content = (
            f"The task is: {task}\n"
            f"Location: {file_path} (lines {lines})\n"
            f"Goal: Modify code to achieve task requirements"
        )

        evidence = [
            f"Task description: {task}",
            f"Target file: {file_path}",
            f"Scope: {context.get('scope', 'function-level')}"
        ]

        # Confidence based on clarity of task description
        confidence = 0.9 if len(task) > 20 else 0.7

        return ThoughtTrace(
            step=ReasoningStep.PROBLEM_UNDERSTANDING,
            content=content,
            confidence=confidence,
            evidence=evidence
        )

    def _analyze_context(
        self,
        context: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]]
    ) -> ThoughtTrace:
        """
        Step 2: Analyze the current code context.

        Args:
            context: Code context
            additional_info: Additional information

        Returns:
            ThoughtTrace for context analysis
        """
        # Analyze code structure
        has_tests = context.get('has_tests', False)
        complexity = context.get('complexity', 'unknown')
        dependencies = context.get('dependencies', [])

        content = (
            f"Current state analysis:\n"
            f"- Has tests: {has_tests}\n"
            f"- Complexity: {complexity}\n"
            f"- Dependencies: {len(dependencies)} modules\n"
        )

        evidence = [
            f"Tests available: {has_tests}",
            f"Cyclomatic complexity: {complexity}",
            f"Imported by: {len(dependencies)} modules"
        ]

        # Confidence based on available context
        confidence = 0.8
        if has_tests:
            confidence += 0.1
        if complexity != 'unknown':
            confidence += 0.1

        return ThoughtTrace(
            step=ReasoningStep.CONTEXT_ANALYSIS,
            content=content,
            confidence=min(confidence, 1.0),
            evidence=evidence
        )

    def _generate_approaches(
        self,
        proposed_change: str,
        context: Dict[str, Any]
    ) -> ThoughtTrace:
        """
        Step 3: Generate alternative approaches.

        Args:
            proposed_change: Proposed modification
            context: Code context

        Returns:
            ThoughtTrace for approach generation
        """
        # Generate alternatives (simplified - would use LLM in production)
        alternatives = [
            proposed_change,
            "Alternative: Add comprehensive error handling",
            "Alternative: Refactor entire function for clarity"
        ]

        content = (
            f"Proposed approach: {proposed_change}\n"
            f"Alternatives considered: {len(alternatives) - 1}\n"
        )

        evidence = alternatives

        # Confidence based on number of alternatives
        confidence = 0.7 + (0.1 * min(len(alternatives), 3))

        return ThoughtTrace(
            step=ReasoningStep.APPROACH_GENERATION,
            content=content,
            confidence=confidence,
            evidence=evidence,
            alternatives_considered=alternatives[1:]  # Exclude primary
        )

    def _predict_consequences(
        self,
        proposed_change: str,
        context: Dict[str, Any]
    ) -> ThoughtTrace:
        """
        Step 4: Predict consequences of the change.

        Args:
            proposed_change: Proposed modification
            context: Code context

        Returns:
            ThoughtTrace for consequence prediction
        """
        # Predict outcomes (would use world model in production)
        has_tests = context.get('has_tests', False)

        positive_consequences = [
            "Fixes the reported bug",
            "Improves code robustness"
        ]

        negative_consequences = []
        if not has_tests:
            negative_consequences.append("No tests to validate fix")

        dependencies = context.get('dependencies', [])
        if len(dependencies) > 5:
            negative_consequences.append("May affect multiple dependent modules")

        content = (
            f"Predicted positive outcomes:\n" +
            "\n".join(f"  + {c}" for c in positive_consequences) + "\n" +
            f"Predicted negative outcomes:\n" +
            "\n".join(f"  - {c}" for c in negative_consequences)
        )

        evidence = positive_consequences + negative_consequences

        # Confidence based on available information
        confidence = 0.75 if has_tests else 0.60

        return ThoughtTrace(
            step=ReasoningStep.CONSEQUENCE_PREDICTION,
            content=content,
            confidence=confidence,
            evidence=evidence
        )

    def _assess_risks(
        self,
        proposed_change: str,
        context: Dict[str, Any],
        consequence_trace: ThoughtTrace
    ) -> ThoughtTrace:
        """
        Step 5: Assess risks of the change.

        Args:
            proposed_change: Proposed modification
            context: Code context
            consequence_trace: Previous consequence prediction

        Returns:
            ThoughtTrace for risk assessment
        """
        risks = []

        # Check for high-risk patterns
        if "delete" in proposed_change.lower():
            risks.append("HIGH: Deleting code may break functionality")

        if not context.get('has_tests', False):
            risks.append("MEDIUM: No tests to validate change")

        if len(context.get('dependencies', [])) > 10:
            risks.append("MEDIUM: Change may affect many dependent modules")

        if context.get('is_production', False):
            risks.append("HIGH: Modifying production code")

        if not risks:
            risks.append("LOW: Minimal risk identified")

        content = (
            f"Risk assessment:\n" +
            "\n".join(f"  {r}" for r in risks)
        )

        evidence = risks

        # Confidence inversely proportional to number of high risks
        high_risks = sum(1 for r in risks if "HIGH" in r)
        confidence = max(0.5, 1.0 - (high_risks * 0.2))

        return ThoughtTrace(
            step=ReasoningStep.RISK_ASSESSMENT,
            content=content,
            confidence=confidence,
            evidence=evidence
        )

    def _make_decision(
        self,
        task: str,
        proposed_change: str,
        reasoning_chain: List[ThoughtTrace]
    ) -> ThoughtTrace:
        """
        Step 6: Make the final decision.

        Args:
            task: Original task
            proposed_change: Proposed modification
            reasoning_chain: Previous reasoning steps

        Returns:
            ThoughtTrace for decision
        """
        # Aggregate evidence from previous steps
        all_evidence = []
        for trace in reasoning_chain:
            all_evidence.extend(trace.evidence)

        # Count risks
        risk_trace = reasoning_chain[-1]  # Risk assessment is last
        high_risks = sum(1 for r in risk_trace.evidence if "HIGH" in r)

        # Make decision
        if high_risks > 1:
            decision = "DEFER to human review due to multiple high risks"
            confidence = 0.5
        elif high_risks == 1:
            decision = f"APPROVE with caution: {proposed_change}"
            confidence = 0.7
        else:
            decision = f"APPROVE: {proposed_change}"
            confidence = 0.9

        content = (
            f"Final decision: {decision}\n"
            f"Reasoning: Based on {len(reasoning_chain)} steps of analysis"
        )

        return ThoughtTrace(
            step=ReasoningStep.DECISION,
            content=content,
            confidence=confidence,
            evidence=all_evidence
        )

    def _calculate_overall_confidence(
        self,
        reasoning_chain: List[ThoughtTrace]
    ) -> float:
        """Calculate overall confidence from reasoning chain"""
        if not reasoning_chain:
            return 0.0

        # Weighted average (later steps weighted more)
        weights = [1.0 + (i * 0.2) for i in range(len(reasoning_chain))]
        confidences = [trace.confidence for trace in reasoning_chain]

        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum

    def _generate_user_explanation(
        self,
        reasoning_chain: List[ThoughtTrace],
        overall_confidence: float,
        execution_approved: bool
    ) -> str:
        """Generate human-readable explanation"""
        explanation = "## Reasoning Summary\n\n"

        for i, trace in enumerate(reasoning_chain, 1):
            explanation += f"**Step {i}: {trace.step.value.replace('_', ' ').title()}**\n"
            explanation += f"{trace.content}\n"
            explanation += f"*Confidence: {trace.confidence:.0%}*\n\n"

        explanation += f"**Overall Confidence: {overall_confidence:.0%}**\n\n"

        if execution_approved:
            explanation += "✅ **Decision: Approved for execution**\n"
        elif overall_confidence >= self.min_confidence_for_flagging:
            explanation += "⚠️ **Decision: Flagged for human review** (confidence 60-75%)\n"
        else:
            explanation += "❌ **Decision: Aborted** (confidence <60%)\n"

        return explanation

    def _generate_fallback_plan(
        self,
        risk_trace: ThoughtTrace
    ) -> Optional[str]:
        """Generate fallback plan based on risks"""
        high_risks = [r for r in risk_trace.evidence if "HIGH" in r]

        if not high_risks:
            return None

        return (
            f"Fallback plan if change causes issues:\n" +
            "1. Run automated tests to detect failures\n" +
            "2. Monitor error logs for new exceptions\n" +
            "3. Revert change if tests fail or errors increase\n" +
            "4. Investigate root cause before retry"
        )

    def get_reasoning_history(self) -> List[ReasoningResult]:
        """Get history of all reasoning decisions"""
        return self.reasoning_history

    def get_calibration_metrics(self) -> Dict[str, float]:
        """
        Calculate confidence calibration metrics.

        Confidence calibration error: |stated_confidence - actual_accuracy|
        """
        if not self.reasoning_history:
            return {'count': 0}

        # This would require actual execution results to compute
        # For now, return placeholder
        return {
            'count': len(self.reasoning_history),
            'avg_confidence': sum(r.overall_confidence for r in self.reasoning_history) / len(self.reasoning_history),
            'approval_rate': sum(1 for r in self.reasoning_history if r.execution_approved) / len(self.reasoning_history)
        }


# Integration with planner
def create_reasoning_wrapper(planner_func):
    """
    Decorator to wrap planner functions with chain-of-thought reasoning.

    Usage:
        @create_reasoning_wrapper
        def plan_modification(task, context):
            # Original planning logic
            return proposed_change
    """
    reasoner = ChainOfThoughtReasoner()

    def wrapper(task: str, context: Dict[str, Any], **kwargs):
        # Get proposed change from original planner
        proposed_change = planner_func(task, context, **kwargs)

        # Apply chain-of-thought reasoning
        result = reasoner.reason_about_modification(
            task=task,
            context=context,
            proposed_change=proposed_change
        )

        # Return result with reasoning
        return {
            'proposed_change': result.decision,
            'reasoning': result.user_explanation,
            'confidence': result.overall_confidence,
            'approved': result.execution_approved,
            'risks': result.risks_identified
        }

    return wrapper


# Example usage
def example_usage():
    """Example of chain-of-thought reasoning"""
    reasoner = ChainOfThoughtReasoner()

    # Simulate code modification decision
    task = "Fix authentication bug where null tokens cause crash"
    context = {
        'file': 'app/auth.py',
        'lines': '45-67',
        'has_tests': True,
        'complexity': 'medium',
        'dependencies': ['app/routes.py', 'app/models.py'],
        'is_production': True
    }
    proposed_change = "Add null check: if token is None: raise ValueError('Token required')"

    result = reasoner.reason_about_modification(
        task=task,
        context=context,
        proposed_change=proposed_change
    )

    print(result.user_explanation)
    print(f"\nExecution approved: {result.execution_approved}")
    print(f"Confidence: {result.overall_confidence:.0%}")


if __name__ == "__main__":
    example_usage()
