"""
Explainable Planner with Chain-of-Thought Reasoning

Wraps the existing Nerion planner with chain-of-thought reasoning
to provide explainable, trustworthy code modification decisions.

Integration:
- Maintains backward compatibility with existing planner API
- Adds reasoning traces to all planning decisions
- Stores reasoning history for learning and audit
- Connects to Mission Control GUI for visualization
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from selfcoder.reasoning.chain_of_thought import (
    ChainOfThoughtReasoner,
    ReasoningResult
)


@dataclass
class ExplainablePlan:
    """A plan with reasoning explanation"""
    actions: List[Dict[str, Any]]
    reasoning: ReasoningResult
    execution_strategy: str
    estimated_risk: str  # low, medium, high
    requires_human_review: bool


class ExplainablePlanner:
    """
    Planner with chain-of-thought reasoning for explainability.

    Wraps the existing planner with explicit reasoning traces,
    enabling users to understand WHY Nerion makes each decision.

    Usage:
        >>> planner = ExplainablePlanner()
        >>> plan = planner.create_plan(
        ...     task="Fix authentication bug",
        ...     context={"file": "app/auth.py"}
        ... )
        >>> print(plan.reasoning.user_explanation)
        >>> if not plan.requires_human_review:
        ...     execute_plan(plan.actions)
    """

    def __init__(
        self,
        base_planner: Optional[Any] = None,
        min_confidence_for_execution: float = 0.75
    ):
        """
        Initialize explainable planner.

        Args:
            base_planner: Existing planner to wrap (optional)
            min_confidence_for_execution: Confidence threshold for autonomous execution
        """
        self.base_planner = base_planner
        self.reasoner = ChainOfThoughtReasoner(
            min_confidence_for_execution=min_confidence_for_execution
        )

        # Planning history
        self.plan_history: List[ExplainablePlan] = []

    def create_plan(
        self,
        task: str,
        context: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ExplainablePlan:
        """
        Create a plan with reasoning explanation.

        Args:
            task: High-level task description
            context: Code context (files, AST, dependencies, etc.)
            user_preferences: User preferences (risk tolerance, style, etc.)

        Returns:
            ExplainablePlan with actions and reasoning
        """
        print(f"[ExplainablePlanner] Creating plan for: {task}")

        # Generate base plan (would call existing planner in production)
        base_actions = self._generate_base_plan(task, context)

        # Convert actions to proposed change description
        proposed_change = self._actions_to_description(base_actions)

        # Apply chain-of-thought reasoning
        reasoning = self.reasoner.reason_about_modification(
            task=task,
            context=context,
            proposed_change=proposed_change,
            additional_info=user_preferences
        )

        # Determine execution strategy based on confidence
        execution_strategy = self._determine_execution_strategy(
            reasoning.overall_confidence
        )

        # Estimate risk level
        estimated_risk = self._estimate_risk_level(reasoning.risks_identified)

        # Check if human review required
        requires_human_review = not reasoning.execution_approved

        # Create explainable plan
        plan = ExplainablePlan(
            actions=base_actions if reasoning.execution_approved else [],
            reasoning=reasoning,
            execution_strategy=execution_strategy,
            estimated_risk=estimated_risk,
            requires_human_review=requires_human_review
        )

        # Store in history
        self.plan_history.append(plan)

        print(f"[ExplainablePlanner] Plan created: {len(base_actions)} actions")
        print(f"[ExplainablePlanner] Confidence: {reasoning.overall_confidence:.0%}")
        print(f"[ExplainablePlanner] Strategy: {execution_strategy}")
        print(f"[ExplainablePlanner] Risk: {estimated_risk}")

        return plan

    def explain_plan(
        self,
        plan: ExplainablePlan,
        detail_level: str = "medium"
    ) -> str:
        """
        Generate human-readable explanation of plan.

        Args:
            plan: Plan to explain
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Human-readable explanation
        """
        if detail_level == "brief":
            return self._brief_explanation(plan)
        elif detail_level == "detailed":
            return self._detailed_explanation(plan)
        else:
            return plan.reasoning.user_explanation

    def _generate_base_plan(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate base plan actions using existing planner.

        Args:
            task: Task description
            context: Code context

        Returns:
            List of action dictionaries
        """
        # If base planner is provided, use it
        if self.base_planner and hasattr(self.base_planner, 'plan_from_text'):
            try:
                file_path = context.get('file') or context.get('target_file')
                plan = self.base_planner.plan_from_text(
                    instruction=task,
                    target_file=file_path,
                    brief_context=context
                )

                # Convert plan format to explainable actions
                return self._convert_plan_to_actions(plan, context)
            except Exception as e:
                print(f"[ExplainablePlanner] Base planner error: {e}, using fallback")

        # Otherwise, try to use the heuristic planner directly
        try:
            from selfcoder.planner.planner import plan_from_text

            file_path = context.get('file') or context.get('target_file')
            plan = plan_from_text(
                instruction=task,
                target_file=file_path,
                brief_context=context
            )

            # Convert plan format to explainable actions
            return self._convert_plan_to_actions(plan, context)

        except Exception as e:
            print(f"[ExplainablePlanner] Planner integration error: {e}, using fallback")

        # Fallback: generate simple actions
        return self._generate_fallback_plan(task, context)

    def _convert_plan_to_actions(
        self,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert planner output format to explainable action format.

        Args:
            plan: Plan from existing planner
            context: Code context

        Returns:
            List of action dictionaries
        """
        actions = []

        # Extract target file
        target_file = plan.get('target_file', context.get('file', 'unknown'))

        # Add precondition checks as actions
        preconditions = plan.get('preconditions', [])
        if preconditions:
            actions.append({
                'type': 'verify_preconditions',
                'file': target_file,
                'checks': preconditions,
                'reason': 'Verify preconditions before modification'
            })

        # Convert planner actions to explainable format
        for action in plan.get('actions', []):
            kind = action.get('kind', 'unknown')
            payload = action.get('payload', {})

            if kind == 'create_file':
                actions.append({
                    'type': 'create_file',
                    'file': payload.get('path', target_file),
                    'doc': payload.get('doc'),
                    'reason': 'Create new file for implementation'
                })

            elif kind == 'insert_function':
                actions.append({
                    'type': 'insert_function',
                    'file': target_file,
                    'name': payload.get('name'),
                    'doc': payload.get('doc'),
                    'reason': f"Add function {payload.get('name')}"
                })

            elif kind == 'insert_class':
                actions.append({
                    'type': 'insert_class',
                    'file': target_file,
                    'name': payload.get('name'),
                    'doc': payload.get('doc'),
                    'reason': f"Add class {payload.get('name')}"
                })

            elif kind == 'add_module_docstring':
                actions.append({
                    'type': 'add_docstring',
                    'file': target_file,
                    'scope': 'module',
                    'doc': payload.get('doc'),
                    'reason': 'Add module documentation'
                })

            elif kind == 'add_function_docstring':
                actions.append({
                    'type': 'add_docstring',
                    'file': target_file,
                    'scope': 'function',
                    'function': payload.get('function'),
                    'doc': payload.get('doc'),
                    'reason': f"Document function {payload.get('function')}"
                })

            elif kind == 'ensure_test':
                actions.append({
                    'type': 'ensure_test',
                    'file': target_file,
                    'symbol': payload.get('symbol'),
                    'test_type': payload.get('type'),
                    'reason': f"Ensure test coverage for {payload.get('symbol')}"
                })

            else:
                # Generic action
                actions.append({
                    'type': kind,
                    'file': target_file,
                    'payload': payload,
                    'reason': f"Apply {kind.replace('_', ' ')}"
                })

        # Add postcondition checks as actions
        postconditions = plan.get('postconditions', [])
        if postconditions:
            actions.append({
                'type': 'verify_postconditions',
                'file': target_file,
                'checks': postconditions,
                'reason': 'Verify changes are correct'
            })

        return actions

    def _generate_fallback_plan(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate fallback plan when existing planner fails.

        Args:
            task: Task description
            context: Code context

        Returns:
            List of action dictionaries
        """
        file_path = context.get('file', 'unknown')
        lines = context.get('lines', 'unknown')

        actions = [
            {
                'type': 'read_file',
                'file': file_path,
                'reason': 'Understand current code state'
            },
            {
                'type': 'modify_code',
                'file': file_path,
                'lines': lines,
                'modification': task,
                'reason': 'Apply requested change'
            },
            {
                'type': 'run_tests',
                'scope': 'affected',
                'reason': 'Validate change did not break functionality'
            }
        ]

        return actions

    def _actions_to_description(
        self,
        actions: List[Dict[str, Any]]
    ) -> str:
        """Convert action list to human-readable description"""
        descriptions = []
        for action in actions:
            action_type = action.get('type', 'unknown')
            reason = action.get('reason', '')

            if action_type == 'modify_code':
                file = action.get('file', 'unknown')
                descriptions.append(f"Modify {file}: {reason}")
            elif action_type == 'read_file':
                descriptions.append(f"Read file: {reason}")
            elif action_type == 'run_tests':
                descriptions.append(f"Run tests: {reason}")
            else:
                descriptions.append(f"{action_type}: {reason}")

        return " → ".join(descriptions)

    def _determine_execution_strategy(
        self,
        confidence: float
    ) -> str:
        """Determine execution strategy based on confidence"""
        if confidence >= 0.90:
            return "autonomous"
        elif confidence >= 0.75:
            return "autonomous_with_monitoring"
        elif confidence >= 0.60:
            return "supervised"
        else:
            return "manual_approval_required"

    def _estimate_risk_level(
        self,
        risks: List[str]
    ) -> str:
        """Estimate overall risk level from identified risks"""
        if not risks:
            return "low"

        high_risks = sum(1 for r in risks if "HIGH" in r)
        medium_risks = sum(1 for r in risks if "MEDIUM" in r)

        if high_risks >= 2:
            return "critical"
        elif high_risks >= 1:
            return "high"
        elif medium_risks >= 2:
            return "medium"
        else:
            return "low"

    def _brief_explanation(self, plan: ExplainablePlan) -> str:
        """Generate brief explanation"""
        confidence = plan.reasoning.overall_confidence
        risk = plan.estimated_risk

        explanation = f"**Decision: {plan.execution_strategy.replace('_', ' ').title()}**\n\n"
        explanation += f"Confidence: {confidence:.0%} | Risk: {risk}\n\n"

        if plan.requires_human_review:
            explanation += "⚠️ **Human review required before execution**\n"
        else:
            explanation += "✅ **Approved for autonomous execution**\n"

        return explanation

    def _detailed_explanation(self, plan: ExplainablePlan) -> str:
        """Generate detailed explanation"""
        explanation = plan.reasoning.user_explanation
        explanation += "\n\n## Planned Actions\n\n"

        for i, action in enumerate(plan.actions, 1):
            action_type = action.get('type', 'unknown')
            reason = action.get('reason', '')
            explanation += f"{i}. **{action_type.replace('_', ' ').title()}**: {reason}\n"

        if plan.reasoning.fallback_plan:
            explanation += f"\n## Fallback Plan\n\n{plan.reasoning.fallback_plan}\n"

        return explanation

    def get_planning_history(self) -> List[ExplainablePlan]:
        """Get history of all plans created"""
        return self.plan_history

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get planner performance metrics.

        Returns metrics for:
        - Decision quality (confidence calibration)
        - Risk assessment accuracy
        - User comprehension rate
        """
        if not self.plan_history:
            return {'count': 0}

        total = len(self.plan_history)
        approved = sum(1 for p in self.plan_history if not p.requires_human_review)
        avg_confidence = sum(p.reasoning.overall_confidence for p in self.plan_history) / total

        risk_distribution = {
            'low': sum(1 for p in self.plan_history if p.estimated_risk == 'low'),
            'medium': sum(1 for p in self.plan_history if p.estimated_risk == 'medium'),
            'high': sum(1 for p in self.plan_history if p.estimated_risk == 'high'),
            'critical': sum(1 for p in self.plan_history if p.estimated_risk == 'critical'),
        }

        return {
            'total_plans': total,
            'autonomous_approval_rate': approved / total,
            'avg_confidence': avg_confidence,
            'risk_distribution': risk_distribution,
            'strategy_distribution': self._get_strategy_distribution()
        }

    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of execution strategies"""
        distribution: Dict[str, int] = {}
        for plan in self.plan_history:
            strategy = plan.execution_strategy
            distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution


# Integration with Mission Control GUI
def export_reasoning_for_gui(plan: ExplainablePlan) -> Dict[str, Any]:
    """
    Export reasoning in format for Mission Control GUI visualization.

    Args:
        plan: Explainable plan

    Returns:
        Dictionary suitable for GUI rendering
    """
    reasoning_chain = []
    for trace in plan.reasoning.reasoning_chain:
        reasoning_chain.append({
            'step': trace.step.value,
            'content': trace.content,
            'confidence': trace.confidence,
            'evidence': trace.evidence,
            'alternatives': trace.alternatives_considered
        })

    return {
        'decision': plan.reasoning.decision,
        'reasoning_chain': reasoning_chain,
        'overall_confidence': plan.reasoning.overall_confidence,
        'execution_approved': not plan.requires_human_review,
        'risk_level': plan.estimated_risk,
        'execution_strategy': plan.execution_strategy,
        'risks': plan.reasoning.risks_identified,
        'fallback_plan': plan.reasoning.fallback_plan,
        'user_explanation': plan.reasoning.user_explanation
    }


# Example integration with existing planner
def integrate_with_existing_planner():
    """
    Example of how to integrate ExplainablePlanner with existing planner.

    In production, replace existing planner calls with:
    """
    from selfcoder.planner.explainable_planner import ExplainablePlanner

    # Initialize
    planner = ExplainablePlanner()

    # Instead of:
    # plan = old_planner.create_plan(task, context)
    # execute_plan(plan)

    # Use:
    explainable_plan = planner.create_plan(task, context)

    # Show reasoning to user
    print(explainable_plan.reasoning.user_explanation)

    # Execute only if approved
    if not explainable_plan.requires_human_review:
        execute_plan(explainable_plan.actions)
    else:
        print("⚠️ Human review required")
        # Send to Mission Control GUI for review


# Example usage
def example_usage():
    """Example of explainable planning"""
    planner = ExplainablePlanner()

    # Create plan with reasoning
    task = "Add input validation to prevent SQL injection"
    context = {
        'file': 'app/database.py',
        'lines': '120-145',
        'has_tests': True,
        'complexity': 'high',
        'dependencies': ['app/routes.py', 'app/models.py', 'app/utils.py'],
        'is_production': True
    }

    plan = planner.create_plan(task, context)

    # Display explanation
    print("\n" + "="*60)
    print("PLAN EXPLANATION")
    print("="*60)
    print(plan.reasoning.user_explanation)

    # Show actions
    if not plan.requires_human_review:
        print("\n" + "="*60)
        print("PLANNED ACTIONS")
        print("="*60)
        for i, action in enumerate(plan.actions, 1):
            print(f"{i}. {action['type']}: {action.get('reason', '')}")

    # Performance metrics
    print("\n" + "="*60)
    print("PLANNER METRICS")
    print("="*60)
    metrics = planner.get_performance_metrics()
    print(f"Total plans created: {metrics['total_plans']}")
    print(f"Autonomous approval rate: {metrics['autonomous_approval_rate']:.0%}")
    print(f"Average confidence: {metrics['avg_confidence']:.0%}")


if __name__ == "__main__":
    example_usage()
