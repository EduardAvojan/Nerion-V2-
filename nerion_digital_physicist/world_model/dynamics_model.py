"""
Dynamics Model

Learns to predict state transitions and execution outcomes
from historical execution data.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


@dataclass
class StateTransition:
    """Represents a state transition"""
    before_state: Dict[str, Any]
    action: str  # Code that was executed
    after_state: Dict[str, Any]
    outcome: str  # success, error, timeout
    execution_time_ms: float


@dataclass
class OutcomePrediction:
    """Predicted execution outcome"""
    outcome: Any  # ExecutionOutcome enum
    confidence: float
    state_transitions: List[StateTransition]
    reasoning: List[str]


class DynamicsModel:
    """
    Learns dynamics of code execution.

    Predicts how code execution changes program state
    based on historical observations.

    This is a simplified version - in production this would
    use a neural network trained on execution traces.
    """

    def __init__(self):
        # Historical transitions
        self.transitions: List[StateTransition] = []

        # Pattern frequency (for heuristics)
        self.pattern_counts: Dict[str, int] = defaultdict(int)
        self.error_patterns: Dict[str, List[str]] = defaultdict(list)

        # Success/failure statistics
        self.success_stats: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))  # (success, total)

    def train_from_transitions(self, transitions: List[StateTransition]):
        """
        Train dynamics model from historical transitions.

        Args:
            transitions: List of observed state transitions
        """
        self.transitions.extend(transitions)

        for transition in transitions:
            # Extract patterns
            pattern_key = self._extract_pattern(transition.action)
            self.pattern_counts[pattern_key] += 1

            # Track success/failure
            success, total = self.success_stats[pattern_key]
            if transition.outcome == "success":
                self.success_stats[pattern_key] = (success + 1, total + 1)
            else:
                self.success_stats[pattern_key] = (success, total + 1)
                self.error_patterns[pattern_key].append(transition.outcome)

        print(f"[DynamicsModel] Trained on {len(transitions)} transitions")
        print(f"[DynamicsModel] Learned {len(self.pattern_counts)} patterns")

    def predict_outcome(
        self,
        code: str,
        initial_state: Dict[str, Any],
        symbolic_paths: Optional[List[Any]] = None
    ) -> OutcomePrediction:
        """
        Predict execution outcome.

        Args:
            code: Code to predict
            initial_state: Initial state
            symbolic_paths: Symbolic execution paths (if available)

        Returns:
            Outcome prediction
        """
        # Import here to avoid circular dependency
        from .simulator import ExecutionOutcome

        # Extract pattern
        pattern = self._extract_pattern(code)

        # Check historical success rate
        if pattern in self.success_stats:
            success, total = self.success_stats[pattern]
            success_rate = success / total if total > 0 else 0.5

            if success_rate >= 0.8:
                return OutcomePrediction(
                    outcome=ExecutionOutcome.SUCCESS,
                    confidence=success_rate,
                    state_transitions=[],
                    reasoning=[
                        f"Pattern '{pattern}' has {success_rate:.1%} success rate",
                        f"Based on {total} historical executions"
                    ]
                )
            elif success_rate < 0.3:
                return OutcomePrediction(
                    outcome=ExecutionOutcome.ERROR,
                    confidence=1.0 - success_rate,
                    state_transitions=[],
                    reasoning=[
                        f"Pattern '{pattern}' has {success_rate:.1%} success rate",
                        f"Common errors: {', '.join(self.error_patterns[pattern][:3])}"
                    ]
                )

        # Analyze code for error patterns
        error_indicators = self._check_error_patterns(code)

        if error_indicators:
            return OutcomePrediction(
                outcome=ExecutionOutcome.ERROR,
                confidence=0.7,
                state_transitions=[],
                reasoning=[f"Detected error pattern: {err}" for err in error_indicators]
            )

        # Check symbolic paths for errors
        if symbolic_paths:
            error_paths = [p for p in symbolic_paths if hasattr(p, 'error_type') and p.error_type]
            if error_paths:
                return OutcomePrediction(
                    outcome=ExecutionOutcome.ERROR,
                    confidence=0.8,
                    state_transitions=[],
                    reasoning=[f"Symbolic execution found {len(error_paths)} error paths"]
                )

        # Default: predict success with medium confidence
        return OutcomePrediction(
            outcome=ExecutionOutcome.SUCCESS,
            confidence=0.6,
            state_transitions=[],
            reasoning=["No strong indicators, default prediction"]
        )

    def predict_next_state(
        self,
        current_state: Dict[str, Any],
        action: str
    ) -> Dict[str, Any]:
        """
        Predict next state after executing action.

        Args:
            current_state: Current program state
            action: Code to execute

        Returns:
            Predicted next state
        """
        # Find similar historical transitions
        similar_transitions = self._find_similar_transitions(action, current_state)

        if not similar_transitions:
            # No history - return current state unchanged
            return current_state.copy()

        # Average the after_states from similar transitions
        next_state = current_state.copy()

        # For each variable, predict new value
        for var in current_state.keys():
            values = []
            for transition in similar_transitions:
                if var in transition.after_state:
                    values.append(transition.after_state[var])

            if values:
                # Take most common value
                next_state[var] = max(set(values), key=values.count)

        return next_state

    def estimate_execution_time(self, code: str) -> float:
        """
        Estimate execution time in milliseconds.

        Args:
            code: Code to estimate

        Returns:
            Estimated time in milliseconds
        """
        pattern = self._extract_pattern(code)

        # Find historical executions
        times = []
        for transition in self.transitions:
            if self._extract_pattern(transition.action) == pattern:
                times.append(transition.execution_time_ms)

        if times:
            # Return median time
            return float(np.median(times))

        # Default estimate based on code complexity
        lines = code.count('\n') + 1
        return 1.0 + lines * 0.5  # 1ms base + 0.5ms per line

    def _extract_pattern(self, code: str) -> str:
        """Extract pattern signature from code"""
        # Simplified pattern extraction
        # In production, this would use AST analysis

        # Keywords that define pattern
        keywords = []
        if 'if' in code:
            keywords.append('conditional')
        if 'for' in code or 'while' in code:
            keywords.append('loop')
        if 'def' in code:
            keywords.append('function')
        if 'class' in code:
            keywords.append('class')
        if 'return' in code:
            keywords.append('return')
        if 'raise' in code or 'except' in code:
            keywords.append('exception')
        if '/' in code:
            keywords.append('division')
        if '[' in code and ']' in code:
            keywords.append('indexing')

        return '_'.join(keywords) if keywords else 'simple'

    def _check_error_patterns(self, code: str) -> List[str]:
        """Check for common error patterns"""
        errors = []

        # Division by zero
        if '/' in code and '0' in code:
            errors.append("potential_division_by_zero")

        # Empty indexing
        if '[' in code and ']' in code:
            errors.append("potential_index_error")

        # None attribute access
        if '.' in code and 'None' in code:
            errors.append("potential_attribute_error")

        # Unclosed resources
        if 'open(' in code and 'close()' not in code and 'with' not in code:
            errors.append("unclosed_file_handle")

        return errors

    def _find_similar_transitions(
        self,
        action: str,
        state: Dict[str, Any],
        k: int = 5
    ) -> List[StateTransition]:
        """Find k most similar historical transitions"""
        pattern = self._extract_pattern(action)

        # Find transitions with same pattern
        candidates = [
            t for t in self.transitions
            if self._extract_pattern(t.action) == pattern
        ]

        # Score by state similarity
        scored = []
        for candidate in candidates:
            similarity = self._state_similarity(state, candidate.before_state)
            scored.append((similarity, candidate))

        # Sort by similarity and return top k
        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:k]]

    def _state_similarity(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two states"""
        if not state1 and not state2:
            return 1.0

        if not state1 or not state2:
            return 0.0

        # Count matching variables
        common_vars = set(state1.keys()) & set(state2.keys())
        if not common_vars:
            return 0.0

        matches = sum(1 for var in common_vars if state1[var] == state2[var])
        return matches / len(common_vars)

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'num_transitions': len(self.transitions),
            'num_patterns': len(self.pattern_counts),
            'most_common_patterns': sorted(
                self.pattern_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'error_prone_patterns': [
                pattern for pattern, (success, total) in self.success_stats.items()
                if total > 5 and success / total < 0.5
            ]
        }


# Example usage
if __name__ == "__main__":
    model = DynamicsModel()

    # Create some training data
    transitions = [
        StateTransition(
            before_state={'x': 5, 'y': 0},
            action="z = x / y",
            after_state={'x': 5, 'y': 0, 'z': None},
            outcome="error",
            execution_time_ms=0.5
        ),
        StateTransition(
            before_state={'x': 10, 'y': 2},
            action="z = x / y",
            after_state={'x': 10, 'y': 2, 'z': 5},
            outcome="success",
            execution_time_ms=0.3
        ),
        StateTransition(
            before_state={'x': 8, 'y': 4},
            action="z = x / y",
            after_state={'x': 8, 'y': 4, 'z': 2},
            outcome="success",
            execution_time_ms=0.3
        ),
    ]

    # Train
    model.train_from_transitions(transitions)

    # Predict
    prediction = model.predict_outcome(
        "result = a / b",
        initial_state={'a': 10, 'b': 0}
    )

    print(f"Prediction: {prediction.outcome}")
    print(f"Confidence: {prediction.confidence:.2f}")
    print(f"Reasoning: {prediction.reasoning}")

    # Statistics
    print(f"\nModel Stats: {model.get_statistics()}")
