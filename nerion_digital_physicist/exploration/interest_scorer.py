"""
Interest Scorer

Scores how "interesting" a code sample is based on multiple intrinsic
motivation signals: learning progress, surprise, complexity, and uncertainty.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import ast


class InterestSignal(Enum):
    """Types of interest signals"""
    LEARNING_PROGRESS = "learning_progress"  # How much can we learn?
    SURPRISE = "surprise"  # How unexpected?
    COMPLEXITY = "complexity"  # How complex?
    UNCERTAINTY = "uncertainty"  # How uncertain are we?
    NOVELTY = "novelty"  # How novel?


@dataclass
class InterestScore:
    """Result of interest scoring"""
    total_interest: float  # Overall interest score (0.0 to 1.0)
    signal_scores: Dict[InterestSignal, float]  # Individual signal scores
    is_interesting: bool  # True if interest exceeds threshold
    explanation: List[str]  # Human-readable explanation
    priority: int  # Priority rank (1 = highest)


class InterestScorer:
    """
    Scores code samples for interestingness using intrinsic motivation signals.

    Combines multiple signals:
    - Learning Progress: Prediction error (more error = more to learn)
    - Surprise: Unexpected outcomes
    - Complexity: Structural complexity
    - Uncertainty: Model uncertainty
    - Novelty: Pattern novelty

    Usage:
        >>> scorer = InterestScorer()
        >>> score = scorer.score_code(
        ...     code="def complex_func(): ...",
        ...     prediction_error=0.8,
        ...     novelty=0.7
        ... )
        >>> if score.is_interesting:
        ...     print(f"Interesting! {score.explanation}")
    """

    def __init__(
        self,
        interest_threshold: float = 0.6,
        weights: Optional[Dict[InterestSignal, float]] = None
    ):
        """
        Initialize interest scorer.

        Args:
            interest_threshold: Threshold for considering sample interesting
            weights: Custom weights for each signal (default: equal weights)
        """
        self.interest_threshold = interest_threshold

        # Default weights (equal importance)
        self.weights = weights or {
            InterestSignal.LEARNING_PROGRESS: 0.3,
            InterestSignal.SURPRISE: 0.2,
            InterestSignal.COMPLEXITY: 0.2,
            InterestSignal.UNCERTAINTY: 0.15,
            InterestSignal.NOVELTY: 0.15
        }

        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Statistics
        self.total_scored = 0
        self.total_interesting = 0
        self.signal_history: Dict[InterestSignal, List[float]] = {
            signal: [] for signal in InterestSignal
        }

    def score_code(
        self,
        code: str,
        prediction_error: Optional[float] = None,
        model_confidence: Optional[float] = None,
        novelty: Optional[float] = None,
        actual_outcome: Optional[int] = None,
        predicted_outcome: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InterestScore:
        """
        Score a code sample for interestingness.

        Args:
            code: Code to score
            prediction_error: Model's prediction error (0.0 to 1.0)
            model_confidence: Model's confidence (0.0 to 1.0)
            novelty: Novelty score from NoveltyDetector (0.0 to 1.0)
            actual_outcome: Actual label/outcome
            predicted_outcome: Predicted label/outcome
            metadata: Optional metadata

        Returns:
            InterestScore with overall interest and breakdown
        """
        self.total_scored += 1

        signal_scores = {}
        explanation = []

        # 1. Learning Progress Signal
        if prediction_error is not None:
            # High error = high learning potential
            learning_progress = prediction_error
            signal_scores[InterestSignal.LEARNING_PROGRESS] = learning_progress
            self.signal_history[InterestSignal.LEARNING_PROGRESS].append(learning_progress)

            if learning_progress > 0.7:
                explanation.append(f"High learning potential (error={learning_progress:.2f})")
        else:
            signal_scores[InterestSignal.LEARNING_PROGRESS] = 0.5  # Neutral

        # 2. Surprise Signal
        if actual_outcome is not None and predicted_outcome is not None:
            # Surprise = outcome doesn't match prediction
            surprise = 1.0 if actual_outcome != predicted_outcome else 0.0
            signal_scores[InterestSignal.SURPRISE] = surprise
            self.signal_history[InterestSignal.SURPRISE].append(surprise)

            if surprise > 0.5:
                explanation.append("Surprising outcome (prediction was wrong)")
        else:
            signal_scores[InterestSignal.SURPRISE] = 0.5  # Neutral

        # 3. Complexity Signal
        complexity = self._compute_complexity(code)
        signal_scores[InterestSignal.COMPLEXITY] = complexity
        self.signal_history[InterestSignal.COMPLEXITY].append(complexity)

        if complexity > 0.7:
            explanation.append(f"High complexity (score={complexity:.2f})")

        # 4. Uncertainty Signal
        if model_confidence is not None:
            # High uncertainty = low confidence
            uncertainty = 1.0 - model_confidence
            signal_scores[InterestSignal.UNCERTAINTY] = uncertainty
            self.signal_history[InterestSignal.UNCERTAINTY].append(uncertainty)

            if uncertainty > 0.7:
                explanation.append(f"High uncertainty (confidence={model_confidence:.2f})")
        else:
            signal_scores[InterestSignal.UNCERTAINTY] = 0.5  # Neutral

        # 5. Novelty Signal
        if novelty is not None:
            signal_scores[InterestSignal.NOVELTY] = novelty
            self.signal_history[InterestSignal.NOVELTY].append(novelty)

            if novelty > 0.7:
                explanation.append(f"Novel pattern (novelty={novelty:.2f})")
        else:
            signal_scores[InterestSignal.NOVELTY] = 0.5  # Neutral

        # Compute weighted total interest
        total_interest = sum(
            signal_scores.get(signal, 0.5) * self.weights[signal]
            for signal in InterestSignal
        )

        is_interesting = total_interest >= self.interest_threshold

        if is_interesting:
            self.total_interesting += 1

        # Determine priority (higher interest = higher priority)
        if total_interest > 0.8:
            priority = 1  # Very high priority
        elif total_interest > 0.7:
            priority = 2  # High priority
        elif total_interest > 0.6:
            priority = 3  # Medium priority
        else:
            priority = 4  # Low priority

        if not explanation:
            explanation.append(f"Moderate interest (score={total_interest:.2f})")

        return InterestScore(
            total_interest=total_interest,
            signal_scores=signal_scores,
            is_interesting=is_interesting,
            explanation=explanation,
            priority=priority
        )

    def _compute_complexity(self, code: str) -> float:
        """
        Compute structural complexity of code.

        Uses multiple metrics:
        - Cyclomatic complexity
        - Nesting depth
        - Number of nodes
        - Unique constructs

        Returns:
            Complexity score (0.0 to 1.0)
        """
        try:
            tree = ast.parse(code)
        except:
            # Parse error = somewhat complex
            return 0.6

        # Count nodes
        node_count = sum(1 for _ in ast.walk(tree))

        # Calculate cyclomatic complexity
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        # Calculate max nesting depth
        max_depth = self._max_nesting_depth(tree)

        # Count unique constructs
        unique_constructs = len(set(type(node).__name__ for node in ast.walk(tree)))

        # Normalize to [0, 1]
        # Thresholds based on typical code
        node_score = min(1.0, node_count / 100.0)  # 100 nodes = very complex
        complexity_score = min(1.0, complexity / 20.0)  # Complexity 20 = very complex
        depth_score = min(1.0, max_depth / 8.0)  # Depth 8 = very complex
        construct_score = min(1.0, unique_constructs / 20.0)  # 20 unique = very complex

        # Weighted average
        total_complexity = (
            0.3 * node_score +
            0.3 * complexity_score +
            0.2 * depth_score +
            0.2 * construct_score
        )

        return total_complexity

    def _max_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of AST"""
        max_depth = current_depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                child_depth = self._max_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._max_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def rank_samples(
        self,
        samples: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, InterestScore, int]]:
        """
        Rank code samples by interestingness.

        Args:
            samples: List of (code, scoring_params) tuples

        Returns:
            List of (code, score, rank) tuples sorted by interest
        """
        scored_samples = []

        for code, params in samples:
            score = self.score_code(code, **params)
            scored_samples.append((code, score))

        # Sort by total interest (descending)
        scored_samples.sort(key=lambda x: x[1].total_interest, reverse=True)

        # Add ranks
        ranked_samples = [
            (code, score, rank + 1)
            for rank, (code, score) in enumerate(scored_samples)
        ]

        return ranked_samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get interest scoring statistics"""
        interest_rate = self.total_interesting / max(1, self.total_scored)

        # Compute average signal scores
        avg_signals = {}
        for signal, history in self.signal_history.items():
            if history:
                avg_signals[signal.value] = np.mean(history)
            else:
                avg_signals[signal.value] = 0.0

        return {
            'total_scored': self.total_scored,
            'total_interesting': self.total_interesting,
            'interest_rate': interest_rate,
            'interest_threshold': self.interest_threshold,
            'signal_weights': {s.value: w for s, w in self.weights.items()},
            'average_signal_scores': avg_signals
        }

    def update_weights(self, new_weights: Dict[InterestSignal, float]):
        """Update signal weights (must sum to 1.0)"""
        total_weight = sum(new_weights.values())
        self.weights = {k: v / total_weight for k, v in new_weights.items()}


# Example usage
if __name__ == "__main__":
    scorer = InterestScorer(interest_threshold=0.6)

    print("=== Interest Scoring Demo ===\n")

    # Test 1: Simple code with low error
    code1 = "x = 1 + 1"
    score1 = scorer.score_code(
        code1,
        prediction_error=0.1,
        model_confidence=0.9,
        novelty=0.2
    )
    print(f"Code 1: {code1}")
    print(f"Interest: {score1.total_interest:.2f} | Interesting: {score1.is_interesting}")
    print(f"Explanation: {score1.explanation}\n")

    # Test 2: Complex code with high error
    code2 = """
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
"""
    score2 = scorer.score_code(
        code2,
        prediction_error=0.8,
        model_confidence=0.4,
        novelty=0.9,
        actual_outcome=1,
        predicted_outcome=0
    )
    print(f"Code 2: Complex nested function")
    print(f"Interest: {score2.total_interest:.2f} | Interesting: {score2.is_interesting}")
    print(f"Priority: {score2.priority}")
    print(f"Explanation: {score2.explanation}\n")

    # Statistics
    print("=== Statistics ===")
    stats = scorer.get_statistics()
    print(f"Total scored: {stats['total_scored']}")
    print(f"Total interesting: {stats['total_interesting']}")
    print(f"Interest rate: {stats['interest_rate']:.2%}")
    print(f"\nAverage signal scores:")
    for signal, avg in stats['average_signal_scores'].items():
        print(f"  {signal}: {avg:.2f}")
