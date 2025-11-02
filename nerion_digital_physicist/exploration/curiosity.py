"""
Curiosity-Driven Exploration Engine

Combines novelty detection and interest scoring to drive autonomous
exploration and discovery of code patterns.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
from datetime import datetime
from enum import Enum

from .novelty_detector import NoveltyDetector, NoveltyScore
from .interest_scorer import InterestScorer, InterestScore, InterestSignal


class ExplorationStrategy(Enum):
    """Exploration strategies"""
    RANDOM = "random"  # Random sampling
    NOVELTY_SEEKING = "novelty_seeking"  # Prioritize novel patterns
    INTEREST_DRIVEN = "interest_driven"  # Prioritize interesting patterns
    BALANCED = "balanced"  # Balance novelty + interest
    ADAPTIVE = "adaptive"  # Adapt based on learning progress


@dataclass
class ExplorationCandidate:
    """A candidate for exploration"""
    code: str
    embedding: np.ndarray
    novelty_score: NoveltyScore
    interest_score: InterestScore
    exploration_value: float  # Combined score for exploration
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DiscoveredPattern:
    """A novel pattern discovered through exploration"""
    code: str
    embedding: np.ndarray
    novelty: float
    interest: float
    complexity: float
    discovery_timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CuriosityEngine:
    """
    Curiosity-driven exploration engine for autonomous code pattern discovery.

    Combines novelty detection and interest scoring to:
    1. Identify novel code patterns
    2. Score patterns for interestingness
    3. Prioritize exploration of high-value patterns
    4. Discover edge cases and interesting behaviors
    5. Build a collection of discovered patterns

    Usage:
        >>> engine = CuriosityEngine()
        >>> embedding = np.random.randn(768)
        >>> candidate = engine.evaluate_candidate(
        ...     code="def novel_function(): pass",
        ...     embedding=embedding,
        ...     prediction_error=0.8
        ... )
        >>> if candidate.exploration_value > 0.7:
        ...     engine.add_discovered_pattern(candidate)
    """

    def __init__(
        self,
        novelty_threshold: float = 0.7,
        interest_threshold: float = 0.6,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.BALANCED,
        novelty_weight: float = 0.5,
        interest_weight: float = 0.5,
        memory_size: int = 10000,
        embedding_dim: int = 768
    ):
        """
        Initialize curiosity engine.

        Args:
            novelty_threshold: Threshold for novelty detection
            interest_threshold: Threshold for interest scoring
            exploration_strategy: Strategy for prioritizing exploration
            novelty_weight: Weight for novelty in exploration value
            interest_weight: Weight for interest in exploration value
            memory_size: Maximum patterns to remember
            embedding_dim: Dimensionality of embeddings
        """
        self.novelty_detector = NoveltyDetector(
            novelty_threshold=novelty_threshold,
            memory_size=memory_size,
            embedding_dim=embedding_dim
        )

        self.interest_scorer = InterestScorer(
            interest_threshold=interest_threshold
        )

        self.exploration_strategy = exploration_strategy
        self.novelty_weight = novelty_weight
        self.interest_weight = interest_weight

        # Normalize weights
        total_weight = novelty_weight + interest_weight
        self.novelty_weight = novelty_weight / total_weight
        self.interest_weight = interest_weight / total_weight

        # Discovered patterns
        self.discovered_patterns: List[DiscoveredPattern] = []
        self.exploration_history: List[ExplorationCandidate] = []

        # Statistics
        self.total_evaluated = 0
        self.total_explored = 0
        self.patterns_discovered_per_month: List[Tuple[str, int]] = []

    def evaluate_candidate(
        self,
        code: str,
        embedding: np.ndarray,
        prediction_error: Optional[float] = None,
        model_confidence: Optional[float] = None,
        actual_outcome: Optional[int] = None,
        predicted_outcome: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExplorationCandidate:
        """
        Evaluate a code sample as candidate for exploration.

        Args:
            code: Code to evaluate
            embedding: Code embedding
            prediction_error: Model's prediction error
            model_confidence: Model's confidence
            actual_outcome: Actual label
            predicted_outcome: Predicted label
            metadata: Optional metadata

        Returns:
            ExplorationCandidate with exploration value
        """
        self.total_evaluated += 1

        # Check novelty
        novelty_score = self.novelty_detector.check_novelty(embedding, metadata)

        # Score interest
        interest_score = self.interest_scorer.score_code(
            code=code,
            prediction_error=prediction_error,
            model_confidence=model_confidence,
            novelty=novelty_score.novelty,
            actual_outcome=actual_outcome,
            predicted_outcome=predicted_outcome,
            metadata=metadata
        )

        # Compute exploration value based on strategy
        exploration_value = self._compute_exploration_value(
            novelty_score,
            interest_score
        )

        candidate = ExplorationCandidate(
            code=code,
            embedding=embedding,
            novelty_score=novelty_score,
            interest_score=interest_score,
            exploration_value=exploration_value,
            metadata=metadata or {}
        )

        self.exploration_history.append(candidate)

        return candidate

    def _compute_exploration_value(
        self,
        novelty_score: NoveltyScore,
        interest_score: InterestScore
    ) -> float:
        """
        Compute exploration value based on strategy.

        Args:
            novelty_score: Novelty assessment
            interest_score: Interest assessment

        Returns:
            Exploration value (0.0 to 1.0)
        """
        if self.exploration_strategy == ExplorationStrategy.RANDOM:
            return np.random.random()

        elif self.exploration_strategy == ExplorationStrategy.NOVELTY_SEEKING:
            return novelty_score.novelty

        elif self.exploration_strategy == ExplorationStrategy.INTEREST_DRIVEN:
            return interest_score.total_interest

        elif self.exploration_strategy == ExplorationStrategy.BALANCED:
            # Weighted combination
            return (
                self.novelty_weight * novelty_score.novelty +
                self.interest_weight * interest_score.total_interest
            )

        elif self.exploration_strategy == ExplorationStrategy.ADAPTIVE:
            # Adapt based on recent discovery rate
            if len(self.discovered_patterns) > 10:
                recent_discoveries = len([
                    p for p in self.discovered_patterns[-100:]
                ])
                discovery_rate = recent_discoveries / 100.0

                # If high discovery rate, favor novelty (explore)
                # If low discovery rate, favor interest (exploit)
                adaptive_novelty_weight = min(0.8, 0.3 + discovery_rate)
                adaptive_interest_weight = 1.0 - adaptive_novelty_weight

                return (
                    adaptive_novelty_weight * novelty_score.novelty +
                    adaptive_interest_weight * interest_score.total_interest
                )
            else:
                # Not enough data, use balanced
                return (
                    self.novelty_weight * novelty_score.novelty +
                    self.interest_weight * interest_score.total_interest
                )

        return 0.5  # Default

    def should_explore(self, candidate: ExplorationCandidate) -> bool:
        """
        Decide whether to explore this candidate.

        Args:
            candidate: Exploration candidate

        Returns:
            True if candidate should be explored
        """
        # Must exceed both thresholds OR have very high exploration value
        meets_thresholds = (
            candidate.novelty_score.is_novel and
            candidate.interest_score.is_interesting
        )

        high_value = candidate.exploration_value > 0.8

        return meets_thresholds or high_value

    def add_discovered_pattern(
        self,
        candidate: ExplorationCandidate
    ) -> DiscoveredPattern:
        """
        Add a pattern to discovered collection.

        Args:
            candidate: Exploration candidate to add

        Returns:
            DiscoveredPattern entry
        """
        self.total_explored += 1

        # Add to novelty detector's memory
        self.novelty_detector.add_to_memory(
            candidate.embedding,
            metadata=candidate.metadata
        )

        # Create discovered pattern entry
        pattern = DiscoveredPattern(
            code=candidate.code,
            embedding=candidate.embedding,
            novelty=candidate.novelty_score.novelty,
            interest=candidate.interest_score.total_interest,
            complexity=candidate.interest_score.signal_scores.get(
                InterestSignal.COMPLEXITY, 0.5
            ),
            discovery_timestamp=datetime.now(),
            metadata=candidate.metadata
        )

        self.discovered_patterns.append(pattern)

        return pattern

    def get_top_discoveries(
        self,
        k: int = 10,
        sort_by: str = 'novelty'  # 'novelty', 'interest', 'complexity'
    ) -> List[DiscoveredPattern]:
        """
        Get top k discovered patterns.

        Args:
            k: Number of patterns to return
            sort_by: Metric to sort by

        Returns:
            List of top patterns
        """
        if sort_by == 'novelty':
            sorted_patterns = sorted(
                self.discovered_patterns,
                key=lambda p: p.novelty,
                reverse=True
            )
        elif sort_by == 'interest':
            sorted_patterns = sorted(
                self.discovered_patterns,
                key=lambda p: p.interest,
                reverse=True
            )
        elif sort_by == 'complexity':
            sorted_patterns = sorted(
                self.discovered_patterns,
                key=lambda p: p.complexity,
                reverse=True
            )
        else:
            sorted_patterns = self.discovered_patterns

        return sorted_patterns[:k]

    def get_recent_discoveries(
        self,
        days: int = 30
    ) -> List[DiscoveredPattern]:
        """
        Get patterns discovered in the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of recent discoveries
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        recent = [
            p for p in self.discovered_patterns
            if p.discovery_timestamp >= cutoff
        ]

        return recent

    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics"""
        novelty_stats = self.novelty_detector.get_statistics()
        interest_stats = self.interest_scorer.get_statistics()

        # Compute discoveries per month
        recent_30_days = self.get_recent_discoveries(days=30)
        discoveries_per_month = len(recent_30_days)

        # Compute exploration efficiency
        exploration_rate = self.total_explored / max(1, self.total_evaluated)

        # Average discovery metrics
        if self.discovered_patterns:
            avg_novelty = np.mean([p.novelty for p in self.discovered_patterns])
            avg_interest = np.mean([p.interest for p in self.discovered_patterns])
            avg_complexity = np.mean([p.complexity for p in self.discovered_patterns])
        else:
            avg_novelty = avg_interest = avg_complexity = 0.0

        return {
            'total_evaluated': self.total_evaluated,
            'total_explored': self.total_explored,
            'exploration_rate': exploration_rate,
            'total_discoveries': len(self.discovered_patterns),
            'discoveries_last_30_days': discoveries_per_month,
            'exploration_strategy': self.exploration_strategy.value,
            'weights': {
                'novelty': self.novelty_weight,
                'interest': self.interest_weight
            },
            'average_discovery_metrics': {
                'novelty': avg_novelty,
                'interest': avg_interest,
                'complexity': avg_complexity
            },
            'novelty_detector': novelty_stats,
            'interest_scorer': interest_stats
        }

    def update_strategy(self, new_strategy: ExplorationStrategy):
        """Update exploration strategy"""
        self.exploration_strategy = new_strategy

    def update_weights(self, novelty_weight: float, interest_weight: float):
        """Update exploration value weights"""
        total = novelty_weight + interest_weight
        self.novelty_weight = novelty_weight / total
        self.interest_weight = interest_weight / total

    def export_discoveries(self, filepath: str):
        """Export discovered patterns to file"""
        import json

        data = {
            'total_discoveries': len(self.discovered_patterns),
            'export_timestamp': datetime.now().isoformat(),
            'patterns': [
                {
                    'code': p.code,
                    'novelty': p.novelty,
                    'interest': p.interest,
                    'complexity': p.complexity,
                    'discovered_at': p.discovery_timestamp.isoformat(),
                    'metadata': p.metadata
                }
                for p in self.discovered_patterns
            ],
            'statistics': self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def clear_discoveries(self):
        """Clear all discovered patterns (use with caution!)"""
        self.discovered_patterns = []
        self.novelty_detector.clear_memory()


# Example usage
if __name__ == "__main__":
    print("=== Curiosity-Driven Exploration Demo ===\n")

    engine = CuriosityEngine(
        exploration_strategy=ExplorationStrategy.BALANCED,
        novelty_threshold=0.7,
        interest_threshold=0.6
    )

    # Simulate exploration of code samples
    test_samples = [
        ("x = 1 + 1", 0.1, 0.9),  # Simple, low error, high confidence
        ("def foo(): pass", 0.3, 0.7),  # Moderate
        ("class Complex:\n    def __init__(self): pass", 0.8, 0.4),  # Complex, high error
        ("eval('1 + 1')", 0.9, 0.3),  # Security issue, very high error
    ]

    print("Evaluating candidates for exploration:\n")

    for code, error, confidence in test_samples:
        embedding = np.random.randn(768)
        candidate = engine.evaluate_candidate(
            code=code,
            embedding=embedding,
            prediction_error=error,
            model_confidence=confidence
        )

        print(f"Code: {code[:50]}...")
        print(f"Exploration Value: {candidate.exploration_value:.2f}")
        print(f"  Novelty: {candidate.novelty_score.novelty:.2f}")
        print(f"  Interest: {candidate.interest_score.total_interest:.2f}")

        if engine.should_explore(candidate):
            print("  → EXPLORING this pattern!")
            pattern = engine.add_discovered_pattern(candidate)
            print(f"  → Added to discoveries (total: {len(engine.discovered_patterns)})")
        else:
            print("  → Skipping (not interesting enough)")

        print()

    # Show statistics
    print("=== Final Statistics ===")
    stats = engine.get_statistics()
    print(f"Total evaluated: {stats['total_evaluated']}")
    print(f"Total explored: {stats['total_explored']}")
    print(f"Exploration rate: {stats['exploration_rate']:.2%}")
    print(f"Total discoveries: {stats['total_discoveries']}")
    print(f"Discoveries last 30 days: {stats['discoveries_last_30_days']}")

    print(f"\nAverage discovery metrics:")
    for metric, value in stats['average_discovery_metrics'].items():
        print(f"  {metric}: {value:.2f}")

    # Show top discoveries
    if stats['total_discoveries'] > 0:
        print(f"\n=== Top Discoveries (by novelty) ===")
        for i, pattern in enumerate(engine.get_top_discoveries(k=3, sort_by='novelty'), 1):
            print(f"{i}. Novelty={pattern.novelty:.2f}, Interest={pattern.interest:.2f}")
            print(f"   Code: {pattern.code[:60]}...")
