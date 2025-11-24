"""
Exploration Module - Curiosity-Driven Pattern Discovery

Enables autonomous exploration and discovery of novel code patterns
through intrinsic motivation (novelty + interest).
"""

from .novelty_detector import (
    NoveltyDetector,
    NoveltyScore,
)

from .interest_scorer import (
    InterestScorer,
    InterestScore,
    InterestSignal,
)

from .curiosity import (
    CuriosityEngine,
    ExplorationCandidate,
    DiscoveredPattern,
    ExplorationStrategy,
)

__all__ = [
    # Novelty Detection
    'NoveltyDetector',
    'NoveltyScore',
    # Interest Scoring
    'InterestScorer',
    'InterestScore',
    'InterestSignal',
    # Curiosity Engine
    'CuriosityEngine',
    'ExplorationCandidate',
    'DiscoveredPattern',
    'ExplorationStrategy',
]
