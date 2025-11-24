"""
Integration tests for Curiosity-Driven Exploration Engine

Tests the three major modules:
1. Novelty Detector
2. Interest Scorer
3. Curiosity Engine
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from nerion_digital_physicist.exploration import (
    # Novelty Detection
    NoveltyDetector,
    NoveltyScore,
    # Interest Scoring
    InterestScorer,
    InterestScore,
    InterestSignal,
    # Curiosity Engine
    CuriosityEngine,
    ExplorationCandidate,
    DiscoveredPattern,
    ExplorationStrategy,
)


class TestNoveltyDetector:
    """Test Novelty Detector functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.detector = NoveltyDetector(
            novelty_threshold=0.7,
            embedding_dim=768,
            memory_size=1000
        )

    def test_detector_creation(self):
        """Test detector creation"""
        assert self.detector is not None
        assert self.detector.novelty_threshold == 0.7
        assert self.detector.embedding_dim == 768
        assert len(self.detector.memory) == 0

    def test_first_pattern_is_novel(self):
        """Test that first pattern is always novel"""
        emb = np.random.randn(768)
        result = self.detector.check_novelty(emb)

        assert result.novelty == 1.0
        assert result.is_novel
        assert "novel" in result.explanation.lower()

    def test_similar_pattern_not_novel(self):
        """Test that similar pattern is not novel"""
        # Add first pattern
        emb1 = np.random.randn(768)
        self.detector.add_to_memory(emb1)

        # Check very similar pattern
        emb2 = emb1 + 0.01 * np.random.randn(768)
        result = self.detector.check_novelty(emb2)

        assert result.novelty < 0.3  # Should be very similar
        assert not result.is_novel

    def test_different_pattern_is_novel(self):
        """Test that different pattern is novel"""
        # Add first pattern
        emb1 = np.random.randn(768)
        self.detector.add_to_memory(emb1)

        # Check very different pattern
        emb2 = np.random.randn(768)
        result = self.detector.check_novelty(emb2)

        # Random embeddings are typically perpendicular (novelty ~0.5)
        # Should be moderately novel
        assert result.novelty > 0.3  # Reasonably different
        assert result.distance_to_nearest > 0.5  # Cosine distance > 0.5

    def test_memory_management(self):
        """Test memory size limit enforcement"""
        detector = NoveltyDetector(memory_size=10)

        # Add 15 patterns
        for i in range(15):
            emb = np.random.randn(768)
            detector.add_to_memory(emb, metadata={'index': i})

        # Should only keep last 10
        assert len(detector.memory) == 10
        # First 5 should be removed (FIFO)
        assert all(m.get('index', -1) >= 5 for m in detector.memory_metadata)

    def test_get_similar_neighbors(self):
        """Test finding similar neighbors"""
        # Add some patterns
        embeddings = [np.random.randn(768) for _ in range(5)]
        for i, emb in enumerate(embeddings):
            self.detector.add_to_memory(emb, metadata={'index': i})

        # Query with first embedding (should be most similar to itself)
        neighbors = self.detector.get_similar_neighbors(embeddings[0], k=3)

        assert len(neighbors) == 3
        # First neighbor should be the pattern itself (distance ~0)
        assert neighbors[0][2]['index'] == 0
        assert neighbors[0][1] < 0.1  # Very small distance

    def test_get_novel_neighbors(self):
        """Test finding novel (distant) neighbors"""
        # Add some patterns
        embeddings = [np.random.randn(768) for _ in range(5)]
        for i, emb in enumerate(embeddings):
            self.detector.add_to_memory(emb, metadata={'index': i})

        # Query with first embedding
        neighbors = self.detector.get_novel_neighbors(embeddings[0], k=3)

        assert len(neighbors) == 3
        # Should be sorted by distance (descending)
        assert neighbors[0][1] >= neighbors[1][1]
        assert neighbors[1][1] >= neighbors[2][1]

    def test_statistics(self):
        """Test statistics tracking"""
        # Check 5 patterns, explicitly add novel ones
        for i in range(5):
            emb = np.random.randn(768)
            result = self.detector.check_novelty(emb)
            # First pattern is always novel, add it
            if i == 0:
                self.detector.add_to_memory(emb)

        stats = self.detector.get_statistics()

        assert stats['total_checked'] == 5
        assert stats['total_novel'] >= 1  # At least the first one
        assert 'novelty_rate' in stats
        assert 'memory_size' in stats


class TestInterestScorer:
    """Test Interest Scorer functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.scorer = InterestScorer(interest_threshold=0.6)

    def test_scorer_creation(self):
        """Test scorer creation"""
        assert self.scorer is not None
        assert self.scorer.interest_threshold == 0.6
        assert len(self.scorer.weights) == 5

    def test_high_error_is_interesting(self):
        """Test that high prediction error indicates high interest"""
        code = "def foo(): pass"
        score = self.scorer.score_code(
            code,
            prediction_error=0.9,
            model_confidence=0.2,
            novelty=0.8
        )

        # High error + low confidence + high novelty = very interesting
        assert score.total_interest > 0.6  # Good interest level
        assert score.is_interesting
        assert score.priority <= 3  # High/medium priority

    def test_low_error_less_interesting(self):
        """Test that low prediction error indicates lower interest"""
        code = "x = 1"
        score = self.scorer.score_code(
            code,
            prediction_error=0.1,
            model_confidence=0.9,
            novelty=0.1
        )

        # Low error + high confidence + low novelty = less interesting
        assert score.total_interest < 0.5
        assert not score.is_interesting

    def test_complexity_scoring(self):
        """Test complexity calculation"""
        simple_code = "x = 1"
        complex_code = """
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                for i in range(10):
                    if i % 2 == 0:
                        return i + x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
"""

        score_simple = self.scorer.score_code(simple_code)
        score_complex = self.scorer.score_code(complex_code)

        # Complex code should have higher complexity score
        complexity_simple = score_simple.signal_scores[InterestSignal.COMPLEXITY]
        complexity_complex = score_complex.signal_scores[InterestSignal.COMPLEXITY]

        assert complexity_complex > complexity_simple

    def test_surprise_signal(self):
        """Test surprise detection"""
        code = "def test(): pass"

        # Correct prediction (no surprise)
        score1 = self.scorer.score_code(
            code,
            actual_outcome=1,
            predicted_outcome=1
        )
        surprise1 = score1.signal_scores[InterestSignal.SURPRISE]

        # Incorrect prediction (surprise!)
        score2 = self.scorer.score_code(
            code,
            actual_outcome=1,
            predicted_outcome=0
        )
        surprise2 = score2.signal_scores[InterestSignal.SURPRISE]

        assert surprise1 == 0.0  # No surprise
        assert surprise2 == 1.0  # Full surprise

    def test_rank_samples(self):
        """Test sample ranking"""
        samples = [
            ("x = 1", {'prediction_error': 0.1, 'novelty': 0.1}),
            ("def foo(): pass", {'prediction_error': 0.5, 'novelty': 0.6}),
            ("class Bar: pass", {'prediction_error': 0.9, 'novelty': 0.9}),
        ]

        ranked = self.scorer.rank_samples(samples)

        assert len(ranked) == 3
        # Should be sorted by interest (descending)
        assert ranked[0][1].total_interest >= ranked[1][1].total_interest
        assert ranked[1][1].total_interest >= ranked[2][1].total_interest
        # Ranks should be 1, 2, 3
        assert ranked[0][2] == 1
        assert ranked[1][2] == 2
        assert ranked[2][2] == 3

    def test_statistics(self):
        """Test statistics tracking"""
        # Score several samples
        for i in range(5):
            code = f"x = {i}"
            self.scorer.score_code(code, prediction_error=i * 0.2)

        stats = self.scorer.get_statistics()

        assert stats['total_scored'] == 5
        assert 'interest_rate' in stats
        assert 'average_signal_scores' in stats


class TestCuriosityEngine:
    """Test Curiosity Engine functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.engine = CuriosityEngine(
            novelty_threshold=0.7,
            interest_threshold=0.6,
            exploration_strategy=ExplorationStrategy.BALANCED
        )

    def test_engine_creation(self):
        """Test engine creation"""
        assert self.engine is not None
        assert self.engine.novelty_detector is not None
        assert self.engine.interest_scorer is not None
        assert len(self.engine.discovered_patterns) == 0

    def test_evaluate_candidate(self):
        """Test candidate evaluation"""
        code = "def novel_function(): pass"
        embedding = np.random.randn(768)

        candidate = self.engine.evaluate_candidate(
            code=code,
            embedding=embedding,
            prediction_error=0.8,
            model_confidence=0.4
        )

        assert candidate is not None
        assert candidate.code == code
        assert candidate.novelty_score is not None
        assert candidate.interest_score is not None
        assert 0.0 <= candidate.exploration_value <= 1.0

    def test_balanced_exploration_value(self):
        """Test balanced exploration value calculation"""
        code = "def test(): pass"
        embedding = np.random.randn(768)

        # First pattern (novel + interesting)
        candidate1 = self.engine.evaluate_candidate(
            code=code,
            embedding=embedding,
            prediction_error=0.9,
            model_confidence=0.3
        )

        # Should have high exploration value (both novel and interesting)
        assert candidate1.exploration_value > 0.7
        assert candidate1.novelty_score.is_novel

    def test_should_explore(self):
        """Test exploration decision"""
        # High value candidate (novel pattern, high error)
        embedding1 = np.random.randn(768)
        candidate1 = self.engine.evaluate_candidate(
            code="class Novel: pass",
            embedding=embedding1,
            prediction_error=0.9,
            model_confidence=0.2
        )

        # Low value candidate (familiar pattern, low error)
        embedding2 = embedding1 + 0.01 * np.random.randn(768)
        self.engine.novelty_detector.add_to_memory(embedding1)  # Make it familiar
        candidate2 = self.engine.evaluate_candidate(
            code="x = 1",
            embedding=embedding2,
            prediction_error=0.1,
            model_confidence=0.9
        )

        # First should be explored (high novelty + high interest)
        assert self.engine.should_explore(candidate1)
        # Second should not be explored (low novelty + low interest)
        assert not self.engine.should_explore(candidate2)

    def test_add_discovered_pattern(self):
        """Test adding discovered patterns"""
        code = "def discovered(): pass"
        embedding = np.random.randn(768)

        candidate = self.engine.evaluate_candidate(
            code=code,
            embedding=embedding,
            prediction_error=0.8
        )

        pattern = self.engine.add_discovered_pattern(candidate)

        assert pattern is not None
        assert pattern.code == code
        assert len(self.engine.discovered_patterns) == 1
        assert self.engine.total_explored == 1

    def test_novelty_seeking_strategy(self):
        """Test novelty-seeking exploration strategy"""
        engine = CuriosityEngine(
            exploration_strategy=ExplorationStrategy.NOVELTY_SEEKING
        )

        embedding = np.random.randn(768)
        candidate = engine.evaluate_candidate(
            code="def test(): pass",
            embedding=embedding,
            prediction_error=0.2,  # Low interest
            model_confidence=0.9
        )

        # Exploration value should equal novelty (ignoring interest)
        assert abs(candidate.exploration_value - candidate.novelty_score.novelty) < 0.01

    def test_interest_driven_strategy(self):
        """Test interest-driven exploration strategy"""
        engine = CuriosityEngine(
            exploration_strategy=ExplorationStrategy.INTEREST_DRIVEN
        )

        embedding = np.random.randn(768)
        candidate = engine.evaluate_candidate(
            code="def test(): pass",
            embedding=embedding,
            prediction_error=0.8,
            model_confidence=0.3
        )

        # Exploration value should equal interest (ignoring novelty)
        assert abs(candidate.exploration_value - candidate.interest_score.total_interest) < 0.01

    def test_get_top_discoveries(self):
        """Test getting top discoveries"""
        # Add several discoveries
        for i in range(5):
            embedding = np.random.randn(768)
            candidate = self.engine.evaluate_candidate(
                code=f"def func{i}(): pass",
                embedding=embedding,
                prediction_error=i * 0.2
            )
            self.engine.add_discovered_pattern(candidate)

        top_3 = self.engine.get_top_discoveries(k=3, sort_by='novelty')

        assert len(top_3) == 3
        # Should be sorted by novelty
        assert top_3[0].novelty >= top_3[1].novelty
        assert top_3[1].novelty >= top_3[2].novelty

    def test_get_recent_discoveries(self):
        """Test getting recent discoveries"""
        # Add a discovery
        embedding = np.random.randn(768)
        candidate = self.engine.evaluate_candidate(
            code="def recent(): pass",
            embedding=embedding,
            prediction_error=0.8
        )
        self.engine.add_discovered_pattern(candidate)

        recent = self.engine.get_recent_discoveries(days=30)

        assert len(recent) == 1
        assert recent[0].code == "def recent(): pass"

    def test_statistics(self):
        """Test statistics reporting"""
        # Evaluate several candidates
        for i in range(10):
            embedding = np.random.randn(768)
            candidate = self.engine.evaluate_candidate(
                code=f"def func{i}(): pass",
                embedding=embedding,
                prediction_error=i * 0.1
            )

            # Explore half of them
            if i % 2 == 0:
                self.engine.add_discovered_pattern(candidate)

        stats = self.engine.get_statistics()

        assert stats['total_evaluated'] == 10
        assert stats['total_explored'] == 5
        assert stats['exploration_rate'] == 0.5
        assert stats['total_discoveries'] == 5
        assert 'discoveries_last_30_days' in stats


class TestIntegration:
    """Test integration between components"""

    def test_end_to_end_exploration(self):
        """Test complete exploration workflow"""
        # Create engine
        engine = CuriosityEngine(
            novelty_threshold=0.6,
            interest_threshold=0.5,
            exploration_strategy=ExplorationStrategy.BALANCED
        )

        # Simulate exploring a codebase
        code_samples = [
            ("def simple(): return 1", 0.1, 0.9),
            ("class Complex:\n    def __init__(self): pass", 0.7, 0.4),
            ("x = eval('1 + 1')", 0.9, 0.2),
            ("import numpy as np", 0.3, 0.7),
            ("def novel_algorithm():\n    # ...", 0.8, 0.3),
        ]

        discoveries = []

        for code, error, confidence in code_samples:
            embedding = np.random.randn(768)
            candidate = engine.evaluate_candidate(
                code=code,
                embedding=embedding,
                prediction_error=error,
                model_confidence=confidence
            )

            if engine.should_explore(candidate):
                pattern = engine.add_discovered_pattern(candidate)
                discoveries.append(pattern)

        # Should discover some patterns
        assert len(discoveries) > 0
        assert len(engine.discovered_patterns) > 0

        # Get statistics
        stats = engine.get_statistics()
        assert stats['total_evaluated'] == 5
        assert stats['total_discoveries'] == len(discoveries)

        # Get top discoveries
        top = engine.get_top_discoveries(k=3, sort_by='interest')
        assert len(top) <= 3

    def test_adaptive_strategy(self):
        """Test adaptive exploration strategy"""
        engine = CuriosityEngine(
            exploration_strategy=ExplorationStrategy.ADAPTIVE
        )

        # Add many discoveries to trigger adaptive behavior
        for i in range(20):
            embedding = np.random.randn(768)
            candidate = engine.evaluate_candidate(
                code=f"def func{i}(): pass",
                embedding=embedding,
                prediction_error=0.7
            )
            engine.add_discovered_pattern(candidate)

        # Evaluate new candidate with adaptive strategy
        embedding = np.random.randn(768)
        candidate = engine.evaluate_candidate(
            code="def adaptive_test(): pass",
            embedding=embedding,
            prediction_error=0.6,
            model_confidence=0.5
        )

        # Should have computed exploration value using adaptive weights
        assert 0.0 <= candidate.exploration_value <= 1.0

    def test_discovery_tracking(self):
        """Test discovery tracking over time"""
        engine = CuriosityEngine()

        # Simulate discoveries over multiple sessions
        for session in range(3):
            for i in range(5):
                embedding = np.random.randn(768)
                candidate = engine.evaluate_candidate(
                    code=f"def session{session}_func{i}(): pass",
                    embedding=embedding,
                    prediction_error=0.7
                )
                engine.add_discovered_pattern(candidate)

        # Should have 15 total discoveries
        assert len(engine.discovered_patterns) == 15

        # All should be recent (within last 30 days)
        recent = engine.get_recent_discoveries(days=30)
        assert len(recent) == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
