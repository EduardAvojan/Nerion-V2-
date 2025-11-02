"""
Novelty Detector

Detects when code patterns are novel/unseen using embedding similarity
and maintains a memory of encountered patterns.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
import hashlib


@dataclass
class NoveltyScore:
    """Result of novelty detection"""
    novelty: float  # 0.0 (seen before) to 1.0 (completely novel)
    distance_to_nearest: float  # Distance to most similar seen pattern
    nearest_neighbor_index: Optional[int]  # Index of most similar pattern
    is_novel: bool  # True if novelty exceeds threshold
    explanation: str


class NoveltyDetector:
    """
    Detects novel code patterns using embedding similarity.

    Uses a memory of previously seen embeddings and computes
    distance to nearest neighbor to quantify novelty.

    Usage:
        >>> detector = NoveltyDetector(novelty_threshold=0.7)
        >>> embedding = np.random.randn(768)
        >>> result = detector.check_novelty(embedding)
        >>> if result.is_novel:
        ...     detector.add_to_memory(embedding)
    """

    def __init__(
        self,
        novelty_threshold: float = 0.7,
        memory_size: int = 10000,
        embedding_dim: int = 768,
        distance_metric: str = 'cosine'
    ):
        """
        Initialize novelty detector.

        Args:
            novelty_threshold: Threshold for considering pattern novel (0.0-1.0)
            memory_size: Maximum number of patterns to remember
            embedding_dim: Dimensionality of embeddings
            distance_metric: 'cosine' or 'euclidean'
        """
        self.novelty_threshold = novelty_threshold
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric

        # Memory of seen embeddings (stored as numpy array)
        self.memory: List[np.ndarray] = []
        self.memory_metadata: List[Dict[str, Any]] = []

        # Statistics
        self.total_checked = 0
        self.total_novel = 0

    def check_novelty(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NoveltyScore:
        """
        Check if an embedding represents a novel pattern.

        Args:
            embedding: Code embedding to check
            metadata: Optional metadata about the code

        Returns:
            NoveltyScore with novelty value and explanation
        """
        self.total_checked += 1

        # Normalize embedding
        embedding = self._normalize(embedding)

        if len(self.memory) == 0:
            # First pattern is always novel
            self.total_novel += 1
            return NoveltyScore(
                novelty=1.0,
                distance_to_nearest=float('inf'),
                nearest_neighbor_index=None,
                is_novel=True,
                explanation="First pattern seen - completely novel"
            )

        # Find nearest neighbor in memory
        distances = self._compute_distances(embedding)
        nearest_idx = int(np.argmin(distances))
        min_distance = distances[nearest_idx]

        # Convert distance to novelty score (0.0 = identical, 1.0 = very different)
        if self.distance_metric == 'cosine':
            # Cosine distance: 0 (identical) to 2 (opposite)
            # Map to novelty: 0 (identical) to 1.0 (very different)
            novelty = min_distance / 2.0
        else:  # euclidean
            # Normalize Euclidean distance to [0, 1] range
            # Assume max distance is sqrt(2 * embedding_dim) for normalized vectors
            max_distance = np.sqrt(2 * self.embedding_dim)
            novelty = min(1.0, min_distance / max_distance)

        is_novel = novelty >= self.novelty_threshold

        if is_novel:
            self.total_novel += 1

        # Generate explanation
        if novelty > 0.9:
            explanation = f"Highly novel pattern (novelty={novelty:.2f})"
        elif novelty > self.novelty_threshold:
            explanation = f"Novel pattern (novelty={novelty:.2f})"
        elif novelty > 0.5:
            explanation = f"Somewhat familiar (novelty={novelty:.2f})"
        else:
            explanation = f"Very similar to seen pattern (novelty={novelty:.2f})"

        return NoveltyScore(
            novelty=novelty,
            distance_to_nearest=min_distance,
            nearest_neighbor_index=nearest_idx,
            is_novel=is_novel,
            explanation=explanation
        )

    def add_to_memory(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a pattern to memory.

        Args:
            embedding: Code embedding to remember
            metadata: Optional metadata about the code
        """
        # Normalize embedding
        embedding = self._normalize(embedding)

        # Add to memory
        self.memory.append(embedding)
        self.memory_metadata.append(metadata or {})

        # Enforce memory size limit (FIFO)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            self.memory_metadata.pop(0)

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _compute_distances(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute distances to all patterns in memory.

        Args:
            embedding: Normalized embedding

        Returns:
            Array of distances
        """
        # Stack all memory embeddings
        memory_array = np.array(self.memory)  # (N, embedding_dim)

        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # For normalized vectors: cosine_sim = dot product
            similarities = np.dot(memory_array, embedding)
            distances = 1.0 - similarities
        else:  # euclidean
            # Euclidean distance
            distances = np.linalg.norm(memory_array - embedding, axis=1)

        return distances

    def get_novel_neighbors(
        self,
        embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Get k most novel (distant) neighbors.

        Args:
            embedding: Query embedding
            k: Number of neighbors to return

        Returns:
            List of (index, distance, metadata) tuples
        """
        if len(self.memory) == 0:
            return []

        # Normalize embedding
        embedding = self._normalize(embedding)

        # Compute distances
        distances = self._compute_distances(embedding)

        # Get top-k most distant (most novel)
        k = min(k, len(distances))
        top_k_indices = np.argsort(distances)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append((
                int(idx),
                distances[idx],
                self.memory_metadata[idx]
            ))

        return results

    def get_similar_neighbors(
        self,
        embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Get k most similar (nearest) neighbors.

        Args:
            embedding: Query embedding
            k: Number of neighbors to return

        Returns:
            List of (index, distance, metadata) tuples
        """
        if len(self.memory) == 0:
            return []

        # Normalize embedding
        embedding = self._normalize(embedding)

        # Compute distances
        distances = self._compute_distances(embedding)

        # Get top-k most similar (nearest)
        k = min(k, len(distances))
        top_k_indices = np.argsort(distances)[:k]

        results = []
        for idx in top_k_indices:
            results.append((
                int(idx),
                distances[idx],
                self.memory_metadata[idx]
            ))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get novelty detection statistics"""
        novelty_rate = self.total_novel / max(1, self.total_checked)

        return {
            'total_checked': self.total_checked,
            'total_novel': self.total_novel,
            'novelty_rate': novelty_rate,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory_size,
            'memory_utilization': len(self.memory) / self.memory_size,
            'novelty_threshold': self.novelty_threshold,
            'distance_metric': self.distance_metric
        }

    def clear_memory(self):
        """Clear all memory"""
        self.memory = []
        self.memory_metadata = []

    def save_memory(self, filepath: str):
        """Save memory to disk"""
        np.savez(
            filepath,
            embeddings=np.array(self.memory),
            metadata=self.memory_metadata,
            stats={
                'total_checked': self.total_checked,
                'total_novel': self.total_novel,
                'novelty_threshold': self.novelty_threshold
            }
        )

    def load_memory(self, filepath: str):
        """Load memory from disk"""
        data = np.load(filepath, allow_pickle=True)
        self.memory = list(data['embeddings'])
        self.memory_metadata = list(data['metadata'])

        stats = data['stats'].item()
        self.total_checked = stats.get('total_checked', 0)
        self.total_novel = stats.get('total_novel', 0)


# Example usage
if __name__ == "__main__":
    detector = NoveltyDetector(novelty_threshold=0.7, embedding_dim=768)

    # Simulate checking patterns
    print("=== Novelty Detection Demo ===\n")

    # First pattern (always novel)
    emb1 = np.random.randn(768)
    result1 = detector.check_novelty(emb1, metadata={'code': 'def foo(): pass'})
    print(f"Pattern 1: {result1.explanation}")
    detector.add_to_memory(emb1, metadata={'code': 'def foo(): pass'})

    # Similar pattern (not novel)
    emb2 = emb1 + 0.1 * np.random.randn(768)
    result2 = detector.check_novelty(emb2, metadata={'code': 'def foo(): return 42'})
    print(f"Pattern 2: {result2.explanation}")

    # Very different pattern (novel)
    emb3 = np.random.randn(768)
    result3 = detector.check_novelty(emb3, metadata={'code': 'class Bar: pass'})
    print(f"Pattern 3: {result3.explanation}")
    detector.add_to_memory(emb3, metadata={'code': 'class Bar: pass'})

    # Statistics
    print(f"\n=== Statistics ===")
    stats = detector.get_statistics()
    print(f"Total checked: {stats['total_checked']}")
    print(f"Total novel: {stats['total_novel']}")
    print(f"Novelty rate: {stats['novelty_rate']:.2%}")
    print(f"Memory size: {stats['memory_size']}/{stats['memory_capacity']}")
