"""
Graph Loading Utilities for Continuous Learning

Converts various data sources to PyTorch Geometric graphs for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import torch

# Import existing graph creation functions
from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import SemanticEmbedder
from nerion_digital_physicist.infrastructure.memory import Experience
from nerion_digital_physicist.types import GraphData, TrainingExample


class GraphLoader:
    """
    Loads and caches graphs for training.

    Converts code (from experiences, lessons, or raw source) into
    PyTorch Geometric Data objects for GNN training.
    """

    def __init__(
        self,
        use_semantic_embeddings: bool = True,
        cache_graphs: bool = True
    ):
        """
        Initialize graph loader.

        Args:
            use_semantic_embeddings: Use CodeBERT embeddings (slower but better)
            cache_graphs: Cache graphs to avoid re-parsing
        """
        self.use_semantic_embeddings = use_semantic_embeddings
        self.cache_graphs = cache_graphs

        # Initialize embedder if needed
        self.embedder: Optional[SemanticEmbedder] = None
        if use_semantic_embeddings:
            try:
                self.embedder = SemanticEmbedder()
            except Exception as e:
                print(f"[GraphLoader] Failed to init embedder: {e}")
                print(f"[GraphLoader] Falling back to hash-based embeddings")

        # Graph cache
        self._cache: Dict[str, GraphData] = {}

    def load_from_experience(
        self,
        experience: Experience
    ) -> Optional[TrainingExample]:
        """
        Load graph from Experience object.

        Args:
            experience: Experience from ReplayStore

        Returns:
            (graph, label) tuple or None if loading fails
        """
        # Extract code from experience metadata
        code = experience.metadata.get('source_code')
        if not code:
            code = experience.metadata.get('code_before')  # Try alternative key

        if not code:
            print(f"[GraphLoader] No code found in experience {experience.experience_id}")
            return None

        # Determine label from status
        if experience.status == "failed":
            label = 1  # Bad quality
        elif experience.status == "solved":
            label = 0  # Good quality
        else:
            label = 0  # Default to good

        # Load graph
        try:
            graph = self._load_graph_from_code(code, experience.experience_id)
            return (graph, label)
        except Exception as e:
            print(f"[GraphLoader] Failed to load graph: {e}")
            return None

    def load_from_experiences(
        self,
        experiences: List[Experience]
    ) -> List[TrainingExample]:
        """
        Load graphs from multiple experiences.

        Args:
            experiences: List of Experience objects

        Returns:
            List of (graph, label) tuples
        """
        data = []
        for i, exp in enumerate(experiences):
            if i % 50 == 0:
                print(f"[GraphLoader] Loading {i}/{len(experiences)}...")

            result = self.load_from_experience(exp)
            if result:
                data.append(result)

        print(f"[GraphLoader] Loaded {len(data)}/{len(experiences)} graphs successfully")
        return data

    def load_from_lesson(
        self,
        lesson: Dict[str, Any]
    ) -> Optional[TrainingExample]:
        """
        Load graph from curriculum lesson.

        Args:
            lesson: Lesson dictionary from curriculum.sqlite

        Returns:
            (graph, label) tuple or None if loading fails
        """
        # Get code from lesson (before or after)
        code = lesson.get('before_code') or lesson.get('buggy_code')
        if not code:
            print(f"[GraphLoader] No code found in lesson {lesson.get('name')}")
            return None

        # Determine label from lesson metadata
        # Lessons with "before" code are typically buggy (label=1)
        if 'before_code' in lesson or 'buggy_code' in lesson:
            label = 1  # Buggy code
        else:
            label = 0  # Good code

        # Load graph
        lesson_id = lesson.get('name', 'unknown')
        try:
            graph = self._load_graph_from_code(code, lesson_id)
            return (graph, label)
        except Exception as e:
            print(f"[GraphLoader] Failed to load graph: {e}")
            return None

    def load_from_lessons(
        self,
        lessons: List[Dict[str, Any]]
    ) -> List[TrainingExample]:
        """
        Load graphs from multiple lessons.

        Args:
            lessons: List of lesson dictionaries

        Returns:
            List of (graph, label) tuples
        """
        data = []
        for i, lesson in enumerate(lessons):
            if i % 50 == 0:
                print(f"[GraphLoader] Loading {i}/{len(lessons)}...")

            result = self.load_from_lesson(lesson)
            if result:
                data.append(result)

        print(f"[GraphLoader] Loaded {len(data)}/{len(lessons)} graphs successfully")
        return data

    def _load_graph_from_code(
        self,
        code: str,
        cache_key: str
    ) -> GraphData:
        """
        Load graph from source code.

        Args:
            code: Source code to parse
            cache_key: Key for caching

        Returns:
            PyTorch Geometric Data object
        """
        # Check cache
        if self.cache_graphs and cache_key in self._cache:
            return self._cache[cache_key]

        # Parse code to graph
        graph = create_graph_data_from_source(
            code,
            embedder=self.embedder
        )

        # Cache if enabled
        if self.cache_graphs:
            self._cache[cache_key] = graph

        return graph

    def clear_cache(self):
        """Clear cached graphs"""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_graphs': len(self._cache),
            'cache_size_mb': sum(
                graph.num_nodes * graph.num_node_features * 4  # 4 bytes per float32
                for graph in self._cache.values()
            ) / (1024 * 1024)
        }


# Convenience functions

def load_graphs_from_experiences(
    experiences: List[Experience],
    use_embeddings: bool = True
) -> List[TrainingExample]:
    """
    Convenience function to load graphs from experiences.

    Args:
        experiences: List of Experience objects
        use_embeddings: Use semantic embeddings

    Returns:
        List of (graph, label) tuples
    """
    loader = GraphLoader(use_semantic_embeddings=use_embeddings)
    return loader.load_from_experiences(experiences)


def load_graphs_from_lessons(
    lessons: List[Dict[str, Any]],
    use_embeddings: bool = True
) -> List[TrainingExample]:
    """
    Convenience function to load graphs from lessons.

    Args:
        lessons: List of lesson dictionaries
        use_embeddings: Use semantic embeddings

    Returns:
        List of (graph, label) tuples
    """
    loader = GraphLoader(use_semantic_embeddings=use_embeddings)
    return loader.load_from_lessons(lessons)


# Example usage
if __name__ == "__main__":
    # Test loading
    loader = GraphLoader()

    # Test code
    test_code = """
def hello(name):
    if not name:
        raise ValueError("Name required")
    return f"Hello, {name}"
"""

    # Create fake experience
    from nerion_digital_physicist.infrastructure.memory import Experience
    exp = Experience(
        experience_id="test_001",
        task_id="test",
        template_id="test",
        status="solved",
        metadata={'source_code': test_code}
    )

    # Load graph
    result = loader.load_from_experience(exp)
    if result:
        graph, label = result
        print(f"Loaded graph: {graph.num_nodes} nodes, {graph.num_edges} edges, label={label}")
    else:
        print("Failed to load graph")
