"""
Hierarchical Episodic Memory with Prioritized Replay

Stores specific code modification experiences for learning from rare events,
analogical reasoning, and anti-forgetting during continuous learning.

Key Concepts:
- Episode: Complete interaction sequence (observation → action → outcome)
- Priority: High-value episodes (surprising, rare, impactful) sampled more
- Consolidation: Extract general principles from specific episodes
- Analogical reasoning: Find similar past experiences

Integration with Nerion:
- Extends existing ReplayStore infrastructure
- Feeds experience replay for continuous learning
- Stores production bug experiences for anti-forgetting
- Enables learning from rare/surprising events
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Use existing ReplayStore infrastructure
from nerion_digital_physicist.infrastructure.memory import ReplayStore, Experience


class EpisodeType(Enum):
    """Types of episodes"""
    CODE_MODIFICATION = "code_modification"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    FEATURE_ADDITION = "feature_addition"
    TEST_GENERATION = "test_generation"
    PRODUCTION_FAILURE = "production_failure"


class EpisodeOutcome(Enum):
    """Episode outcome"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    REVERTED = "reverted"


@dataclass
class Episode:
    """A complete interaction episode"""
    episode_id: str
    episode_type: EpisodeType

    # Observation
    task: str
    code_before: str
    context: Dict[str, Any]

    # Action
    action_taken: str

    # Outcome
    code_after: str

    # Optional fields with defaults
    reasoning: Optional[str] = None
    outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS
    test_results: Optional[Dict[str, Any]] = None

    # Metrics
    surprise: float = 0.0              # How unexpected was the outcome?
    impact: float = 0.0                # How significant was the change?
    rarity: float = 0.0                # How unusual is this pattern?

    # Priority (auto-calculated)
    priority: float = 0.0

    # Metadata
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    learned_from: bool = False         # Has this been used in training?

    def calculate_priority(self):
        """Calculate priority from surprise, impact, and rarity"""
        # Weighted combination
        self.priority = (
            0.4 * self.surprise +
            0.3 * self.impact +
            0.3 * self.rarity
        )


@dataclass
class ConsolidatedPrinciple:
    """A general principle extracted from multiple episodes"""
    principle_id: str
    description: str
    source_episodes: List[str]       # Episode IDs
    confidence: float
    applicability: str               # "general", "domain-specific", "language-specific"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class EpisodicMemory:
    """
    Hierarchical episodic memory with prioritized replay.

    Features:
    - Store complete interaction episodes
    - Priority-based sampling (focus on high-value experiences)
    - Analogical reasoning (find similar past experiences)
    - Memory consolidation (extract principles)
    - Integration with continuous learning

    Usage:
        >>> memory = EpisodicMemory(Path("data/episodic_memory"))
        >>>
        >>> # Store episode
        >>> episode = Episode(
        ...     episode_id="ep_001",
        ...     episode_type=EpisodeType.BUG_FIX,
        ...     task="Fix null pointer exception",
        ...     code_before=buggy_code,
        ...     action_taken="Add null check",
        ...     code_after=fixed_code,
        ...     outcome=EpisodeOutcome.SUCCESS,
        ...     surprise=0.8,  # High surprise
        ...     impact=0.9     # High impact
        ... )
        >>> memory.store_episode(episode)
        >>>
        >>> # Recall similar episodes
        >>> similar = memory.recall_similar(query_episode, k=5)
    """

    def __init__(
        self,
        storage_path: Path,
        max_episodes: int = 10000,
        replay_store: Optional[ReplayStore] = None
    ):
        """
        Initialize episodic memory.

        Args:
            storage_path: Path to storage directory
            max_episodes: Maximum episodes to store (LRU eviction)
            replay_store: Optional ReplayStore for integration
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.max_episodes = max_episodes
        self.replay_store = replay_store

        # Storage files
        self.episodes_file = self.storage_path / "episodes.jsonl"
        self.principles_file = self.storage_path / "principles.json"
        self.index_file = self.storage_path / "index.json"

        # In-memory index for fast lookup
        self.episode_index: Dict[str, Episode] = {}
        self.tag_index: Dict[str, List[str]] = defaultdict(list)  # tag -> episode_ids
        self.type_index: Dict[EpisodeType, List[str]] = defaultdict(list)

        # Consolidated principles
        self.principles: Dict[str, ConsolidatedPrinciple] = {}

        # Load existing data
        self._load_from_disk()

        print(f"[EpisodicMemory] Initialized with {len(self.episode_index)} episodes")

    def store_episode(
        self,
        episode: Episode,
        sync_to_replay_store: bool = True
    ):
        """
        Store an episode in memory.

        Args:
            episode: Episode to store
            sync_to_replay_store: Also store in ReplayStore if available
        """
        # Calculate priority
        episode.calculate_priority()

        # Store in index
        self.episode_index[episode.episode_id] = episode

        # Update tag index
        for tag in episode.tags:
            self.tag_index[tag].append(episode.episode_id)

        # Update type index
        self.type_index[episode.episode_type].append(episode.episode_id)

        # Evict if over capacity
        if len(self.episode_index) > self.max_episodes:
            self._evict_lowest_priority()

        # Sync to ReplayStore if available
        if sync_to_replay_store and self.replay_store:
            self._sync_to_replay_store(episode)

        # Persist to disk
        self._append_to_disk(episode)

        print(f"[EpisodicMemory] Stored episode {episode.episode_id} "
              f"(priority={episode.priority:.3f})")

    def recall_similar(
        self,
        query_episode: Episode,
        k: int = 5,
        filter_type: Optional[EpisodeType] = None
    ) -> List[Episode]:
        """
        Recall similar episodes (analogical reasoning).

        Args:
            query_episode: Query episode
            k: Number of similar episodes to return
            filter_type: Optional filter by episode type

        Returns:
            List of k most similar episodes
        """
        candidates = list(self.episode_index.values())

        # Filter by type if specified
        if filter_type:
            candidates = [e for e in candidates if e.episode_type == filter_type]

        # Calculate similarity scores
        similarities = []
        for candidate in candidates:
            if candidate.episode_id == query_episode.episode_id:
                continue  # Skip self

            similarity = self._calculate_similarity(query_episode, candidate)
            similarities.append((similarity, candidate))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        return [episode for _, episode in similarities[:k]]

    def recall_by_priority(
        self,
        k: int = 10,
        episode_type: Optional[EpisodeType] = None
    ) -> List[Episode]:
        """
        Recall high-priority episodes for replay learning.

        Args:
            k: Number of episodes to return
            episode_type: Optional filter by type

        Returns:
            List of k highest priority episodes
        """
        candidates = list(self.episode_index.values())

        # Filter by type if specified
        if episode_type:
            candidates = [e for e in candidates if e.episode_type == episode_type]

        # Sort by priority (descending)
        candidates.sort(key=lambda e: e.priority, reverse=True)

        return candidates[:k]

    def recall_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> List[Episode]:
        """
        Recall episodes by tags.

        Args:
            tags: Tags to match
            match_all: If True, require all tags; if False, match any tag

        Returns:
            List of matching episodes
        """
        if match_all:
            # Intersection of all tag lists
            episode_ids = set(self.tag_index[tags[0]])
            for tag in tags[1:]:
                episode_ids &= set(self.tag_index[tag])
        else:
            # Union of all tag lists
            episode_ids = set()
            for tag in tags:
                episode_ids |= set(self.tag_index[tag])

        return [self.episode_index[eid] for eid in episode_ids if eid in self.episode_index]

    def consolidate_memory(
        self,
        min_episodes_per_principle: int = 5
    ) -> List[ConsolidatedPrinciple]:
        """
        Consolidate episodes into general principles.

        Extract patterns from multiple similar episodes to create
        reusable principles for future decision-making.

        Args:
            min_episodes_per_principle: Minimum episodes required

        Returns:
            List of extracted principles
        """
        print(f"[EpisodicMemory] Consolidating memory...")

        # Group episodes by type and tags
        episode_groups = self._group_episodes_for_consolidation()

        new_principles = []
        for group_key, episodes in episode_groups.items():
            if len(episodes) < min_episodes_per_principle:
                continue

            # Extract principle from group
            principle = self._extract_principle(group_key, episodes)
            if principle:
                self.principles[principle.principle_id] = principle
                new_principles.append(principle)

        # Save principles
        self._save_principles()

        print(f"[EpisodicMemory] Extracted {len(new_principles)} new principles")
        return new_principles

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        total_episodes = len(self.episode_index)

        if total_episodes == 0:
            return {'total_episodes': 0}

        # Distribution by type
        type_distribution = {
            etype.value: len(self.type_index[etype])
            for etype in EpisodeType
        }

        # Distribution by outcome
        outcome_distribution = defaultdict(int)
        for episode in self.episode_index.values():
            outcome_distribution[episode.outcome.value] += 1

        # Average metrics
        avg_surprise = sum(e.surprise for e in self.episode_index.values()) / total_episodes
        avg_impact = sum(e.impact for e in self.episode_index.values()) / total_episodes
        avg_rarity = sum(e.rarity for e in self.episode_index.values()) / total_episodes
        avg_priority = sum(e.priority for e in self.episode_index.values()) / total_episodes

        # Learning progress
        learned_count = sum(1 for e in self.episode_index.values() if e.learned_from)
        learning_rate = learned_count / total_episodes

        return {
            'total_episodes': total_episodes,
            'max_capacity': self.max_episodes,
            'utilization': total_episodes / self.max_episodes,
            'type_distribution': type_distribution,
            'outcome_distribution': dict(outcome_distribution),
            'avg_surprise': avg_surprise,
            'avg_impact': avg_impact,
            'avg_rarity': avg_rarity,
            'avg_priority': avg_priority,
            'learned_from_rate': learning_rate,
            'total_principles': len(self.principles)
        }

    def _calculate_similarity(
        self,
        episode1: Episode,
        episode2: Episode
    ) -> float:
        """
        Calculate similarity between two episodes.

        Uses combination of:
        - Type similarity (same type = 1.0)
        - Tag overlap (Jaccard similarity)
        - Task description similarity (simple keyword overlap)
        - Code similarity (simple token overlap)
        """
        similarity = 0.0

        # Type similarity (weight: 0.2)
        if episode1.episode_type == episode2.episode_type:
            similarity += 0.2

        # Tag overlap (weight: 0.3)
        tags1 = set(episode1.tags)
        tags2 = set(episode2.tags)
        if tags1 or tags2:
            tag_similarity = len(tags1 & tags2) / len(tags1 | tags2)
            similarity += 0.3 * tag_similarity

        # Task similarity (weight: 0.25)
        task1_tokens = set(episode1.task.lower().split())
        task2_tokens = set(episode2.task.lower().split())
        if task1_tokens or task2_tokens:
            task_similarity = len(task1_tokens & task2_tokens) / len(task1_tokens | task2_tokens)
            similarity += 0.25 * task_similarity

        # Code similarity (weight: 0.25)
        # Simplified: use token overlap
        code1_tokens = set(episode1.code_before.split())
        code2_tokens = set(episode2.code_before.split())
        if code1_tokens or code2_tokens:
            code_similarity = len(code1_tokens & code2_tokens) / len(code1_tokens | code2_tokens)
            similarity += 0.25 * code_similarity

        return similarity

    def _evict_lowest_priority(self):
        """Evict lowest priority episode when over capacity"""
        if not self.episode_index:
            return

        # Find lowest priority episode
        lowest_episode = min(
            self.episode_index.values(),
            key=lambda e: (e.priority, e.timestamp)  # Break ties by timestamp
        )

        # Remove from indices
        del self.episode_index[lowest_episode.episode_id]

        for tag in lowest_episode.tags:
            if lowest_episode.episode_id in self.tag_index[tag]:
                self.tag_index[tag].remove(lowest_episode.episode_id)

        if lowest_episode.episode_id in self.type_index[lowest_episode.episode_type]:
            self.type_index[lowest_episode.episode_type].remove(lowest_episode.episode_id)

        print(f"[EpisodicMemory] Evicted episode {lowest_episode.episode_id} "
              f"(priority={lowest_episode.priority:.3f})")

    def _sync_to_replay_store(self, episode: Episode):
        """Sync episode to ReplayStore for continuous learning"""
        if not self.replay_store:
            return

        # Convert episode to ReplayStore Experience format
        metadata = {
            'episode_type': episode.episode_type.value,
            'task': episode.task,
            'code_before': episode.code_before,
            'code_after': episode.code_after,
            'action_taken': episode.action_taken,
            'reasoning': episode.reasoning,
            'outcome': episode.outcome.value,
            'test_results': episode.test_results,
            'impact': episode.impact,
            'rarity': episode.rarity,
            'tags': episode.tags,
            'provenance': 'episodic_memory'
        }

        # Determine status from outcome
        status = "solved" if episode.outcome == EpisodeOutcome.SUCCESS else "failed"

        self.replay_store.append(
            task_id=episode.episode_id,
            template_id=episode.episode_type.value,
            status=status,
            surprise=episode.surprise,
            metadata=metadata
        )

    def _group_episodes_for_consolidation(self) -> Dict[str, List[Episode]]:
        """Group similar episodes for consolidation"""
        groups: Dict[str, List[Episode]] = defaultdict(list)

        for episode in self.episode_index.values():
            # Group by type + common tags
            group_key = f"{episode.episode_type.value}"
            if episode.tags:
                # Use most common tags
                group_key += ":" + ":".join(sorted(episode.tags[:3]))

            groups[group_key].append(episode)

        return groups

    def _extract_principle(
        self,
        group_key: str,
        episodes: List[Episode]
    ) -> Optional[ConsolidatedPrinciple]:
        """Extract principle from episode group"""
        # Calculate confidence based on consistency
        success_rate = sum(
            1 for e in episodes if e.outcome == EpisodeOutcome.SUCCESS
        ) / len(episodes)

        # Generate principle description (simplified - would use LLM in production)
        episode_type = episodes[0].episode_type
        avg_impact = sum(e.impact for e in episodes) / len(episodes)

        description = (
            f"When handling {episode_type.value}, "
            f"success rate is {success_rate:.0%} "
            f"with average impact {avg_impact:.2f}. "
            f"Based on {len(episodes)} episodes."
        )

        principle_id = hashlib.sha256(group_key.encode()).hexdigest()[:16]

        return ConsolidatedPrinciple(
            principle_id=principle_id,
            description=description,
            source_episodes=[e.episode_id for e in episodes],
            confidence=success_rate,
            applicability="domain-specific"
        )

    def _load_from_disk(self):
        """Load episodes from disk"""
        if not self.episodes_file.exists():
            return

        with open(self.episodes_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    episode = self._dict_to_episode(data)
                    self.episode_index[episode.episode_id] = episode

                    # Rebuild indices
                    for tag in episode.tags:
                        self.tag_index[tag].append(episode.episode_id)
                    self.type_index[episode.episode_type].append(episode.episode_id)
                except Exception as e:
                    print(f"[EpisodicMemory] Error loading episode: {e}")

        # Load principles
        if self.principles_file.exists():
            data = json.loads(self.principles_file.read_text())
            for p_data in data.get('principles', []):
                principle = ConsolidatedPrinciple(**p_data)
                self.principles[principle.principle_id] = principle

    def _append_to_disk(self, episode: Episode):
        """Append episode to disk storage"""
        with open(self.episodes_file, 'a') as f:
            f.write(json.dumps(self._episode_to_dict(episode)) + '\n')

    def _save_principles(self):
        """Save principles to disk"""
        data = {
            'principles': [asdict(p) for p in self.principles.values()]
        }
        self.principles_file.write_text(json.dumps(data, indent=2))

    def _episode_to_dict(self, episode: Episode) -> Dict[str, Any]:
        """Convert episode to dictionary"""
        d = asdict(episode)
        d['episode_type'] = episode.episode_type.value
        d['outcome'] = episode.outcome.value
        return d

    def _dict_to_episode(self, d: Dict[str, Any]) -> Episode:
        """Convert dictionary to episode"""
        d['episode_type'] = EpisodeType(d['episode_type'])
        d['outcome'] = EpisodeOutcome(d['outcome'])
        return Episode(**d)


# Integration example
def example_usage():
    """Example of episodic memory usage"""
    memory = EpisodicMemory(Path("data/episodic_memory"))

    # Store episode
    episode = Episode(
        episode_id="ep_001",
        episode_type=EpisodeType.BUG_FIX,
        task="Fix null pointer exception in authentication",
        code_before="def validate(token):\n    return decode(token)",
        action_taken="Add null check before decode",
        code_after="def validate(token):\n    if token is None:\n        raise ValueError\n    return decode(token)",
        outcome=EpisodeOutcome.SUCCESS,
        surprise=0.8,  # Model didn't expect this bug
        impact=0.9,    # Critical authentication bug
        rarity=0.6,    # Moderately rare pattern
        tags=['authentication', 'null_check', 'critical']
    )

    memory.store_episode(episode)

    # Recall similar episodes
    similar = memory.recall_similar(episode, k=5)
    print(f"Found {len(similar)} similar episodes")

    # Get statistics
    stats = memory.get_statistics()
    print(f"Memory statistics: {stats}")


if __name__ == "__main__":
    example_usage()
