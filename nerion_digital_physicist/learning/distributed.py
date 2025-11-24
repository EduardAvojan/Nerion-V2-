"""
Distributed Learning

Enables knowledge sharing and distributed learning across multiple agents.
Agents can share experiences, models, and learned patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import hashlib
import json


@dataclass
class KnowledgeItem:
    """A piece of knowledge that can be shared"""
    knowledge_id: str = field(default_factory=lambda: hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16])
    source_agent_id: str = ""
    knowledge_type: str = ""  # 'pattern', 'rule', 'model_weights', 'experience'
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # How confident the source agent is
    success_rate: float = 0.0  # Historical success rate
    usage_count: int = 0  # How many times this was used
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class LearningExperience:
    """An experience that contributed to learning"""
    experience_id: str = field(default_factory=lambda: hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16])
    agent_id: str = ""
    task_type: str = ""
    code: str = ""
    action_taken: str = ""
    outcome: str = ""  # 'success' or 'failure'
    reward: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBase:
    """
    Shared knowledge base for distributed learning.

    Stores knowledge items, experiences, and enables knowledge retrieval
    and sharing across agents.
    """

    def __init__(self):
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.experiences: Dict[str, LearningExperience] = {}

        # Index by agent
        self.knowledge_by_agent: Dict[str, Set[str]] = {}

        # Index by type
        self.knowledge_by_type: Dict[str, Set[str]] = {}

        # Statistics
        self.total_shares = 0
        self.total_accesses = 0

    def add_knowledge(self, item: KnowledgeItem):
        """Add knowledge item to the base"""
        self.knowledge_items[item.knowledge_id] = item

        # Update indices
        if item.source_agent_id not in self.knowledge_by_agent:
            self.knowledge_by_agent[item.source_agent_id] = set()
        self.knowledge_by_agent[item.source_agent_id].add(item.knowledge_id)

        if item.knowledge_type not in self.knowledge_by_type:
            self.knowledge_by_type[item.knowledge_type] = set()
        self.knowledge_by_type[item.knowledge_type].add(item.knowledge_id)

        self.total_shares += 1

    def add_experience(self, experience: LearningExperience):
        """Add learning experience"""
        self.experiences[experience.experience_id] = experience

    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Retrieve knowledge by ID"""
        item = self.knowledge_items.get(knowledge_id)
        if item:
            item.usage_count += 1
            item.last_updated = datetime.now()
            self.total_accesses += 1
        return item

    def query_knowledge(
        self,
        knowledge_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        min_success_rate: float = 0.0
    ) -> List[KnowledgeItem]:
        """
        Query knowledge base with filters.

        Args:
            knowledge_type: Filter by knowledge type
            tags: Filter by tags (match any)
            min_confidence: Minimum confidence threshold
            min_success_rate: Minimum success rate threshold

        Returns:
            List of matching knowledge items
        """
        candidates = self.knowledge_items.values()

        # Filter by type
        if knowledge_type:
            candidates = [
                k for k in candidates
                if k.knowledge_type == knowledge_type
            ]

        # Filter by tags
        if tags:
            candidates = [
                k for k in candidates
                if any(tag in k.tags for tag in tags)
            ]

        # Filter by confidence
        candidates = [
            k for k in candidates
            if k.confidence >= min_confidence
        ]

        # Filter by success rate
        candidates = [
            k for k in candidates
            if k.success_rate >= min_success_rate
        ]

        # Sort by (success_rate * confidence) descending
        candidates.sort(
            key=lambda k: k.success_rate * k.confidence,
            reverse=True
        )

        return list(candidates)

    def get_agent_knowledge(self, agent_id: str) -> List[KnowledgeItem]:
        """Get all knowledge contributed by an agent"""
        knowledge_ids = self.knowledge_by_agent.get(agent_id, set())
        return [self.knowledge_items[kid] for kid in knowledge_ids if kid in self.knowledge_items]

    def get_top_knowledge(self, n: int = 10) -> List[KnowledgeItem]:
        """Get top N most valuable knowledge items"""
        items = list(self.knowledge_items.values())
        items.sort(
            key=lambda k: (k.success_rate * k.confidence * (1 + k.usage_count)),
            reverse=True
        )
        return items[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            'total_knowledge_items': len(self.knowledge_items),
            'total_experiences': len(self.experiences),
            'total_shares': self.total_shares,
            'total_accesses': self.total_accesses,
            'knowledge_by_type': {
                ktype: len(items)
                for ktype, items in self.knowledge_by_type.items()
            },
            'active_agents': len(self.knowledge_by_agent)
        }


class DistributedLearner:
    """
    Manages distributed learning across multiple agents.

    Coordinates knowledge sharing, experience aggregation, and
    collective learning.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()

        # Learning parameters
        self.learning_rate = 0.01
        self.experience_replay_size = 1000

        # Synchronization
        self.last_sync: Dict[str, datetime] = {}

    def share_knowledge(
        self,
        agent_id: str,
        knowledge_type: str,
        content: Dict[str, Any],
        confidence: float,
        success_rate: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Agent shares knowledge with the collective.

        Returns:
            Knowledge ID
        """
        item = KnowledgeItem(
            source_agent_id=agent_id,
            knowledge_type=knowledge_type,
            content=content,
            confidence=confidence,
            success_rate=success_rate,
            tags=tags or []
        )

        self.knowledge_base.add_knowledge(item)
        return item.knowledge_id

    def record_experience(
        self,
        agent_id: str,
        task_type: str,
        code: str,
        action_taken: str,
        outcome: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record an agent's learning experience.

        Returns:
            Experience ID
        """
        experience = LearningExperience(
            agent_id=agent_id,
            task_type=task_type,
            code=code,
            action_taken=action_taken,
            outcome=outcome,
            reward=reward,
            metadata=metadata or {}
        )

        self.knowledge_base.add_experience(experience)
        return experience.experience_id

    def request_knowledge(
        self,
        agent_id: str,
        knowledge_type: str,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> List[KnowledgeItem]:
        """
        Agent requests knowledge from collective.

        Args:
            agent_id: Requesting agent
            knowledge_type: Type of knowledge needed
            tags: Optional tags to filter by
            min_confidence: Minimum confidence threshold

        Returns:
            List of relevant knowledge items
        """
        items = self.knowledge_base.query_knowledge(
            knowledge_type=knowledge_type,
            tags=tags,
            min_confidence=min_confidence
        )

        # Exclude knowledge from the requesting agent itself
        items = [item for item in items if item.source_agent_id != agent_id]

        return items

    def aggregate_experiences(
        self,
        task_type: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate experiences to extract collective learning.

        Args:
            task_type: Filter by task type
            outcome: Filter by outcome ('success' or 'failure')

        Returns:
            Aggregated insights
        """
        experiences = list(self.knowledge_base.experiences.values())

        # Filter
        if task_type:
            experiences = [e for e in experiences if e.task_type == task_type]
        if outcome:
            experiences = [e for e in experiences if e.outcome == outcome]

        if not experiences:
            return {'total_experiences': 0}

        # Aggregate
        total_reward = sum(e.reward for e in experiences)
        avg_reward = total_reward / len(experiences)

        success_count = sum(1 for e in experiences if e.outcome == 'success')
        success_rate = success_count / len(experiences)

        # Extract common patterns
        actions = {}
        for exp in experiences:
            action = exp.action_taken
            if action not in actions:
                actions[action] = {'count': 0, 'total_reward': 0.0, 'successes': 0}
            actions[action]['count'] += 1
            actions[action]['total_reward'] += exp.reward
            if exp.outcome == 'success':
                actions[action]['successes'] += 1

        # Find best actions
        best_actions = []
        for action, stats in actions.items():
            avg_action_reward = stats['total_reward'] / stats['count']
            action_success_rate = stats['successes'] / stats['count']
            best_actions.append({
                'action': action,
                'count': stats['count'],
                'avg_reward': avg_action_reward,
                'success_rate': action_success_rate
            })

        best_actions.sort(key=lambda x: x['avg_reward'], reverse=True)

        return {
            'total_experiences': len(experiences),
            'avg_reward': avg_reward,
            'overall_success_rate': success_rate,
            'best_actions': best_actions[:5]  # Top 5
        }

    def synchronize_agent(
        self,
        agent_id: str,
        agent_knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronize an agent's knowledge with collective.

        Args:
            agent_id: Agent to sync
            agent_knowledge: Agent's current knowledge

        Returns:
            Updated knowledge from collective
        """
        # Record sync time
        self.last_sync[agent_id] = datetime.now()

        # Get knowledge since last sync
        # (In a real system, would track deltas)

        # For now, return top knowledge relevant to agent
        top_knowledge = self.knowledge_base.get_top_knowledge(n=10)

        return {
            'top_knowledge': [
                {
                    'id': k.knowledge_id,
                    'type': k.knowledge_type,
                    'content': k.content,
                    'confidence': k.confidence,
                    'source': k.source_agent_id
                }
                for k in top_knowledge
            ],
            'sync_timestamp': datetime.now().isoformat()
        }

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get distributed learning statistics"""
        kb_stats = self.knowledge_base.get_statistics()

        # Add distributed learning specific stats
        total_agents_synced = len(self.last_sync)

        return {
            **kb_stats,
            'total_agents_synced': total_agents_synced,
            'learning_rate': self.learning_rate,
            'experience_replay_size': self.experience_replay_size
        }


# Example usage
if __name__ == "__main__":
    print("=== Distributed Learning Demo ===\n")

    learner = DistributedLearner()

    # Agent 1 shares knowledge
    print("=== Agent 1 shares security pattern ===")
    k1_id = learner.share_knowledge(
        agent_id="security_001",
        knowledge_type="security_pattern",
        content={'pattern': 'hardcoded_credentials', 'regex': r"password\s*=\s*['\"]"},
        confidence=0.95,
        success_rate=0.98,
        tags=['security', 'credentials']
    )
    print(f"Shared knowledge: {k1_id}")

    # Agent 2 shares performance pattern
    print("\n=== Agent 2 shares performance pattern ===")
    k2_id = learner.share_knowledge(
        agent_id="perf_001",
        knowledge_type="performance_pattern",
        content={'pattern': 'nested_loops', 'severity': 'medium'},
        confidence=0.85,
        success_rate=0.90,
        tags=['performance', 'complexity']
    )
    print(f"Shared knowledge: {k2_id}")

    # Agent 3 requests security knowledge
    print("\n=== Agent 3 requests security knowledge ===")
    knowledge = learner.request_knowledge(
        agent_id="generalist_001",
        knowledge_type="security_pattern",
        min_confidence=0.8
    )
    print(f"Found {len(knowledge)} relevant items")
    for item in knowledge:
        print(f"  - From {item.source_agent_id}: {item.content['pattern']}")

    # Record experiences
    print("\n=== Recording experiences ===")
    for i in range(5):
        learner.record_experience(
            agent_id=f"agent_{i % 3}",
            task_type="bug_fixing",
            code="sample_code",
            action_taken=f"action_{i % 2}",
            outcome='success' if i % 2 == 0 else 'failure',
            reward=1.0 if i % 2 == 0 else -0.5
        )

    # Aggregate experiences
    print("\n=== Aggregating experiences ===")
    insights = learner.aggregate_experiences(task_type="bug_fixing")
    print(f"Total experiences: {insights['total_experiences']}")
    print(f"Success rate: {insights['overall_success_rate']:.2%}")
    print(f"Best actions:")
    for action in insights['best_actions']:
        print(f"  - {action['action']}: {action['success_rate']:.2%} success")

    # Show statistics
    print("\n=== Statistics ===")
    stats = learner.get_learning_statistics()
    print(f"Total knowledge items: {stats['total_knowledge_items']}")
    print(f"Total experiences: {stats['total_experiences']}")
    print(f"Total shares: {stats['total_shares']}")
    print(f"Active agents: {stats['active_agents']}")
