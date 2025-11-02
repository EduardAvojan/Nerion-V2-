"""
Multi-Agent Coordinator

Coordinates multiple specialist agents, assigns tasks, resolves conflicts,
and aggregates results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from .protocol import (
    AgentMessage,
    MessageType,
    TaskRequest,
    TaskResponse,
    Proposal,
    Vote,
    Conflict,
    ConflictType,
    CoordinationStrategy,
    CoordinationPlan,
    AgentRole
)
from .specialists import SpecialistAgent


@dataclass
class AgentRegistry:
    """Registry of available agents"""
    agents: Dict[str, SpecialistAgent] = field(default_factory=dict)
    agents_by_role: Dict[AgentRole, List[str]] = field(default_factory=dict)

    def register_agent(self, agent: SpecialistAgent):
        """Register an agent"""
        self.agents[agent.agent_id] = agent

        if agent.role not in self.agents_by_role:
            self.agents_by_role[agent.role] = []
        self.agents_by_role[agent.role].append(agent.agent_id)

    def get_agent(self, agent_id: str) -> Optional[SpecialistAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_role(self, role: AgentRole) -> List[SpecialistAgent]:
        """Get all agents with a specific role"""
        agent_ids = self.agents_by_role.get(role, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def get_all_agents(self) -> List[SpecialistAgent]:
        """Get all registered agents"""
        return list(self.agents.values())


class MultiAgentCoordinator:
    """
    Coordinates multiple specialist agents.

    Responsibilities:
    - Task assignment to appropriate agents
    - Conflict resolution
    - Result aggregation
    - Performance monitoring
    - Knowledge sharing coordination
    """

    def __init__(self, coordinator_id: str = "coordinator_001"):
        self.coordinator_id = coordinator_id
        self.registry = AgentRegistry()

        # Task tracking
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.completed_tasks: Dict[str, List[TaskResponse]] = {}

        # Conflict management
        self.active_conflicts: List[Conflict] = []
        self.resolved_conflicts: List[Conflict] = []

        # Message queue
        self.message_queue: List[AgentMessage] = []

        # Statistics
        self.total_tasks_coordinated = 0
        self.total_conflicts_resolved = 0

    def register_agent(self, agent: SpecialistAgent):
        """Register a specialist agent"""
        self.registry.register_agent(agent)

    def assign_task(
        self,
        task: TaskRequest,
        strategy: CoordinationStrategy = CoordinationStrategy.PARALLEL
    ) -> List[TaskResponse]:
        """
        Assign task to appropriate agents.

        Args:
            task: Task to execute
            strategy: How to coordinate agents

        Returns:
            List of responses from agents
        """
        self.total_tasks_coordinated += 1
        self.active_tasks[task.task_id] = task

        # Find capable agents
        capable_agents = self._find_capable_agents(task)

        if not capable_agents:
            # No agents can handle this task
            return [TaskResponse(
                task_id=task.task_id,
                success=False,
                result={},
                confidence=0.0,
                errors=["No capable agents found"],
                responder_id=self.coordinator_id
            )]

        # Execute based on strategy
        if strategy == CoordinationStrategy.PARALLEL:
            responses = self._execute_parallel(task, capable_agents)
        elif strategy == CoordinationStrategy.SEQUENTIAL:
            responses = self._execute_sequential(task, capable_agents)
        elif strategy == CoordinationStrategy.VOTING:
            responses = self._execute_voting(task, capable_agents)
        elif strategy == CoordinationStrategy.CONSENSUS:
            responses = self._execute_consensus(task, capable_agents)
        else:
            # Default to parallel
            responses = self._execute_parallel(task, capable_agents)

        # Store responses
        self.completed_tasks[task.task_id] = responses

        # Remove from active
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]

        return responses

    def _find_capable_agents(
        self,
        task: TaskRequest,
        min_confidence: float = 0.3
    ) -> List[Tuple[SpecialistAgent, float]]:
        """
        Find agents capable of handling task.

        Returns:
            List of (agent, confidence) tuples sorted by confidence
        """
        capable = []

        for agent in self.registry.get_all_agents():
            confidence = agent.can_handle(task)
            if confidence >= min_confidence:
                capable.append((agent, confidence))

        # Sort by confidence (highest first)
        capable.sort(key=lambda x: x[1], reverse=True)

        return capable

    def _execute_parallel(
        self,
        task: TaskRequest,
        agents: List[Tuple[SpecialistAgent, float]]
    ) -> List[TaskResponse]:
        """Execute task in parallel (all agents work simultaneously)"""
        responses = []

        for agent, confidence in agents:
            response = agent.execute_task(task)
            responses.append(response)

        return responses

    def _execute_sequential(
        self,
        task: TaskRequest,
        agents: List[Tuple[SpecialistAgent, float]]
    ) -> List[TaskResponse]:
        """Execute task sequentially (stop when one succeeds with high confidence)"""
        responses = []

        for agent, capability_confidence in agents:
            response = agent.execute_task(task)
            responses.append(response)

            # Stop if successful with high confidence
            if response.success and response.confidence >= 0.8:
                break

        return responses

    def _execute_voting(
        self,
        task: TaskRequest,
        agents: List[Tuple[SpecialistAgent, float]]
    ) -> List[TaskResponse]:
        """Execute task with voting (all agents vote on best solution)"""
        # All agents execute
        responses = self._execute_parallel(task, agents)

        # Find most agreed-upon solution
        # For simplicity, use highest confidence response
        if responses:
            best_response = max(responses, key=lambda r: r.confidence)
            return [best_response]

        return responses

    def _execute_consensus(
        self,
        task: TaskRequest,
        agents: List[Tuple[SpecialistAgent, float]]
    ) -> List[TaskResponse]:
        """Execute task requiring consensus (all agents must agree)"""
        responses = self._execute_parallel(task, agents)

        # Check if all agents succeeded
        all_success = all(r.success for r in responses)

        if all_success:
            # Return aggregated response
            return [TaskResponse(
                task_id=task.task_id,
                success=True,
                result={'consensus': True, 'all_responses': [r.result for r in responses]},
                confidence=min(r.confidence for r in responses),  # Weakest link
                execution_time=max(r.execution_time for r in responses),
                responder_id=self.coordinator_id
            )]
        else:
            # Consensus failed
            return [TaskResponse(
                task_id=task.task_id,
                success=False,
                result={'consensus': False},
                confidence=0.0,
                errors=["Consensus not reached"],
                responder_id=self.coordinator_id
            )]

    def aggregate_responses(
        self,
        responses: List[TaskResponse],
        method: str = 'weighted_average'
    ) -> TaskResponse:
        """
        Aggregate multiple responses into one.

        Args:
            responses: Responses to aggregate
            method: 'weighted_average', 'majority_vote', 'highest_confidence'

        Returns:
            Aggregated response
        """
        if not responses:
            return TaskResponse(
                success=False,
                errors=["No responses to aggregate"],
                responder_id=self.coordinator_id
            )

        if method == 'highest_confidence':
            return max(responses, key=lambda r: r.confidence)

        elif method == 'weighted_average':
            # Weight by confidence
            total_weight = sum(r.confidence for r in responses)
            if total_weight == 0:
                return responses[0]

            # Aggregate results (simplified - just merge dicts)
            aggregated_result = {}
            for response in responses:
                weight = response.confidence / total_weight
                for key, value in response.result.items():
                    if key not in aggregated_result:
                        aggregated_result[key] = []
                    aggregated_result[key].append((value, weight))

            return TaskResponse(
                task_id=responses[0].task_id,
                success=any(r.success for r in responses),
                result=aggregated_result,
                confidence=sum(r.confidence for r in responses) / len(responses),
                execution_time=max(r.execution_time for r in responses),
                responder_id=self.coordinator_id
            )

        else:  # majority_vote
            # Find most common result
            success_count = sum(1 for r in responses if r.success)
            return TaskResponse(
                task_id=responses[0].task_id,
                success=success_count > len(responses) / 2,
                result={'majority_success': success_count > len(responses) / 2},
                confidence=success_count / len(responses),
                execution_time=sum(r.execution_time for r in responses) / len(responses),
                responder_id=self.coordinator_id
            )

    def detect_conflict(self, responses: List[TaskResponse]) -> Optional[Conflict]:
        """Detect conflicts between agent responses"""
        if len(responses) < 2:
            return None

        # Check for disagreement
        success_count = sum(1 for r in responses if r.success)
        failure_count = len(responses) - success_count

        # If agents disagree significantly, it's a conflict
        if success_count > 0 and failure_count > 0:
            conflict = Conflict(
                conflict_type=ConflictType.DISAGREEMENT,
                agents_involved=[r.responder_id for r in responses],
                description=f"{success_count} agents succeeded, {failure_count} failed",
                resolved=False
            )
            self.active_conflicts.append(conflict)
            return conflict

        return None

    def resolve_conflict(self, conflict: Conflict) -> str:
        """
        Resolve a conflict.

        Returns:
            Resolution decision
        """
        # Simple resolution: Use highest confidence response
        # In production, could be more sophisticated
        resolution = "Use highest confidence agent response"

        conflict.resolution = resolution
        conflict.resolved = True

        self.active_conflicts.remove(conflict)
        self.resolved_conflicts.append(conflict)
        self.total_conflicts_resolved += 1

        return resolution

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        total_agents = len(self.registry.agents)

        # Aggregate agent stats
        agent_stats = []
        for agent in self.registry.get_all_agents():
            agent_stats.append(agent.get_statistics())

        return {
            'coordinator_id': self.coordinator_id,
            'total_agents': total_agents,
            'agents_by_role': {
                role.value: len(agents)
                for role, agents in self.registry.agents_by_role.items()
            },
            'total_tasks_coordinated': self.total_tasks_coordinated,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'active_conflicts': len(self.active_conflicts),
            'total_conflicts_resolved': self.total_conflicts_resolved,
            'agent_statistics': agent_stats
        }


# Example usage
if __name__ == "__main__":
    from .specialists import PythonSpecialist, SecuritySpecialist, PerformanceSpecialist

    print("=== Multi-Agent Coordinator Demo ===\n")

    # Create coordinator
    coordinator = MultiAgentCoordinator()

    # Register agents
    python_agent = PythonSpecialist("python_001")
    security_agent = SecuritySpecialist("security_001")
    perf_agent = PerformanceSpecialist("perf_001")

    coordinator.register_agent(python_agent)
    coordinator.register_agent(security_agent)
    coordinator.register_agent(perf_agent)

    print(f"Registered {len(coordinator.registry.agents)} agents")

    # Create task
    task = TaskRequest(
        task_type="analyze_code",
        code="password = 'hardcoded123'\ndef slow_function():\n    for i in range(1000):\n        for j in range(1000):\n            pass",
        language="python",
        requester_id="user_001"
    )

    # Assign task with parallel strategy
    print("\n=== Parallel Execution ===")
    responses = coordinator.assign_task(task, strategy=CoordinationStrategy.PARALLEL)
    print(f"Received {len(responses)} responses")

    for i, response in enumerate(responses, 1):
        print(f"\nResponse {i}:")
        print(f"  Agent: {response.responder_id}")
        print(f"  Success: {response.success}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Time: {response.execution_time:.3f}s")

    # Aggregate responses
    print("\n=== Aggregated Result ===")
    aggregated = coordinator.aggregate_responses(responses, method='highest_confidence')
    print(f"Aggregated confidence: {aggregated.confidence:.2f}")
    print(f"Aggregated success: {aggregated.success}")

    # Detect conflicts
    conflict = coordinator.detect_conflict(responses)
    if conflict:
        print(f"\n=== Conflict Detected ===")
        print(f"Type: {conflict.conflict_type.value}")
        print(f"Agents: {conflict.agents_involved}")
        resolution = coordinator.resolve_conflict(conflict)
        print(f"Resolution: {resolution}")

    # Show statistics
    print("\n=== Statistics ===")
    stats = coordinator.get_statistics()
    print(f"Total tasks coordinated: {stats['total_tasks_coordinated']}")
    print(f"Total agents: {stats['total_agents']}")
    print(f"Conflicts resolved: {stats['total_conflicts_resolved']}")
