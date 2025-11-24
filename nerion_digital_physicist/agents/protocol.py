"""
Agent Communication Protocol

Defines message types, communication patterns, and coordination protocols
for multi-agent collaboration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import uuid


class MessageType(Enum):
    """Types of messages agents can exchange"""
    TASK_REQUEST = "task_request"  # Request another agent to perform a task
    TASK_RESPONSE = "task_response"  # Response to a task request
    QUERY = "query"  # Query for information
    ANSWER = "answer"  # Answer to a query
    PROPOSAL = "proposal"  # Propose a solution/approach
    VOTE = "vote"  # Vote on a proposal
    DECISION = "decision"  # Final decision from coordinator
    STATUS_UPDATE = "status_update"  # Agent status update
    KNOWLEDGE_SHARE = "knowledge_share"  # Share learned knowledge
    CONFLICT = "conflict"  # Report a conflict
    ERROR = "error"  # Error notification


class AgentRole(Enum):
    """Agent specialization roles"""
    # Language specialists
    PYTHON_SPECIALIST = "python_specialist"
    JAVASCRIPT_SPECIALIST = "javascript_specialist"
    JAVA_SPECIALIST = "java_specialist"
    TYPESCRIPT_SPECIALIST = "typescript_specialist"

    # Domain specialists
    SECURITY_SPECIALIST = "security_specialist"
    PERFORMANCE_SPECIALIST = "performance_specialist"
    TESTING_SPECIALIST = "testing_specialist"
    DOCUMENTATION_SPECIALIST = "documentation_specialist"

    # Task specialists
    REFACTORING_SPECIALIST = "refactoring_specialist"
    BUG_FIXING_SPECIALIST = "bug_fixing_specialist"
    CODE_REVIEW_SPECIALIST = "code_review_specialist"

    # Coordinator
    COORDINATOR = "coordinator"

    # Generalist
    GENERALIST = "generalist"


@dataclass
class AgentMessage:
    """Message exchanged between agents"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # Can be "all" for broadcast
    message_type: MessageType = MessageType.QUERY
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    in_reply_to: Optional[str] = None  # ID of message this replies to
    priority: int = 5  # 1 (highest) to 10 (lowest)

    def is_broadcast(self) -> bool:
        """Check if message is broadcast to all agents"""
        return self.receiver_id == "all"

    def create_reply(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> AgentMessage:
        """Create a reply to this message"""
        return AgentMessage(
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=message_type,
            payload=payload,
            in_reply_to=self.message_id,
            priority=self.priority
        )


@dataclass
class TaskRequest:
    """Request for an agent to perform a task"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""  # e.g., "analyze_security", "optimize_performance"
    code: str = ""
    language: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    requester_id: str = ""


@dataclass
class TaskResponse:
    """Response to a task request"""
    task_id: str = ""
    success: bool = False
    result: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0.0 to 1.0
    execution_time: float = 0.0  # seconds
    errors: List[str] = field(default_factory=list)
    responder_id: str = ""


@dataclass
class Proposal:
    """Proposal from an agent"""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposer_id: str = ""
    task_id: str = ""
    solution: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0
    estimated_effort: float = 0.0  # hours
    votes_for: int = 0
    votes_against: int = 0


@dataclass
class Vote:
    """Vote on a proposal"""
    proposal_id: str = ""
    voter_id: str = ""
    vote: bool = True  # True = for, False = against
    reason: str = ""


class ConflictType(Enum):
    """Types of conflicts between agents"""
    DISAGREEMENT = "disagreement"  # Agents disagree on approach
    RESOURCE = "resource"  # Resource contention
    PRIORITY = "priority"  # Priority conflict
    CAPABILITY = "capability"  # Capability overlap/gap


@dataclass
class Conflict:
    """Conflict between agents"""
    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.DISAGREEMENT
    agents_involved: List[str] = field(default_factory=list)
    description: str = ""
    proposals: List[Proposal] = field(default_factory=list)
    resolution: Optional[str] = None
    resolved: bool = False


class CoordinationStrategy(Enum):
    """Strategies for coordinating multiple agents"""
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"  # Agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Tree-like delegation
    CONSENSUS = "consensus"  # All agents must agree
    VOTING = "voting"  # Majority vote
    AUCTION = "auction"  # Agents bid on tasks


@dataclass
class CoordinationPlan:
    """Plan for coordinating multiple agents on a task"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    agents_assigned: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> list of prerequisite agent_ids
    estimated_completion: Optional[datetime] = None


class ProtocolValidator:
    """Validates protocol compliance"""

    @staticmethod
    def validate_message(message: AgentMessage) -> bool:
        """Validate message structure"""
        if not message.sender_id:
            return False
        if not message.receiver_id:
            return False
        if not isinstance(message.message_type, MessageType):
            return False
        if message.priority < 1 or message.priority > 10:
            return False
        return True

    @staticmethod
    def validate_task_request(request: TaskRequest) -> bool:
        """Validate task request"""
        if not request.task_type:
            return False
        if not request.requester_id:
            return False
        return True

    @staticmethod
    def validate_proposal(proposal: Proposal) -> bool:
        """Validate proposal"""
        if not proposal.proposer_id:
            return False
        if not proposal.task_id:
            return False
        if proposal.confidence < 0.0 or proposal.confidence > 1.0:
            return False
        return True


# Example usage
if __name__ == "__main__":
    print("=== Multi-Agent Protocol Demo ===\n")

    # Create a task request
    task = TaskRequest(
        task_type="analyze_security",
        code="password = 'hardcoded123'",
        language="python",
        requester_id="coordinator_001"
    )

    # Create message with task request
    message = AgentMessage(
        sender_id="coordinator_001",
        receiver_id="security_specialist_001",
        message_type=MessageType.TASK_REQUEST,
        payload={"task": task.__dict__},
        priority=2
    )

    print(f"Message ID: {message.message_id}")
    print(f"Type: {message.message_type.value}")
    print(f"From: {message.sender_id} → To: {message.receiver_id}")
    print(f"Priority: {message.priority}")
    print(f"Valid: {ProtocolValidator.validate_message(message)}")

    # Create response
    response_msg = message.create_reply(
        sender_id="security_specialist_001",
        message_type=MessageType.TASK_RESPONSE,
        payload={
            "success": True,
            "findings": ["Hardcoded credential detected"],
            "confidence": 0.95
        }
    )

    print(f"\n=== Reply ===")
    print(f"From: {response_msg.sender_id} → To: {response_msg.receiver_id}")
    print(f"In reply to: {response_msg.in_reply_to}")
    print(f"Payload: {response_msg.payload}")
