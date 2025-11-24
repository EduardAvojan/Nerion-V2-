"""
Multi-Agent System

Enables multiple specialized agents to collaborate, share knowledge,
and solve complex tasks together.
"""

from .protocol import (
    AgentMessage,
    MessageType,
    AgentRole,
    TaskRequest,
    TaskResponse,
    Proposal,
    Vote,
    Conflict,
    ConflictType,
    CoordinationStrategy,
    CoordinationPlan,
    ProtocolValidator,
)

from .specialists import (
    SpecialistAgent,
    AgentCapability,
    PythonSpecialist,
    JavaScriptSpecialist,
    JavaSpecialist,
    SecuritySpecialist,
    PerformanceSpecialist,
    TestingSpecialist,
    RefactoringSpecialist,
    BugFixingSpecialist,
    DocumentationSpecialist,
    CoordinatorAgent,
    GeneralistAgent,
)

from .coordinator import (
    MultiAgentCoordinator,
    AgentRegistry,
)

__all__ = [
    # Protocol
    'AgentMessage',
    'MessageType',
    'AgentRole',
    'TaskRequest',
    'TaskResponse',
    'Proposal',
    'Vote',
    'Conflict',
    'ConflictType',
    'CoordinationStrategy',
    'CoordinationPlan',
    'ProtocolValidator',
    # Specialists
    'SpecialistAgent',
    'AgentCapability',
    # Language Specialists
    'PythonSpecialist',
    'JavaScriptSpecialist',
    'JavaSpecialist',
    # Domain Specialists
    'SecuritySpecialist',
    'PerformanceSpecialist',
    'TestingSpecialist',
    # Task Specialists
    'RefactoringSpecialist',
    'BugFixingSpecialist',
    'DocumentationSpecialist',
    # Meta Agents
    'CoordinatorAgent',
    'GeneralistAgent',
    # Coordinator
    'MultiAgentCoordinator',
    'AgentRegistry',
]
