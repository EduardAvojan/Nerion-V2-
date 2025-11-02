"""Top-level package for the Nerion Digital Physicist stack."""

from .agent import AgentV2, CodeGraphNN
from .environment.actions import Action
from .environment.core import EnvironmentV2

# Advanced AGI Capabilities
from .architecture import ArchitecturalGraphBuilder, ArchitectureGraph, PatternDetector
from .world_model import WorldModelSimulator, SymbolicExecutor, DynamicsModel
from .learning import ContrastiveLearner, CodeAugmentor

__all__ = [
    "Action",
    "AgentV2",
    "CodeGraphNN",
    "EnvironmentV2",
    # Architecture + World Model + Contrastive Learning
    "ArchitecturalGraphBuilder",
    "ArchitectureGraph",
    "PatternDetector",
    "WorldModelSimulator",
    "SymbolicExecutor",
    "DynamicsModel",
    "ContrastiveLearner",
    "CodeAugmentor",
]
