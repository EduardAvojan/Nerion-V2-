"""
Architectural Analysis Module

Builds repository-wide architectural graphs for system-level understanding.
"""

from .graph_builder import ArchitecturalGraphBuilder, ArchitectureGraph
from .pattern_detector import PatternDetector, ArchitecturalPattern

__all__ = [
    'ArchitecturalGraphBuilder',
    'ArchitectureGraph',
    'PatternDetector',
    'ArchitecturalPattern',
]
