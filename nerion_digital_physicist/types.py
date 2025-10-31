"""
Type definitions for Nerion Digital Physicist.

Centralized type definitions to ensure consistency across modules.
"""
from __future__ import annotations

from typing import TypeAlias, Dict, Any, List, Tuple, Optional

try:
    from torch_geometric.data import Data, Batch
    GraphData: TypeAlias = Data
    GraphBatch: TypeAlias = Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    # Fallback for when torch_geometric not installed
    GraphData: TypeAlias = Any  # type: ignore
    GraphBatch: TypeAlias = Any  # type: ignore
    HAS_TORCH_GEOMETRIC = False

# Training data types
TrainingExample: TypeAlias = Tuple[GraphData, int]  # (graph, label)
TrainingBatch: TypeAlias = List[TrainingExample]

# Experience types (for MAML tasks)
SupportSet: TypeAlias = List[TrainingExample]
QuerySet: TypeAlias = List[TrainingExample]

# Model state
ModelStateDict: TypeAlias = Dict[str, Any]
