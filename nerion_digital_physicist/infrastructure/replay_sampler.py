"""Utilities to sample replay experiences for agent training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from .memory import ReplayStore
from ..agent.data import create_graph_data_object, create_graph_data_from_source
from ..agent.semantics import get_global_embedder


@dataclass
class TrainingSample:
    graph_data: any
    label: int
    metadata: dict


STATUS_TO_LABEL = {
    "solved": 0,
    "failed": 1,
    "pending": 1,
}


def sample_training_batch(
    replay_root: Path,
    batch_size: int,
    strategy: str = "priority",
) -> List[TrainingSample]:
    replay = ReplayStore(replay_root)
    embedder = get_global_embedder()
    experiences = replay.sample(batch_size, strategy=strategy)
    samples: List[TrainingSample] = []
    for exp in experiences:
        # Try to get source code from metadata first (new format)
        source_code = exp.metadata.get("source_code")

        if source_code:
            # Create graph directly from source code
            graph_data = create_graph_data_from_source(source_code, embedder=embedder)
        else:
            # Fall back to file path (old format)
            source_path = exp.metadata.get("source_path")
            if not source_path:
                artifacts_path = exp.metadata.get("artifacts_path")
                if artifacts_path:
                    source_path = str(Path(artifacts_path) / "src" / "module.py")
            if not source_path:
                continue
            graph_data = create_graph_data_object(source_path, embedder=embedder)

        if not hasattr(graph_data, "batch") or graph_data.batch is None:
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
        label = STATUS_TO_LABEL.get(exp.status, 1)
        samples.append(
            TrainingSample(
                graph_data=graph_data,
                label=label,
                metadata={
                    "experience_id": exp.experience_id,
                    "task_id": exp.task_id,
                    "template_id": exp.template_id,
                },
            )
        )
    return samples
