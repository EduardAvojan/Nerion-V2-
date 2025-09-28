"""Replay-driven training utilities."""
from __future__ import annotations


import torch
import torch.nn.functional as F

from ..agent.brain import CodeGraphNN
from .replay_sampler import sample_training_batch
from .memory import ReplayStore


def run_replay_training_step(
    replay_root,
    model: CodeGraphNN,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 4,
    strategy: str = "priority",
) -> float | None:
    samples = sample_training_batch(replay_root, batch_size=batch_size, strategy=strategy)
    if not samples:
        return None

    model.train()
    optimizer.zero_grad()

    losses = []
    for sample in samples:
        data = sample.graph_data
        logits = model(data.x, data.edge_index, data.batch)
        target = torch.tensor([sample.label])
        loss = F.cross_entropy(logits, target)
        loss.backward()
        losses.append(loss.item())

    optimizer.step()
    return sum(losses) / len(losses)


def replay_ready(replay_root) -> bool:
    store = ReplayStore(replay_root)
    return len(list(store.load())) > 0
