"""Command-line utility to train the GNN using replay samples."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..agent.brain import CodeGraphNN
from .replay_trainer import run_replay_training_step, replay_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nerion brain from replay experiences")
    parser.add_argument("--replay-root", type=str, required=True, help="Directory containing replay.jsonl")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Replay batch size")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Optimizer learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    replay_root = Path(args.replay_root)
    if not replay_ready(replay_root):
        print("No replay data available. Exiting.")
        return

    # Determine node feature size from first sample
    from .replay_sampler import sample_training_batch

    initial_batch = sample_training_batch(replay_root, batch_size=1)
    if not initial_batch:
        print("Unable to fetch initial batch from replay store.")
        return

    feature_dim = initial_batch[0].graph_data.x.shape[1]
    model = CodeGraphNN(num_node_features=feature_dim, hidden_channels=64, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = run_replay_training_step(
            replay_root=replay_root,
            model=model,
            optimizer=optimizer,
            batch_size=args.batch_size,
        )
        if loss is None:
            print("Epoch", epoch, "- no samples available")
            break
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

    print("Replay training complete.")


if __name__ == "__main__":
    main()
