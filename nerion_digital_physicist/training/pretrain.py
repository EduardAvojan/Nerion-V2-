"""Self-supervised pretraining for the Digital Physicist GNN."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from nerion_digital_physicist.agent.brain import build_gnn
from nerion_digital_physicist.training.run_training import (
    GraphSampleDataset,
    _load_samples,
    _prepare_output_dir,
)


@dataclass
class PretrainConfig:
    dataset_path: Path
    output_dir: Path
    architecture: str = "gat"
    hidden_channels: int = 256
    num_layers: int = 4
    residual: bool = True
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    mask_prob: float = 0.15
    val_ratio: float = 0.1
    seed: int = 42
    attention_heads: int = 4


def _mask_features(x: torch.Tensor, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask a proportion of features for reconstruction."""

    if not 0.0 < mask_prob <= 1.0:
        raise ValueError("mask_prob must be in the interval (0, 1]")
    mask = torch.rand_like(x).lt(mask_prob)
    masked = x.clone()
    masked[mask] = 0.0
    return masked, mask


def _split_dataset(
    dataset: GraphSampleDataset, val_ratio: float, seed: int
) -> Tuple[GraphSampleDataset, GraphSampleDataset]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator
    )
    return GraphSampleDataset(train_subset), GraphSampleDataset(val_subset)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    mask_prob: float,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    is_train = optimiser is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    total_mask_ratio = 0.0

    for batch in loader:
        batch = batch.to(device)
        original = batch.x.clone()
        masked, mask = _mask_features(original, mask_prob)
        batch.x = masked

        if is_train:
            optimiser.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(outputs[mask], original[mask])

            if is_train:
                loss.backward()
                optimiser.step()

        total_loss += loss.item()
        total_batches += 1
        total_mask_ratio += mask.float().mean().item()

    if total_batches == 0:
        return {"loss": 0.0, "mask_ratio": 0.0}

    return {
        "loss": total_loss / total_batches,
        "mask_ratio": total_mask_ratio / total_batches,
    }


def pretrain_model(config: PretrainConfig) -> Dict[str, object]:
    samples = _load_samples(config.dataset_path)
    dataset = GraphSampleDataset(samples)

    train_dataset, val_dataset = _split_dataset(dataset, config.val_ratio, config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    feature_dim = train_dataset[0].num_node_features  # type: ignore[index]

    model = build_gnn(
        architecture=config.architecture,
        num_node_features=feature_dim,
        hidden_channels=config.hidden_channels,
        num_classes=feature_dim,
        num_layers=config.num_layers,
        residual=config.residual,
        dropout=config.dropout,
        attention_heads=config.attention_heads,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: List[Dict[str, float]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, config.mask_prob, device, optimiser)
        val_metrics = _run_epoch(model, val_loader, config.mask_prob, device, None)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mask_ratio": train_metrics["mask_ratio"],
            "val_mask_ratio": val_metrics["mask_ratio"],
        }
        history.append(record)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:03d}: train_loss={record['train_loss']:.5f} "
                f"val_loss={record['val_loss']:.5f} mask={record['train_mask_ratio']:.2%}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to(torch.device("cpu"))

    return {
        "model": model,
        "history": history,
        "feature_dim": feature_dim,
        "best_val_loss": best_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the Digital Physicist GNN with masked-node modelling")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/datasets/gnn/latest/pretrain/dataset.pt"),
        help="Path to the pretraining dataset .pt file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/gnn_pretrain"),
        help="Directory where pretraining artefacts will be stored",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", type=str, default="gat", choices=["gcn", "sage", "gin", "gat"])
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-heads", type=int, default=4)
    args = parser.parse_args()

    config = PretrainConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        architecture=args.architecture,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        residual=args.residual,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mask_prob=args.mask_prob,
        val_ratio=args.val_ratio,
        seed=args.seed,
        attention_heads=args.attention_heads,
    )

    print(f"Loading pretraining dataset from {config.dataset_path} ...")
    results = pretrain_model(config)

    artifact_dir = _prepare_output_dir(config.output_dir)
    model_path = artifact_dir / "digital_physicist_pretrain.pt"
    torch.save(results["model"].state_dict(), model_path)
    print(
        f"Saved pretrained model to {model_path} (best val loss {results['best_val_loss']:.5f})"
    )

    config_dict = asdict(config)
    config_dict["dataset_path"] = str(config.dataset_path)
    config_dict["output_dir"] = str(config.output_dir)

    metrics = {
        "config": config_dict,
        "history": results["history"],
        "best_val_loss": results["best_val_loss"],
        "feature_dim": results["feature_dim"],
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    live_path = Path("digital_physicist_pretrain.pt")
    torch.save(results["model"].state_dict(), live_path)
    print(f"Updated live pretraining checkpoint at {live_path}")

    meta = {
        "architecture": config.architecture,
        "hidden_channels": config.hidden_channels,
        "num_layers": config.num_layers,
        "residual": config.residual,
        "dropout": config.dropout,
        "mask_prob": config.mask_prob,
        "epochs": config.epochs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": results["best_val_loss"],
        "run_dir": str(artifact_dir),
        "dataset_path": str(config.dataset_path),
    }
    Path("digital_physicist_pretrain.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Wrote pretraining metadata to digital_physicist_pretrain.meta.json")

    print("Pretraining complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
