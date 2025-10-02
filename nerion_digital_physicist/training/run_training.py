"""Train the Digital Physicist GNN brain from a versioned dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from nerion_digital_physicist.agent.brain import build_gnn


@dataclass
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    val_ratio: float = 0.15
    seed: int = 42
    hidden_channels: int = 256
    architecture: str = "gcn"
    pooling: str = "mean"
    num_layers: int = 4
    residual: bool = False
    dropout: float = 0.2
    attention_heads: int = 4
    pretrained_path: Optional[Path] = None


POOLING_REGISTRY: Dict[str, Callable] = {
    "mean": global_mean_pool,
    "sum": global_add_pool,
    "max": global_max_pool,
}


class GraphSampleDataset(Dataset):
    """Thin Dataset wrapper around a list of Data samples."""

    def __init__(self, samples: Sequence[object]):
        self._samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


def _load_samples(path: Path) -> List[object]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    payload = torch.load(path, weights_only=False)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"Dataset at {path} does not contain any samples")
    return samples


def _split_dataset(
    dataset: GraphSampleDataset, val_ratio: float, seed: int
) -> Tuple[GraphSampleDataset, GraphSampleDataset]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator)
    return GraphSampleDataset(train_subset), GraphSampleDataset(val_subset)


def _safe_auc(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute ROC-AUC without introducing an sklearn dependency."""

    if probabilities.numel() == 0:
        return math.nan

    unique_targets = torch.unique(targets)
    if unique_targets.numel() < 2:
        return math.nan

    sorted_probs, indices = torch.sort(probabilities)
    ranks = torch.empty_like(indices, dtype=torch.float64)
    ranks[indices] = torch.arange(1, probabilities.numel() + 1, dtype=torch.float64)

    pos_mask = targets == 1
    n_pos = pos_mask.sum().item()
    n_neg = probabilities.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan

    sum_ranks_pos = ranks[pos_mask].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _safe_f1(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    if probabilities.numel() == 0:
        return math.nan

    preds = (probabilities >= 0.5).to(torch.int32)
    targets = targets.to(torch.int32)
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return math.nan
    return float((2 * tp) / denom)


def _select_pooling(name: str) -> Callable:
    key = name.lower()
    if key not in POOLING_REGISTRY:
        raise ValueError(
            f"Unknown pooling '{name}'. Available: {', '.join(sorted(POOLING_REGISTRY))}"
        )
    return POOLING_REGISTRY[key]


def _run_epoch(model, loader, pool_fn, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    collected_probs: List[torch.Tensor] = []
    collected_targets: List[torch.Tensor] = []

    for batch in loader:
        if is_train:
            optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)
        graph_logits = pool_fn(logits, batch.batch)
        targets = batch.y.view(-1)
        loss = F.cross_entropy(graph_logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = graph_logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_graphs += batch.num_graphs

        if not is_train:
            probs = graph_logits.softmax(dim=1)[:, 1].detach().cpu()
            collected_probs.append(probs)
            collected_targets.append(targets.detach().cpu())

    avg_loss = total_loss / max(total_graphs, 1)
    accuracy = total_correct / max(total_graphs, 1)
    extra_metrics: Dict[str, float] = {}
    if collected_probs:
        probs = torch.cat(collected_probs)
        targets = torch.cat(collected_targets)
        extra_metrics["auc"] = _safe_auc(probs, targets)
        extra_metrics["f1"] = _safe_f1(probs, targets)
    return avg_loss, accuracy, extra_metrics


def train_model(config: TrainingConfig) -> Dict[str, object]:
    samples = _load_samples(config.dataset_path)
    dataset = GraphSampleDataset(samples)

    train_dataset, val_dataset = _split_dataset(dataset, config.val_ratio, config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    num_node_features = train_dataset[0].num_node_features  # type: ignore[index]
    model = build_gnn(
        architecture=config.architecture,
        num_node_features=num_node_features,
        hidden_channels=config.hidden_channels,
        num_classes=2,
        num_layers=config.num_layers,
        residual=config.residual,
        dropout=config.dropout,
        attention_heads=config.attention_heads,
    )

    if config.pretrained_path:
        print(f"Loading pretrained weights from {config.pretrained_path} ...")
        state_dict = torch.load(config.pretrained_path, map_location="cpu")
        # Drop mismatched parameters (e.g., classifier head or different hidden width)
        filtered = {}
        model_state = model.state_dict()
        for key, tensor in state_dict.items():
            if key.startswith("head.3."):
                continue
            target = model_state.get(key)
            if target is None:
                filtered[key] = tensor
                continue
            if target.shape != tensor.shape:
                continue
            filtered[key] = tensor
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if unexpected:
            print(
                " - Warning: unexpected keys encountered in pretrained state: "
                + ", ".join(sorted(unexpected))
            )
        if missing:
            print(
                " - Info: skipped classification head parameters (will be reinitialised): "
                + ", ".join(sorted(missing))
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    pool_fn = _select_pooling(config.pooling)

    history = []
    best_snapshot: dict[str, object] | None = None
    best_val_metric = float("-inf")
    epochs_since_improvement = 0
    patience = max(5, int(config.epochs * 0.1))

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, _ = _run_epoch(model, train_loader, pool_fn, optimizer)
        val_loss, val_acc, val_metrics = _run_epoch(model, val_loader, pool_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_metrics.get("auc"),
                "val_f1": val_metrics.get("f1"),
            }
        )
        if epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                f"val_auc={val_metrics.get('auc', float('nan')):.3f} "
                f"val_f1={val_metrics.get('f1', float('nan')):.3f}"
            )

        current_val_metric = val_acc
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_snapshot = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_metrics.get("auc"),
                "val_f1": val_metrics.get("f1"),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(
                "Early stopping triggered: no validation improvement for "
                f"{epochs_since_improvement} epochs (patience={patience})."
            )
            break

    assert best_snapshot is not None
    model.load_state_dict(best_snapshot.pop("state_dict"))
    best_epoch = best_snapshot

    return {
        "model": model,
        "history": history,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "best_epoch": best_epoch,
        "num_node_features": num_node_features,
        "pooling": config.pooling,
        "residual": config.residual,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "attention_heads": config.attention_heads,
    }


def _prepare_output_dir(base: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Digital Physicist GNN")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/datasets/gnn/latest/dataset.pt"),
        help="Path to the dataset .pt file produced by dataset_builder.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/training_runs/supervised"),
        help="Directory where training artefacts will be stored",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation split ratio (0-1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=256,
        help="Hidden channel width for the GNN",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="gcn",
        choices=["gcn", "sage", "gin", "gat"],
        help="GNN architecture to train",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=sorted(POOLING_REGISTRY.keys()),
        help="Global pooling strategy applied before graph-level prediction",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Enable residual connections between layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied after each layer",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
        help="Number of attention heads (for GAT architecture)",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Path to a pretrained state dict for warm starting the model",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hidden_channels=args.hidden_channels,
        architecture=args.architecture,
        pooling=args.pooling,
        num_layers=args.num_layers,
        residual=args.residual,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        pretrained_path=args.pretrained,
    )

    print(f"Loading dataset from {config.dataset_path} ...")
    results = train_model(config)

    artifact_dir = _prepare_output_dir(config.output_dir)
    model_path = artifact_dir / "digital_physicist_brain.pt"
    torch.save(results["model"].state_dict(), model_path)
    print(
        f"Saved trained model to {model_path} (best val acc {results['best_epoch']['val_accuracy']:.3f}"
        f" @ epoch {results['best_epoch']['epoch']})"
    )

    config_dict = asdict(config)
    config_dict["dataset_path"] = str(config.dataset_path)
    config_dict["output_dir"] = str(config.output_dir)
    if config.pretrained_path is not None:
        config_dict["pretrained_path"] = str(config.pretrained_path)
    config_dict["architecture"] = config.architecture

    metrics = {
        "config": config_dict,
        "train_size": results["train_size"],
        "val_size": results["val_size"],
        "history": results["history"],
        "best_epoch": results["best_epoch"],
        "num_node_features": results["num_node_features"],
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Symlink/copy latest model into root for live inference without reconfiguring callers.
    live_model_path = Path("digital_physicist_brain.pt")
    torch.save(results["model"].state_dict(), live_model_path)
    print(f"Updated live checkpoint at {live_model_path}")

    meta = {
        "architecture": config.architecture,
        "hidden_channels": config.hidden_channels,
        "num_node_features": results["num_node_features"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_accuracy": results["best_epoch"]["val_accuracy"],
        "best_epoch": results["best_epoch"]["epoch"],
        "run_dir": str(artifact_dir),
        "pooling": config.pooling,
        "residual": config.residual,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "attention_heads": config.attention_heads,
        "best_val_auc": results["best_epoch"].get("val_auc"),
        "best_val_f1": results["best_epoch"].get("val_f1"),
    }
    if config.pretrained_path is not None:
        meta["pretrained_source"] = str(config.pretrained_path)
    (Path("digital_physicist_brain.meta.json")).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Wrote model metadata to digital_physicist_brain.meta.json")

    print("Training complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
