"""Run hyperparameter sweeps for the Digital Physicist GNN."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from nerion_digital_physicist.training.run_training import (
    TrainingConfig,
    _prepare_output_dir,
    train_model,
)


def _parse_list(raw: str) -> List[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(float(item)) for item in raw.split(",") if item.strip()]


def _parse_bool_list(raw: str) -> List[bool]:
    values: List[bool] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "yes"}:
            values.append(True)
        elif token in {"0", "false", "no"}:
            values.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value from '{item}'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/datasets/gnn/latest/dataset.pt"),
        help="Path to the dataset snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/training_runs/sweeps"),
        help="Where sweep run artefacts are written",
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default="128,256,384",
        help="Comma-separated list of hidden channel sizes",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="1e-3,5e-4",
        help="Comma-separated list of learning rates",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="40",
        help="Comma-separated list of epoch counts",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="32",
        help="Comma-separated list of batch sizes",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="gcn,sage,gin,gat",
        help="Comma-separated list of architectures (gcn,sage,gin,gat)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean,sum",
        help="Comma-separated list of pooling strategies (mean,sum,max)",
    )
    parser.add_argument(
        "--num-layers",
        type=str,
        default="4",
        help="Comma-separated list of layer counts",
    )
    parser.add_argument(
        "--residual",
        type=str,
        default="false,true",
        help="Comma-separated booleans to toggle residual connections",
    )
    parser.add_argument(
        "--dropout",
        type=str,
        default="0.2",
        help="Comma-separated list of dropout probabilities",
    )
    parser.add_argument(
        "--attention-heads",
        type=str,
        default="4",
        help="Comma-separated list of attention head counts (for GAT)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Optional pretrained state dict to warm-start each sweep run",
    )
    args = parser.parse_args()

    hidden_sizes = _parse_int_list(args.hidden)
    lrs = _parse_list(args.lr)
    epochs_list = _parse_int_list(args.epochs)
    batch_sizes = _parse_int_list(args.batch)
    architectures = [arch.strip() for arch in args.architecture.split(",") if arch.strip()]
    poolings = [name.strip() for name in args.pooling.split(",") if name.strip()]
    num_layers_list = _parse_int_list(args.num_layers)
    residual_flags = _parse_bool_list(args.residual)
    dropouts = _parse_list(args.dropout)
    attention_heads = _parse_int_list(args.attention_heads)

    cumulative_results = []

    for arch, hidden, lr, epochs, batch, pooling, num_layers, residual, dropout, heads in itertools.product(
        architectures,
        hidden_sizes,
        lrs,
        epochs_list,
        batch_sizes,
        poolings,
        num_layers_list,
        residual_flags,
        dropouts,
        attention_heads,
    ):
        combo_dir = (
            args.output_dir
            / (
                f"arch{arch}_h{hidden}_pool{pooling}_res{int(residual)}"
                f"_nl{num_layers}_drop{dropout:g}_heads{heads}_lr{lr:g}_ep{epochs}_bs{batch}"
            )
        )
        config = TrainingConfig(
            dataset_path=args.dataset,
            output_dir=combo_dir,
            learning_rate=float(lr),
            batch_size=batch,
            epochs=epochs,
            val_ratio=args.val_ratio,
            seed=args.seed,
            hidden_channels=hidden,
            architecture=arch,
            pooling=pooling,
            num_layers=num_layers,
            residual=residual,
            dropout=float(dropout),
            attention_heads=heads,
            pretrained_path=args.pretrained,
        )

        print(
            "\n=== Sweep run: arch={arch} hidden={hidden} lr={lr} epochs={epochs} "
            "batch={batch} pooling={pooling} residual={residual} layers={num_layers} "
            "dropout={dropout} heads={heads} ===".format(
                arch=arch,
                hidden=hidden,
                lr=lr,
                epochs=epochs,
                batch=batch,
                pooling=pooling,
                residual=residual,
                num_layers=num_layers,
                dropout=dropout,
                heads=heads,
            )
        )
        results = train_model(config)

        artifact_dir = _prepare_output_dir(combo_dir)
        model_path = artifact_dir / "digital_physicist_brain.pt"
        torch.save(results["model"].state_dict(), model_path)

        metrics = {
            "config": {
                "dataset_path": str(config.dataset_path),
                "output_dir": str(config.output_dir),
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "val_ratio": config.val_ratio,
                "seed": config.seed,
                "hidden_channels": config.hidden_channels,
                "architecture": config.architecture,
                "pooling": config.pooling,
                "num_layers": config.num_layers,
                "residual": config.residual,
                "dropout": config.dropout,
                "attention_heads": config.attention_heads,
            },
            "train_size": results["train_size"],
            "val_size": results["val_size"],
            "history": results["history"],
            "best_epoch": results["best_epoch"],
            "num_node_features": results["num_node_features"],
        }
        (artifact_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

        cumulative_results.append(
            (
                artifact_dir,
                results["best_epoch"],
                config.architecture,
                config.pooling,
                config.residual,
            )
        )

    print("\nSweep complete. Best results:")
    cumulative_results.sort(key=lambda x: x[1]["val_accuracy"], reverse=True)
    for artifact_dir, best, arch, pooling, residual in cumulative_results[:5]:
        print(
            " - {path} :: arch={arch} pooling={pooling} residual={residual} "
            "val_acc={acc:.3f} val_auc={auc:.3f} val_f1={f1:.3f} @ epoch {epoch}".format(
                path=artifact_dir,
                arch=arch,
                pooling=pooling,
                residual=residual,
                acc=best["val_accuracy"],
                auc=best.get("val_auc", float("nan")),
                f1=best.get("val_f1", float("nan")),
                epoch=best["epoch"],
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
