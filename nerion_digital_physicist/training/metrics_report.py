"""Summarise GNN training run metrics."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunSummary:
    run_dir: Path
    config: Dict[str, object]
    train_size: int
    val_size: int
    best_epoch: Dict[str, object]
    history: List[Dict[str, object]]

    @property
    def best_val_acc(self) -> float:
        if self.best_epoch:
            return float(self.best_epoch.get("val_accuracy", 0.0))
        if not self.history:
            return 0.0
        return max(float(item.get("val_accuracy", 0.0)) for item in self.history)

    @property
    def best_epoch_number(self) -> int:
        if self.best_epoch:
            return int(self.best_epoch.get("epoch", 0))
        if not self.history:
            return 0
        best = max(self.history, key=lambda item: float(item.get("val_accuracy", 0.0)))
        return int(best.get("epoch", 0))

    @property
    def best_val_auc(self) -> float:
        if self.best_epoch:
            value = self.best_epoch.get("val_auc")
            if value is not None:
                return float(value)
        for item in reversed(self.history):
            value = item.get("val_auc")
            if value is not None:
                return float(value)
        return float("nan")

    @property
    def best_val_f1(self) -> float:
        if self.best_epoch:
            value = self.best_epoch.get("val_f1")
            if value is not None:
                return float(value)
        for item in reversed(self.history):
            value = item.get("val_f1")
            if value is not None:
                return float(value)
        return float("nan")


def _load_summary(run_dir: Path) -> Optional[RunSummary]:
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None

    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    return RunSummary(
        run_dir=run_dir,
        config=data.get("config", {}),
        train_size=int(data.get("train_size", 0)),
        val_size=int(data.get("val_size", 0)),
        best_epoch=data.get("best_epoch", {}),
        history=data.get("history", []),
    )


def collect_runs(root: Path) -> List[RunSummary]:
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")
    summaries: List[RunSummary] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        summary = _load_summary(entry)
        if summary:
            summaries.append(summary)
            continue
        for child in sorted(entry.iterdir()):
            if not child.is_dir():
                continue
            nested_summary = _load_summary(child)
            if nested_summary:
                summaries.append(nested_summary)
    return summaries


def format_table(summaries: List[RunSummary]) -> str:
    if not summaries:
        return "No runs found."

    headers = [
        "run",
        "arch",
        "pool",
        "res",
        "layers",
        "epochs",
        "hidden",
        "lr",
        "batch",
        "best_val_acc",
        "best_val_auc",
        "best_val_f1",
        "best_epoch",
    ]

    rows = [headers]
    for summary in summaries:
        config = summary.config
        rows.append(
            [
                summary.run_dir.name,
                str(config.get("architecture", "gcn")),
                str(config.get("pooling", "-")),
                "yes" if config.get("residual") else "no",
                str(config.get("num_layers", "-")),
                str(config.get("epochs", "-")),
                str(config.get("hidden_channels", "-")),
                str(config.get("learning_rate")),
                str(config.get("batch_size")),
                f"{summary.best_val_acc:.3f}",
                f"{summary.best_val_auc:.3f}",
                f"{summary.best_val_f1:.3f}",
                str(summary.best_epoch_number),
            ]
        )

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def _format(row: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in col_widths)
    table_lines = [_format(rows[0]), separator]
    table_lines.extend(_format(row) for row in rows[1:])
    return "\n".join(table_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise GNN training runs")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/runs/gnn"),
        help="Directory containing run subfolders",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only the top-N runs by best validation accuracy (0 = all)",
    )
    args = parser.parse_args()

    summaries = collect_runs(args.runs_dir)
    summaries.sort(key=lambda run: run.best_val_acc, reverse=True)

    if args.top > 0:
        summaries = summaries[: args.top]

    if args.format == "json":
        serialisable = [
            {
                "run": summary.run_dir.name,
                "config": summary.config,
                "train_size": summary.train_size,
                "val_size": summary.val_size,
                "best_epoch": summary.best_epoch,
            }
            for summary in summaries
        ]
        print(json.dumps(serialisable, indent=2, sort_keys=True))
    else:
        print(format_table(summaries))


if __name__ == "__main__":  # pragma: no cover
    main()
