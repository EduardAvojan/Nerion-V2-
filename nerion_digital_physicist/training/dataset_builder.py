"""Utilities for exporting curated GNN training datasets."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

import torch
from torch_geometric.data import Data

from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder


@dataclass
class DatasetExportConfig:
    """Configuration describing a dataset export task."""

    db_path: Path
    output_dir: Path
    dataset_name: str = "binary_before_after"
    description: str = "Before/after curriculum graphs labelled for structural training"
    seed: int = 42
    mode: Literal["supervised", "pretrain"] = "supervised"


def _load_lessons(db_path: Path) -> Sequence[sqlite3.Row]:
    if not db_path.exists():
        raise FileNotFoundError(f"Curriculum database not found at {db_path}")

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cursor = con.cursor()
        cursor.execute(
            "SELECT name, before_code, after_code, timestamp FROM lessons ORDER BY id"
        )
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(f"No lessons available in {db_path}")

    return rows


def _annotate_graph(graph: Data, *, label: int, lesson: str, sample_type: str) -> Data:
    graph.y = torch.tensor([label], dtype=torch.long)
    graph.sample_meta = {  # type: ignore[attr-defined]
        "lesson": lesson,
        "sample_type": sample_type,
    }
    return graph


def build_before_after_graphs(lessons: Iterable[sqlite3.Row]) -> List[Data]:
    """Convert lesson rows into labelled graphs."""

    embedder = get_global_embedder()
    graphs: List[Data] = []

    for lesson in lessons:
        name = str(lesson["name"])
        try:
            before = create_graph_data_from_source(
                str(lesson["before_code"]), embedder=embedder
            )
            graphs.append(
                _annotate_graph(before, label=0, lesson=name, sample_type="before")
            )

            after = create_graph_data_from_source(
                str(lesson["after_code"]), embedder=embedder
            )
            graphs.append(
                _annotate_graph(after, label=1, lesson=name, sample_type="after")
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f" - Skipping lesson '{name}' due to graph extraction error: {exc}")
            continue

    if not graphs:
        raise RuntimeError("No graphs could be produced from the provided lessons")

    return graphs


def build_unlabelled_graphs(lessons: Iterable[sqlite3.Row]) -> List[Data]:
    """Convert lesson rows into unlabelled graphs for self-supervised pretraining."""

    embedder = get_global_embedder()
    graphs: List[Data] = []

    for lesson in lessons:
        name = str(lesson["name"])
        for sample_type, code in (
            ("before", lesson["before_code"]),
            ("after", lesson["after_code"]),
        ):
            try:
                graph = create_graph_data_from_source(str(code), embedder=embedder)
                graph.sample_meta = {  # type: ignore[attr-defined]
                    "lesson": name,
                    "sample_type": sample_type,
                    "mode": "pretrain",
                }
                graphs.append(graph)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f" - Skipping lesson '{name}' ({sample_type}) due to graph extraction error: {exc}"
                )
                continue

    if not graphs:
        raise RuntimeError("No graphs could be produced from the provided lessons")

    return graphs


def export_dataset(config: DatasetExportConfig) -> Dict[str, object]:
    """Build and persist a dataset artefact returning its metadata."""

    lessons = _load_lessons(config.db_path)
    if config.mode == "supervised":
        graphs = build_before_after_graphs(lessons)
        label_counts = {
            "0_before": sum(1 for g in graphs if int(g.y.item()) == 0),
            "1_after": sum(1 for g in graphs if int(g.y.item()) == 1),
        }
    elif config.mode == "pretrain":
        graphs = build_unlabelled_graphs(lessons)
        label_counts = {}
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported export mode: {config.mode}")

    counts = {
        "total_graphs": len(graphs),
        "label_counts": label_counts,
        "num_features": graphs[0].num_node_features,
    }

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = output_dir / "dataset.pt"
    torch.save({"samples": graphs}, dataset_file)

    manifest = {
        "name": config.dataset_name,
        "description": config.description,
        "mode": config.mode,
        "source_db": str(config.db_path),
        "dataset_file": str(dataset_file),
        "num_lessons": len(set(str(row["name"]) for row in lessons)),
        **counts,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return manifest


def main() -> None:
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Export GNN training dataset artefacts")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("out/learning/curriculum.sqlite"),
        help="Path to the curriculum SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/datasets/gnn/latest"),
        help="Directory where the dataset artefacts will be written",
    )
    parser.add_argument(
        "--name",
        default="binary_before_after",
        help="Dataset name stored in the manifest",
    )
    parser.add_argument(
        "--mode",
        default="supervised",
        choices=["supervised", "pretrain"],
        help="Export mode: labelled supervised dataset or unlabeled pretraining corpus",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_output = args.output_dir / args.mode / timestamp

    dataset_name = args.name
    description = (
        "Before/after curriculum graphs labelled for structural training"
        if args.mode == "supervised"
        else "Unlabelled curriculum code graphs for self-supervised pretraining"
    )
    if dataset_name == "binary_before_after" and args.mode == "pretrain":
        dataset_name = "code_graph_pretrain"

    config = DatasetExportConfig(
        db_path=args.db,
        output_dir=final_output,
        dataset_name=dataset_name,
        description=description,
        mode=args.mode,
    )

    print(f"Exporting dataset to {final_output} ...")
    metadata = export_dataset(config)
    print(json.dumps(metadata, indent=2))
    print("Dataset export complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
