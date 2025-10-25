"""Utilities for exporting curated GNN training datasets."""
from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder, SemanticEmbedder


@dataclass
class DatasetExportConfig:
    """Configuration describing a dataset export task."""

    db_path: Path
    output_dir: Path
    dataset_name: str = "binary_before_after"
    description: str = "Before/after curriculum graphs labelled for structural training"
    seed: int = 42
    mode: Literal["supervised", "pretrain"] = "supervised"
    workers: int = 8
    limit: Optional[int] = None


def _load_lessons(db_path: Path, limit: Optional[int] = None) -> Sequence[sqlite3.Row]:
    if not db_path.exists():
        raise FileNotFoundError(f"Curriculum database not found at {db_path}")

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cursor = con.cursor()
        if limit:
            cursor.execute(
                "SELECT name, before_code, after_code, timestamp FROM lessons ORDER BY id LIMIT ?",
                (limit,)
            )
        else:
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


def _process_single_lesson(
    lesson_dict: Dict[str, str]
) -> Tuple[Optional[Data], Optional[Data], Optional[str]]:
    """Process a single lesson and return (before_graph, after_graph, error_msg)."""
    # Recreate embedder in each worker (avoids pickling issues)
    embedder = get_global_embedder()
    name = lesson_dict["name"]

    try:
        before = create_graph_data_from_source(
            lesson_dict["before_code"], embedder=embedder
        )
        before_graph = _annotate_graph(before, label=0, lesson=name, sample_type="before") if before.num_nodes > 0 else None

        after = create_graph_data_from_source(
            lesson_dict["after_code"], embedder=embedder
        )
        after_graph = _annotate_graph(after, label=1, lesson=name, sample_type="after") if after.num_nodes > 0 else None

        return (before_graph, after_graph, None)
    except Exception as exc:
        return (None, None, f"Skipping lesson '{name}': {exc}")


def build_before_after_graphs(lessons: Iterable[sqlite3.Row], workers: int = 8, output_file: Optional[Path] = None) -> List[Data]:
    """Convert lesson rows into labelled graphs using multiprocessing.

    Args:
        lessons: Lesson data from database
        workers: Number of parallel workers
        output_file: If provided, saves after EVERY lesson (crash-resistant)
    """

    lessons_list = list(lessons)
    total = len(lessons_list)

    print(f"Building before/after graphs from {total} lessons...")
    print(f"Using {workers} worker processes for parallel processing...")
    if output_file:
        print(f"💾 Incremental saves to: {output_file}")
        print(f"📊 Saving after EVERY lesson (crash-resistant)\n")

    # Convert sqlite3.Row to dicts (for pickling)
    lessons_dicts = [
        {
            "name": str(lesson["name"]),
            "before_code": str(lesson["before_code"]),
            "after_code": str(lesson["after_code"]),
        }
        for lesson in lessons_list
    ]

    graphs: List[Data] = []
    errors: List[str] = []

    # Use multiprocessing pool
    try:
        with mp.Pool(processes=workers) as pool:
            # Process lessons in parallel with progress tracking
            results = []
            for i, result in enumerate(pool.imap_unordered(_process_single_lesson, lessons_dicts), 1):
                results.append(result)
                before_graph, after_graph, error_msg = result

                # Collect graphs immediately
                if error_msg:
                    errors.append(error_msg)
                if before_graph:
                    graphs.append(before_graph)
                if after_graph:
                    graphs.append(after_graph)

                # Show progress after EVERY lesson
                lesson_name = lessons_dicts[i-1]['name'] if i <= len(lessons_dicts) else "unknown"
                status = "✓" if (before_graph or after_graph) else "✗"
                print(f"[{i}/{total}] {status} {lesson_name[:50]:<50} | Graphs: {len(graphs)}")

                # SAVE after EVERY lesson (crash-resistant)
                if output_file and len(graphs) > 0:
                    torch.save({"samples": graphs}, output_file)
                    print(f"          💾 Saved {len(graphs)} graphs to disk")

    except Exception as exc:
        print(f"Multiprocessing failed, falling back to sequential processing: {exc}")
        # Fallback to sequential processing
        embedder = get_global_embedder()
        return _build_sequential(lessons_list, embedder)

    print(f"Completed: {total} lessons processed, {len(graphs)} graphs created")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    if not graphs:
        raise RuntimeError("No graphs could be produced from the provided lessons")

    return graphs


def _build_sequential(lessons_list: List[sqlite3.Row], embedder: SemanticEmbedder) -> List[Data]:
    """Fallback sequential processing (original implementation)."""
    graphs: List[Data] = []
    total = len(lessons_list)

    for idx, lesson in enumerate(lessons_list, 1):
        name = str(lesson["name"])
        if idx % 50 == 0:
            print(f"Progress: {idx}/{total} lessons processed ({len(graphs)} graphs created)")
        try:
            before = create_graph_data_from_source(
                str(lesson["before_code"]), embedder=embedder
            )
            if before.num_nodes > 0:
                graphs.append(
                    _annotate_graph(before, label=0, lesson=name, sample_type="before")
                )

            after = create_graph_data_from_source(
                str(lesson["after_code"]), embedder=embedder
            )
            if after.num_nodes > 0:
                graphs.append(
                    _annotate_graph(after, label=1, lesson=name, sample_type="after")
                )
        except Exception as exc:
            print(f" - Skipping lesson '{name}' due to graph extraction error: {exc}")
            continue

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

    lessons = _load_lessons(config.db_path, limit=config.limit)

    # Create output directory early
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = output_dir / "dataset.pt"

    print(f"\n💾 Output: {dataset_file}")
    print(f"📊 Processing {len(lessons)} lessons with {config.workers} workers\n")

    if config.mode == "supervised":
        graphs = build_before_after_graphs(lessons, workers=config.workers, output_file=dataset_file)
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

    # Final save
    print(f"\n💾 Saving {len(graphs)} graphs to {dataset_file}...")
    torch.save({"samples": graphs}, dataset_file)
    print(f"✅ Dataset saved successfully!")

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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for parallel processing (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of lessons to process (for testing)",
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
        workers=args.workers,
        limit=args.limit,
    )

    print(f"Exporting dataset to {final_output} ...")
    metadata = export_dataset(config)
    print(json.dumps(metadata, indent=2))
    print("Dataset export complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
