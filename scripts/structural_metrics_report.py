#!/usr/bin/env python3
"""Summarise structural vetting telemetry for the Digital Physicist."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

DEFAULT_PATH = Path("out/learning/structural_metrics.jsonl")


@dataclass
class StructuralSample:
    timestamp: datetime
    lesson: str
    status: str
    delta: Optional[float]
    message: Optional[str]
    model_architecture: Optional[str]


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def iter_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def to_samples(records: Iterable[dict]) -> List[StructuralSample]:
    samples: List[StructuralSample] = []
    for record in records:
        if record.get("phase") != "structural":
            continue
        ts_raw = record.get("timestamp")
        if not ts_raw:
            continue
        try:
            timestamp = parse_timestamp(ts_raw)
        except ValueError:
            continue
        metrics = record.get("metrics") or {}
        delta = metrics.get("delta")
        if isinstance(delta, (int, float)):
            delta = float(delta)
        else:
            delta = None
        samples.append(
            StructuralSample(
                timestamp=timestamp,
                lesson=str(record.get("lesson", "unknown")),
                status=str(record.get("structural_status", "unknown")),
                delta=delta,
                message=record.get("message"),
                model_architecture=record.get("model_architecture"),
            )
        )
    return samples


def filter_samples(samples: List[StructuralSample], *, hours: Optional[float], limit: Optional[int]) -> List[StructuralSample]:
    filtered = samples
    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered = [sample for sample in filtered if sample.timestamp >= cutoff]
    filtered.sort(key=lambda sample: sample.timestamp, reverse=True)
    if limit is not None:
        filtered = filtered[: limit]
    filtered.sort(key=lambda sample: sample.timestamp)  # chronological output
    return filtered


def compute_summary(samples: List[StructuralSample]) -> None:
    if not samples:
        print("No structural vetting records matched the filter.")
        return

    total = len(samples)
    passed = sum(1 for sample in samples if sample.status == "passed")
    failed = sum(1 for sample in samples if sample.status == "failed")
    skipped = total - passed - failed

    deltas = [sample.delta for sample in samples if sample.delta is not None]
    avg_delta = mean(deltas) if deltas else None
    min_delta = min(deltas) if deltas else None
    max_delta = max(deltas) if deltas else None

    architectures = sorted({sample.model_architecture for sample in samples if sample.model_architecture})

    print("Structural Vetting Summary")
    print("===========================")
    print(f"Samples analysed : {total}")
    print(f"Passed           : {passed}")
    print(f"Failed           : {failed}")
    print(f"Skipped/Other    : {skipped}")
    if avg_delta is not None:
        print(f"Avg delta (Δ)    : {avg_delta:+.3f}")
        print(f"Min delta (Δ)    : {min_delta:+.3f}")
        print(f"Max delta (Δ)    : {max_delta:+.3f}")
    else:
        print("Avg delta (Δ)    : n/a")
    pass_rate = passed / total if total else 0.0
    print(f"Pass rate        : {pass_rate:.1%}")
    if architectures:
        print(f"Architectures    : {', '.join(architectures)}")
    print()
    print("Recent Events")
    print("-------------")
    for sample in samples[-10:]:
        status = sample.status.upper()
        delta_text = f"Δ={sample.delta:+.3f}" if sample.delta is not None else "Δ=n/a"
        ts = sample.timestamp.isoformat()
        print(f"[{ts}] {status:<7} {delta_text} lesson={sample.lesson}")
        if sample.status != "passed" and sample.message:
            print(f"           note={sample.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report summary statistics from structural_metrics.jsonl")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Path to structural_metrics.jsonl")
    parser.add_argument("--hours", type=float, default=None, help="Only include records from the last N hours")
    parser.add_argument("--limit", type=int, default=None, help="Limit to the most recent N records (after hours filter)")
    args = parser.parse_args()

    records = iter_records(args.path)
    samples = to_samples(records)
    filtered = filter_samples(samples, hours=args.hours, limit=args.limit)
    compute_summary(filtered)


if __name__ == "__main__":
    main()
