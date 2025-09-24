"""Generate metrics report for Nerion Phase 3 data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - script invocation support
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from nerion_digital_physicist.experiments.metrics import summarize_metrics
else:
    from .metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Nerion Phase 3 metrics")
    parser.add_argument("root", type=str, help="Directory containing telemetry/replay/catalog files")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summary = summarize_metrics(root)
    summary_dict = summary.to_dict()

    print(json.dumps(summary_dict, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(summary_dict, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
