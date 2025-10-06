"""DEPRECATED: Orchestrate generation workers until the queue is drained.

This template-based generation system has been deprecated in favor of LLM-based generators.
See DEPRECATED_TEMPLATE_SYSTEM.md for details.
"""
from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - script invocation support
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from nerion_digital_physicist.generation.worker import process_next
else:
    from .worker import process_next


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation workers on the queue")
    parser.add_argument("queue_root", type=str, help="Directory containing generation_queue.json")
    parser.add_argument("output_root", type=str, help="Directory where tasks should be generated")
    parser.add_argument("--max-iterations", type=int, default=None, help="Optional cap on worker iterations")
    parser.add_argument("--seed", type=int, default=0, help="Seed for template sampling")
    return parser.parse_args()


def main() -> None:
    print("WARNING: This template-based generation system is DEPRECATED.")
    print("Please use the LLM-based generators instead.")
    print("See DEPRECATED_TEMPLATE_SYSTEM.md for details.")
    
    args = parse_args()
    queue_root = Path(args.queue_root)
    output_root = Path(args.output_root)

    iterations = 0
    while True:
        if args.max_iterations is not None and iterations >= args.max_iterations:
            break
        processed = process_next(queue_root, output_root, seed=args.seed)
        if not processed:
            break
        iterations += 1

    print(f"Processed {iterations} queue items.")


if __name__ == "__main__":
    main()
