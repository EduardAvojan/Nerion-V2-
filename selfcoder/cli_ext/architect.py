"""CLI helpers for architect briefs."""

from __future__ import annotations

import argparse
import json
from typing import List

from selfcoder.planner.architect_briefs import generate_architect_briefs
from selfcoder.planner.prioritizer import prioritize_briefs


def _cmd_briefs(args: argparse.Namespace) -> int:
    briefs = generate_architect_briefs(
        max_briefs=args.max,
        telemetry_window_hours=args.window,
        include_smells=not args.no_smells,
    )
    prioritized = prioritize_briefs(briefs)

    if args.json:
        payload = [item.to_dict() for item in prioritized]
        print(json.dumps(payload, indent=2))
        return 0

    if not prioritized:
        print("[architect] no briefs available")
        return 0

    for item in prioritized:
        brief = item.brief
        print(
            f"# {brief.title} [{brief.component}] "
            f"(priority {brief.priority:.1f}, effective {item.effective_priority:.1f})"
        )
        print(
            f"Decision: {item.decision.upper()} (policy {item.policy}; "
            f"risk {item.risk_score:.1f}, effort {item.effort_score:.1f}, cost ${item.estimated_cost:.0f})"
        )
        if item.reasons:
            print("  Policy gate:")
            for reason in item.reasons:
                print(f"    - {reason}")
        print(f"Summary: {brief.summary}")
        if brief.rationale:
            print("  Rationale:")
            for line in brief.rationale:
                print(f"    - {line}")
        if brief.acceptance_criteria:
            print("  Acceptance:")
            for line in brief.acceptance_criteria:
                print(f"    - {line}")
        print("")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    arch = subparsers.add_parser(
        "architect", help="Generate architect upgrade briefs from telemetry and roadmap"
    )
    arch.add_argument("--max", type=int, default=5, help="maximum number of briefs to show")
    arch.add_argument(
        "--window", type=int, default=48, help="telemetry window in hours for hotspot analysis"
    )
    arch.add_argument("--json", action="store_true", help="emit briefs as JSON")
    arch.add_argument("--no-smells", action="store_true", help="skip static analysis inputs")
    arch.set_defaults(func=_cmd_briefs)


__all__ = ["register"]
