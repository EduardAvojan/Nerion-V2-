from __future__ import annotations
from datetime import datetime, timezone
from selfcoder.scoring import score_plan
from selfcoder.artifacts import PlanArtifact, save_artifact
import click
from selfcoder.orchestrator import apply_plan
from selfcoder.plans.schema import validate_plan
from dataclasses import asdict
import json
import sys

@click.command("simulate")
@click.argument("plan_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Only show what would be done, do not apply changes")
def cli(plan_file: str, dry_run: bool) -> None:
    """
    Simulate applying a plan safely using validation and repo jail.
    """
    try:
        with open(plan_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[ERR] Could not load plan file: {e}", file=sys.stderr)
        raise SystemExit(1)

    try:
        plan = asdict(validate_plan(raw))
    except Exception as e:
        print(f"[SECURITY] Invalid plan rejected: {e}", file=sys.stderr)
        raise SystemExit(1)

    try:
        created = apply_plan(plan, dry_run=dry_run)
    except Exception as e:
        print(f"[ERR] Failed to apply plan: {e}", file=sys.stderr)
        raise SystemExit(1)
    # --- Score plan and log artifact ---
    try:
        score, why = score_plan(plan, None)
        print(f"[SCORE] {score} ({why})")
        artifact = PlanArtifact(
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            origin="cli.simulate",
            score=score,
            rationale=why,
            plan=plan,
            files_touched=[str(p) for p in (created or [])],
            sim=None,
            meta={"dry_run": bool(dry_run)},
        )
        save_artifact(artifact)
    except Exception:
        pass
    if dry_run:
        print(f"[SIMULATION] Would affect {len(created)} files")
    else:
        print(f"[APPLY] Modified {len(created)} files")
