


from __future__ import annotations
import argparse
from typing import Optional
import os

# Phase-2 helpers
from app.learning.proactive import evaluate_readiness
from app.learning.notifications import get_notifier
from app.learning.scheduler import tonight_time, build_self_learn_cmd, advice_for_os_schedule


def add_self_learn_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `self-learn` command group with a `fine-tune` subcommand (placeholder)."""
    p = subparsers.add_parser("self-learn", help="Self-learning utilities (incubator phase)")
    sp = p.add_subparsers(dest="self_learn_cmd", required=True)

    ft = sp.add_parser("fine-tune", help="Prepare/trigger a fine-tuning session (placeholder)")
    ft.add_argument("--data-since", type=str, default="30d",
                    help='relative window like "7d", "30d" or ISO date "2025-07-01"')
    ft.add_argument("--domain", type=str, default="all",
                    help="optional skill domain filter")
    ft.add_argument("--dry-run", action="store_true", help="just validate and print parameters")
    ft.set_defaults(_handler=_handle_fine_tune, func=_handle_fine_tune)

    # Phase-2: propose
    ev = sp.add_parser("propose", help="Evaluate readiness and optionally notify")
    ev.add_argument("--notify", action="store_true", help="show OS notification if ready")
    ev.set_defaults(_handler=_handle_propose, func=_handle_propose)

    # Phase-2: schedule
    sc = sp.add_parser("schedule", help="Suggest a schedule for self-learn based on a user choice")
    sc.add_argument("--choice", choices=["now", "tonight", "remind_later"], required=True)
    sc.add_argument("--data-since", type=str, default="30d")
    sc.add_argument("--domain", type=str, default="all")
    sc.add_argument("--dry-run", action="store_true")
    sc.set_defaults(_handler=_handle_schedule, func=_handle_schedule)


def _handle_fine_tune(args: argparse.Namespace) -> int:
    # Build a small local dataset and a training stub under out/self_learn/
    from pathlib import Path
    import json as _json
    out_root = Path('out/self_learn')
    out_root.mkdir(parents=True, exist_ok=True)
    # Dataset builder: derive simple (input, tag) pairs from experience log
    try:
        from selfcoder.learning.dataset import build_dataset
        ds, stats = build_dataset(since=args.data_since, domain=args.domain)
    except Exception:
        ds, stats = [], {"count": 0}
    (out_root / 'dataset.jsonl').write_text('\n'.join(_json.dumps(r, ensure_ascii=False) for r in ds), encoding='utf-8')
    # Training stub
    train_cfg = {
        "provider": os.getenv('NERION_V2_CODE_PROVIDER', 'openai:gpt-5'),
        "method": "LoRA",
        "epochs": 1,
        "per_intent_tags": True,
        "notes": "Offline-only, user-managed training. Integrate with PEFT if installed.",
    }
    (out_root / 'train_config.json').write_text(_json.dumps(train_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_root / 'README.txt').write_text(
        "This folder contains self-learning artifacts.\n\n"
        "- dataset.jsonl: simple instruction->tag examples derived from experience log.\n"
        "- train_config.json: stub config for local LoRA fine-tuning.\n"
        "To fine-tune, install a local trainer (e.g., PEFT/QLoRA) and point it at this dataset.\n",
        encoding='utf-8')
    print("[self-learn] dataset and training stub written:")
    print(" ", str(out_root))
    print("[self-learn] stats:", _json.dumps(stats))
    return 0


# Phase-2 handlers
def _handle_propose(args: argparse.Namespace) -> int:
    result = evaluate_readiness()
    # Print JSON-ish summary without importing json to keep deps light
    print("[self-learn] readiness:")
    print(" ready:", result.get("ready"))
    print(" summary:", result.get("summary"))
    print(" audit:", result.get("audit"))
    if result.get("ready") and args.notify:
        prop = result.get("proposal") or {}
        title = "Nerion – Evolution Available"
        msg = prop.get("message") or "I have enough new experience data to start a self-tuning session."
        get_notifier().notify(title, msg)
        print("[self-learn] Notification sent.")
    return 0

def _handle_schedule(args: argparse.Namespace) -> int:
    choice = args.choice
    if choice == "now":
        cmd = build_self_learn_cmd(args.data_since, args.domain, args.dry_run)
        print("[self-learn] Run now:")
        print(" ", " ".join(cmd))
        return 0
    if choice == "tonight":
        when = tonight_time(local_hour=2)
    else:  # remind_later → tomorrow 9am by convention
        from datetime import datetime, timedelta
        when = (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    cmd = build_self_learn_cmd(args.data_since, args.domain, args.dry_run)
    print(advice_for_os_schedule(when, cmd))
    return 0


# Allow running as a module for local testing:
#   python -m selfcoder.cli_ext.self_learn fine-tune --data-since 30d

def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nerion-self-learn", description="Self-learning utilities")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Mirror the group directly when run standalone
    ft = subparsers.add_parser("fine-tune", help="Prepare/trigger a fine-tuning session (placeholder)")
    ft.add_argument("--data-since", type=str, default="30d")
    ft.add_argument("--domain", type=str, default="all")
    ft.add_argument("--dry-run", action="store_true")
    ft.set_defaults(_handler=_handle_fine_tune)

    ev = subparsers.add_parser("propose", help="Evaluate readiness and optionally notify")
    ev.add_argument("--notify", action="store_true")
    ev.set_defaults(_handler=_handle_propose)

    sc = subparsers.add_parser("schedule", help="Suggest a schedule for self-learn based on a user choice")
    sc.add_argument("--choice", choices=["now", "tonight", "remind_later"], required=True)
    sc.add_argument("--data-since", type=str, default="30d")
    sc.add_argument("--domain", type=str, default="all")
    sc.add_argument("--dry-run", action="store_true")
    sc.set_defaults(_handler=_handle_schedule)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_standalone_parser()
    args = parser.parse_args(argv)
    return int(args._handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
