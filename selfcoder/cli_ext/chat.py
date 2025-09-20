from __future__ import annotations

import argparse
import json


def cmd_chat(args: argparse.Namespace) -> int:
    """Launch the interactive Nerion chat (voice + text), run one-shot clarify, or interactive clarify."""
    # One-shot clarify mode for Step 1 interactive learning
    if getattr(args, "clarify", None):
        try:
            from selfcoder.planner import planner
            plan = planner.plan_from_text(args.clarify, target_file=getattr(args, "target_file", None))
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            if plan.get("metadata", {}).get("clarify_required"):
                # Non-zero exit indicates more information is required
                return 2
            return 0
        except Exception as e:
            print(f"[chat] clarify error: {e}")
            return 1

    # Interactive clarify mode
    if getattr(args, "interactive", False):
        try:
            from selfcoder.planner import planner
            while True:
                text = input("[clarify] Enter instruction (or blank to exit): ").strip()
                if not text:
                    return 0
                plan = planner.plan_from_text(text, target_file=getattr(args, "target_file", None))
                print(json.dumps(plan, ensure_ascii=False, indent=2))
                if not plan.get("metadata", {}).get("clarify_required"):
                    print("[clarify] Plan complete.")
                    return 0
                else:
                    print("[clarify] More information needed.")
        except Exception as e:
            print(f"[chat] interactive clarify error: {e}")
            return 1

    # Default: launch the full chat app
    try:
        from app import nerion_chat
    except Exception as e:
        print(f"[chat] unable to import app.nerion_chat: {e}")
        return 1
    try:
        # Apply a profile-scoped env for chat (llm/chat model, network gate)
        try:
            from selfcoder.policy.profile_resolver import decide as _decide, apply_env_scoped as _scope
            dec = _decide('chat')
            with _scope(dec):
                nerion_chat.main()
        except Exception:
            nerion_chat.main()
        return 0
    except SystemExit as se:
        # If nerion_chat uses sys.exit internally, translate to return code
        return int(getattr(se, 'code', 0) or 0)
    except Exception as e:
        print(f"[chat] error: {e}")
        return 1


def register(sub: argparse._SubParsersAction) -> None:
    p_chat = sub.add_parser("chat", help="launch interactive voice/text chat or run a one-shot clarify")
    p_chat.add_argument("--clarify", metavar="TEXT", help="Generate a plan from natural language and print it (with clarify metadata); does not launch chat.")
    p_chat.add_argument("--target-file", metavar="PATH", help="Optional target file for --clarify mode.")
    p_chat.add_argument("--interactive", action="store_true", help="Run clarify loop interactively until plan is complete.")
    p_chat.set_defaults(func=cmd_chat)
