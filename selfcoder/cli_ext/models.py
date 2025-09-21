from __future__ import annotations

import argparse
import os
import time
from typing import List


def _bench_once(provider_id: str) -> float | None:
    try:
        from app.parent.coder import Coder
        from app.chat.providers import ProviderError, ProviderNotConfigured
    except Exception:
        return None
    prev = os.environ.get("NERION_V2_CODE_PROVIDER")
    os.environ["NERION_V2_CODE_PROVIDER"] = provider_id
    coder = Coder(role="code")
    t0 = time.time()
    try:
        out = coder.complete("Reply with OK only.")
    except (ProviderError, ProviderNotConfigured):
        out = None
    finally:
        if prev is not None:
            os.environ["NERION_V2_CODE_PROVIDER"] = prev
        else:
            os.environ.pop("NERION_V2_CODE_PROVIDER", None)
    if not out:
        return None
    return time.time() - t0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("models", help="provider utilities")
    sp = p.add_subparsers(dest="models_cmd", required=True)

    bench = sp.add_parser("bench", help="measure latency for configured providers")
    bench.add_argument("--providers", nargs="*", default=[
        "openai:o4-mini",
        "anthropic:claude-3-5-sonnet",
    ])

    def _run_bench(args: argparse.Namespace) -> int:
        rows: List[str] = ["provider\tlatency_s"]
        for provider_id in args.providers:
            dt = _bench_once(provider_id)
            rows.append(f"{provider_id}\t{dt if dt is not None else 'n/a'}")
        print("\n".join(rows))
        return 0

    bench.set_defaults(func=_run_bench)

    ensure = sp.add_parser("ensure", help="set the default code provider")
    ensure.add_argument("--provider", help="provider identifier (e.g., openai:o4-mini)")

    def _run_ensure(args: argparse.Namespace) -> int:
        provider_id = args.provider or os.getenv("NERION_V2_CODE_PROVIDER")
        if not provider_id:
            print("Provide --provider (e.g., openai:o4-mini).")
            return 2
        os.environ["NERION_V2_CODE_PROVIDER"] = provider_id
        print(f"Default code provider set to {provider_id}. Persist this in .env for future runs.")
        return 0

    ensure.set_defaults(func=_run_ensure)

    router = sp.add_parser("router", help="explain router decision for a task")
    router.add_argument("--task", choices=["chat", "code"], default="code")
    router.add_argument("-i", "--instruction")
    router.add_argument("-f", "--file")

    def _run_router(args: argparse.Namespace) -> int:
        try:
            from selfcoder.llm_router import apply_router_env
            provider, model, base = apply_router_env(
                instruction=getattr(args, 'instruction', None),
                file=getattr(args, 'file', None),
                task=getattr(args, 'task', 'code'),
            )
            import json
            print(json.dumps({"provider": provider, "model": model, "base": base}, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"router explain error: {exc}")
            return 2

    router.set_defaults(func=_run_router)
