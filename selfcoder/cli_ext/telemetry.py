"""Telemetry-related CLI commands."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.chat.providers import get_registry
from app.chat.providers.base import ProviderError, ProviderNotConfigured
from app.learning.scheduler import advice_for_os_schedule, user_choice_to_time
from ops.telemetry import (
    ReflectionConfig,
    build_knowledge_graph,
    create_experiment,
    get_latest_reflection,
    list_experiments,
    run_reflection,
    summarize_knowledge_graph,
    update_experiment,
    write_knowledge_graph,
)


def _build_embedder(provider_override: Optional[str] = None) -> Callable[[List[str]], List[Sequence[float]]]:
    registry = get_registry()

    def _embed(texts: List[str]) -> List[Sequence[float]]:
        if not texts:
            return []
        return registry.embed(texts, provider_override=provider_override)

    return _embed


def _cmd_reflect(args: argparse.Namespace) -> int:
    kinds: Optional[List[str]] = sorted(set(args.kind)) if args.kind else None
    vector_namespace: Optional[str] = None
    embedder: Optional[Callable[[List[str]], List[Sequence[float]]]] = None
    if getattr(args, "embed", False):
        vector_namespace = args.vector_namespace or ReflectionConfig().vector_namespace
        try:
            embedder = _build_embedder(getattr(args, "embed_provider", None))
        except (ProviderError, ProviderNotConfigured) as exc:
            print(f"[telemetry] failed to initialise embeddings provider: {exc}", file=sys.stderr)
            return 1
    cfg = ReflectionConfig(
        window_hours=args.window_hours,
        limit_events=args.limit,
        include_kinds=kinds,
        vector_namespace=vector_namespace,
    )
    path = run_reflection(cfg, embedder=embedder)
    print(f"[telemetry] reflection written to {path.as_posix()}")
    return 0


def _build_reflect_command(args: argparse.Namespace, defaults: ReflectionConfig) -> List[str]:
    cmd = [sys.executable, "-m", "selfcoder.cli", "telemetry", "reflect"]
    if args.window_hours != defaults.window_hours:
        cmd.extend(["--window-hours", str(args.window_hours)])
    if args.limit != defaults.limit_events:
        cmd.extend(["--limit", str(args.limit)])
    kinds = sorted(set(args.kind or []))
    for kind in kinds:
        cmd.extend(["--kind", kind])
    if getattr(args, "embed", False):
        cmd.append("--embed")
        if args.vector_namespace:
            cmd.extend(["--vector-namespace", args.vector_namespace])
        if getattr(args, "embed_provider", None):
            cmd.extend(["--embed-provider", args.embed_provider])
    return cmd


def _build_knowledge_command(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, "-m", "selfcoder.cli", "telemetry", "knowledge"]
    limit = getattr(args, "knowledge_limit", None)
    if limit:
        cmd.extend(["--limit-events", str(limit)])
    if getattr(args, "knowledge_no_git", False):
        cmd.append("--no-git")
    if getattr(args, "knowledge_out", None):
        cmd.extend(["--out", args.knowledge_out])
    return cmd


def _shell_join(cmd: Sequence[str]) -> str:
    return shlex.join([str(item) for item in cmd])


def _cmd_schedule(args: argparse.Namespace) -> int:
    defaults = ReflectionConfig()
    try:
        when = user_choice_to_time(args.at)
    except ValueError as exc:
        print(f"[telemetry] invalid schedule time: {exc}", file=sys.stderr)
        return 1
    command = _build_reflect_command(args, defaults)
    include_knowledge = getattr(args, "with_knowledge", True)
    knowledge_cmd: Optional[List[str]] = None
    if include_knowledge:
        knowledge_cmd = _build_knowledge_command(args)
    if knowledge_cmd:
        combined = " && ".join([_shell_join(command), _shell_join(knowledge_cmd)])
        command = ["bash", "-lc", combined]
    advice = advice_for_os_schedule(when, command)
    if knowledge_cmd:
        print("[telemetry] reflections will run the knowledge graph export right after completes")
    print(advice)
    return 0


def _cmd_experiments_log(args: argparse.Namespace) -> int:
    reflection_path: Optional[str] = None
    reflection_ts: Optional[str] = None
    if args.reflection:
        if args.reflection == "latest":
            latest = get_latest_reflection()
            if not latest:
                print("[telemetry] no reflections found to attach", file=sys.stderr)
            else:
                reflection_path = latest.get("path")
                data = latest.get("data") or {}
                reflection_ts = data.get("timestamp") if isinstance(data, dict) else None
        else:
            reflection_path = args.reflection

    arms = _parse_csv(args.arms)
    tags = _parse_csv(args.tags)
    metrics = _parse_key_value(args.metric)

    record = create_experiment(
        title=args.title,
        hypothesis=args.hypothesis,
        reflection_path=reflection_path,
        reflection_timestamp=reflection_ts,
        arms=arms,
        tags=tags,
        notes=args.notes,
        metrics=metrics,
        status=args.status,
    )
    print(f"[telemetry] experiment recorded: {record.id}")
    return 0


def _cmd_experiments_update(args: argparse.Namespace) -> int:
    metrics = _parse_key_value(args.metric)
    arms = _parse_csv(args.arms)
    tags = _parse_csv(args.tags)

    payload: Dict[str, Any] = {}
    if args.title is not None:
        payload["title"] = args.title
    if args.hypothesis is not None:
        payload["hypothesis"] = args.hypothesis
    if args.status is not None:
        payload["status"] = args.status
    if args.notes is not None:
        payload["notes"] = args.notes
    if args.outcome is not None:
        payload["outcome"] = args.outcome
    if metrics:
        payload["metrics"] = metrics
    if arms:
        payload["arms"] = arms
    if tags:
        payload["tags"] = tags
    if args.reflection:
        if args.reflection == "latest":
            latest = get_latest_reflection()
            if latest:
                payload["reflection_path"] = latest.get("path")
                data = latest.get("data") or {}
                payload["reflection_timestamp"] = data.get("timestamp") if isinstance(data, dict) else None
        else:
            payload["reflection_path"] = args.reflection

    updated = update_experiment(args.id, **payload)
    if not updated:
        print(f"[telemetry] experiment not found: {args.id}", file=sys.stderr)
        return 1
    print(f"[telemetry] experiment updated: {updated.id}")
    return 0


def _cmd_experiments_list(args: argparse.Namespace) -> int:
    records = list_experiments(limit=args.limit)
    output = []
    for record in records:
        output.append(
            {
                "id": record.id,
                "title": record.title,
                "status": record.status,
                "updated_at": record.updated_at,
                "hypothesis": record.hypothesis,
                "reflection_timestamp": record.reflection_timestamp,
                "outcome": record.outcome,
                "tags": record.tags,
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _cmd_knowledge(args: argparse.Namespace) -> int:
    graph = build_knowledge_graph(
        include_git=not args.no_git,
        telemetry_events_limit=args.limit_events,
    )
    target = write_knowledge_graph(graph, args.out)
    if args.json:
        print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
        print(f"[telemetry] knowledge graph saved to {target.as_posix()}", file=sys.stderr)
    else:
        for line in summarize_knowledge_graph(graph):
            print(line)
        print(f"[telemetry] knowledge graph saved to {target.as_posix()}")
    return 0


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item and item.strip()]


def _parse_key_value(pairs: Optional[List[str]]) -> Dict[str, Any]:
    if not pairs:
        return {}
    data: Dict[str, Any] = {}
    for item in pairs:
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key:
            data[key.strip()] = value.strip()
    return data


def register(subparsers: argparse._SubParsersAction) -> None:
    tele = subparsers.add_parser("telemetry", help="telemetry inspection and summaries")
    tele_sub = tele.add_subparsers(dest="telemetry_command")
    tele_sub.required = True

    defaults = ReflectionConfig()
    ref = tele_sub.add_parser("reflect", help="compute a telemetry reflection over recent events")
    ref.add_argument("--window-hours", type=int, default=defaults.window_hours, help="lookback window in hours (default: 24)")
    ref.add_argument("--limit", type=int, default=defaults.limit_events, help="maximum events to analyse (default: 5000)")
    ref.add_argument("--kind", action="append", help="restrict to specific event kinds (PROMPT, COMPLETION, PLAN, ...)")
    ref.add_argument("--embed", action="store_true", help="store reflection embeddings into the vector store")
    default_ns = defaults.vector_namespace or "reflections"
    ref.add_argument("--vector-namespace", help=f"vector namespace for embeddings (default: {default_ns})")
    ref.add_argument("--embed-provider", help="override embeddings provider id (default: config/env setting)")
    ref.set_defaults(func=_cmd_reflect)

    sched = tele_sub.add_parser("schedule", help="print OS scheduling instructions for telemetry reflections")
    sched.add_argument("--at", default="tonight", help="when to run first (e.g. tonight, tomorrow, now, 02:00)")
    sched.add_argument("--window-hours", type=int, default=defaults.window_hours, help="lookback window in hours")
    sched.add_argument("--limit", type=int, default=defaults.limit_events, help="maximum events per reflection")
    sched.add_argument("--kind", action="append", help="restrict to specific event kinds")
    sched.add_argument("--embed", action="store_true", help="include embeddings when reflections run")
    sched.add_argument("--vector-namespace", help=f"vector namespace for embeddings (default: {default_ns})")
    sched.add_argument("--embed-provider", help="override embeddings provider id for scheduled runs")
    sched.add_argument("--with-knowledge", dest="with_knowledge", action="store_true", default=True, help="also run knowledge graph export after reflections (default)")
    sched.add_argument("--no-knowledge", dest="with_knowledge", action="store_false", help="skip the knowledge graph export")
    sched.add_argument("--knowledge-limit", type=int, help="limit telemetry events when building the knowledge graph")
    sched.add_argument("--knowledge-no-git", action="store_true", help="skip git churn analysis when exporting knowledge graph")
    sched.add_argument("--knowledge-out", help="custom output path for knowledge graph export")
    sched.set_defaults(func=_cmd_schedule)

    experiments = tele_sub.add_parser("experiments", help="record and inspect experiment journal entries")
    exp_sub = experiments.add_subparsers(dest="experiments_command")
    exp_sub.required = True

    exp_log = exp_sub.add_parser("log", help="record a new experiment hypothesis")
    exp_log.add_argument("--title", required=True, help="experiment title")
    exp_log.add_argument("--hypothesis", required=True, help="concise hypothesis or goal")
    exp_log.add_argument(
        "--reflection",
        help="attach a reflection: path to JSON or 'latest'",
    )
    exp_log.add_argument("--arms", help="comma-separated experiment arms (baseline,treatment)")
    exp_log.add_argument("--tags", help="comma-separated tags for filtering")
    exp_log.add_argument("--notes", help="additional notes")
    exp_log.add_argument("--status", default="planned", help="experiment status (planned/running/completed)")
    exp_log.add_argument("--metric", action="append", help="key=value metric pairs to track (can repeat)")
    exp_log.set_defaults(func=_cmd_experiments_log)

    exp_update = exp_sub.add_parser("update", help="update an existing experiment entry")
    exp_update.add_argument("id", help="experiment identifier")
    exp_update.add_argument("--title", help="new title")
    exp_update.add_argument("--hypothesis", help="new hypothesis")
    exp_update.add_argument("--status", help="set status (planned/running/completed)")
    exp_update.add_argument("--notes", help="update notes")
    exp_update.add_argument("--outcome", help="record the outcome summary")
    exp_update.add_argument("--reflection", help="update reflection path or 'latest'")
    exp_update.add_argument("--arms", help="replace arms (comma-separated)")
    exp_update.add_argument("--tags", help="replace tags (comma-separated)")
    exp_update.add_argument("--metric", action="append", help="replace metrics with key=value pairs (repeat)")
    exp_update.set_defaults(func=_cmd_experiments_update)

    exp_list = exp_sub.add_parser("list", help="list experiment entries (JSON)")
    exp_list.add_argument("--limit", type=int, help="limit the number of entries returned")
    exp_list.set_defaults(func=_cmd_experiments_list)

    knowledge = tele_sub.add_parser("knowledge", help="build the telemetry knowledge graph")
    knowledge.add_argument("--out", help="output path (default: out/telemetry/knowledge_graph.json)")
    knowledge.add_argument("--limit-events", type=int, default=5000, help="max telemetry events to scan")
    knowledge.add_argument("--no-git", action="store_true", help="skip git history analysis")
    knowledge.add_argument("--json", action="store_true", help="emit graph JSON to stdout")
    knowledge.set_defaults(func=_cmd_knowledge)


__all__ = ["register"]
