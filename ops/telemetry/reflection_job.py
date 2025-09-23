"""Telemetry reflection job implementation."""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from core.memory.vector_store import VectorStore

from .schema import EventKind
from .store import TelemetryStore
from .reflection import (
    Anomaly,
    ClusterSummary,
    ReflectionConfig,
    ReflectionSummary,
    reflection_output_path,
)


EmbedFn = Callable[[List[str]], List[Sequence[float]]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_state() -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip() or None
        except Exception:
            return None

    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["status", "--short"])),
    }


def run_reflection(
    config: Optional[ReflectionConfig] = None,
    *,
    embedder: Optional[EmbedFn] = None,
) -> Path:
    config = config or ReflectionConfig()
    window_start, window_end = config.window_bounds()

    store = TelemetryStore()
    events = store.events_between(
        since=window_start,
        until=window_end,
        kinds=config.include_kinds,
        limit=config.limit_events,
        descending=False,
    )
    store.close()

    event_count = len(events)
    counts_by_kind = Counter(e.get("kind") for e in events)
    counts_by_source = Counter(e.get("source") for e in events)

    provider_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "latency_ms_total": 0.0,
        "latency_samples": 0,
        "cost_usd": 0.0,
        "errors": 0,
    })

    for event in events:
        if event.get("kind") != EventKind.COMPLETION.value:
            continue
        metadata = event.get("metadata") or {}
        provider = metadata.get("provider") or _infer_provider(event.get("source"))
        stats = provider_stats[provider or "unknown"]
        stats["count"] += 1
        lat = metadata.get("latency_ms")
        if isinstance(lat, (int, float)) and lat >= 0:
            stats["latency_ms_total"] += float(lat)
            stats["latency_samples"] += 1
        cost = metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            stats["cost_usd"] += float(cost)
        status = (metadata.get("status") or "").lower()
        if "error" in status or metadata.get("error"):
            stats["errors"] += 1

    provider_summary: Dict[str, Dict[str, Any]] = {}
    for provider, stats in provider_stats.items():
        count = stats["count"] or 1
        avg_latency = stats["latency_ms_total"] / stats["latency_samples"] if stats["latency_samples"] else None
        error_rate = stats["errors"] / count
        provider_summary[provider] = {
            "count": count,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
            "max_latency_ms": None,
            "cost_usd": round(stats["cost_usd"], 4) if stats["cost_usd"] else 0.0,
            "error_rate": round(error_rate, 3),
        }

    summary_block: Dict[str, Any] = {
        "counts_by_kind": dict(counts_by_kind),
        "counts_by_source": counts_by_source.most_common(15),
        "providers": provider_summary,
    }

    lessons: List[str] = []
    if config.include_memory:
        memory_summary, memory_lessons = _memory_rollup(window_start, window_end)
        if memory_summary:
            summary_block["memory"] = memory_summary
        if memory_lessons:
            lessons.extend(memory_lessons)

    clusters = _build_clusters(events)
    anomalies = _detect_anomalies(summary_block, provider_summary, event_count, config)
    if anomalies:
        lessons.extend(
            [
                f"Anomaly: {item.detail}" if item.detail else f"Anomaly detected ({item.kind})"
                for item in anomalies
            ]
        )

    reflection = ReflectionSummary(
        timestamp=_utcnow_iso(),
        window_start=window_start,
        window_end=window_end,
        event_count=event_count,
        summary=summary_block,
        clusters=clusters,
        anomalies=anomalies,
        lessons=lessons,
        metadata={
            "config": {
                "window_hours": config.window_hours,
                "limit_events": config.limit_events,
                "include_kinds": list(config.include_kinds or []),
                "include_memory": bool(config.include_memory),
            },
            "git": _git_state(),
        },
    )

    path = reflection_output_path()
    payload = json.dumps(asdict(reflection), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")

    if embedder and config.vector_namespace:
        _persist_embedding(reflection, config.vector_namespace, embedder)

    return path


def _build_clusters(events: List[Dict[str, Any]], top_n: int = 6) -> List[ClusterSummary]:
    tag_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        tags = event.get("tags") or []
        if not tags:
            continue
        kind = event.get("kind") or "unknown"
        for tag in tags:
            tag_groups[(kind, str(tag))].append(event)

    ranked = sorted(tag_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]
    clusters: List[ClusterSummary] = []
    for (kind, tag), group in ranked:
        example_event = group[0]
        event_ids = [str(ev.get("id")) for ev in group[:10] if ev.get("id") is not None]
        clusters.append(
            ClusterSummary(
                tag=f"{kind}:{tag}",
                event_ids=event_ids,
                example=_format_example(example_event),
                metadata={
                    "count": len(group),
                    "sources": _top_sources(group),
                },
            )
        )
    return clusters


def _detect_anomalies(
    summary_block: Dict[str, Any],
    provider_summary: Dict[str, Dict[str, Any]],
    event_count: int,
    config: ReflectionConfig,
) -> List[Anomaly]:
    anomalies: List[Anomaly] = []

    counts = summary_block.get("counts_by_kind", {})
    prompts = counts.get(EventKind.PROMPT.value, 0)
    completions = counts.get(EventKind.COMPLETION.value, 0)
    if completions and prompts and completions / max(prompts, 1) < 0.5:
        anomalies.append(
            Anomaly(
                kind="conversion_drop",
                detail="Completion volume significantly lower than prompts",
                score=0.6,
                metadata={"prompts": prompts, "completions": completions},
            )
        )

    for provider, stats in provider_summary.items():
        avg_latency = stats.get("avg_latency_ms") or 0
        if avg_latency > 4000:
            anomalies.append(
                Anomaly(
                    kind="high_latency",
                    detail=f"Average latency {avg_latency}ms for {provider}",
                    score=min(1.0, avg_latency / 8000.0),
                    metadata={"provider": provider, "avg_latency_ms": avg_latency},
                )
            )
        error_rate = stats.get("error_rate") or 0.0
        if error_rate > 0.15 and stats.get("count", 0) >= 5:
            anomalies.append(
                Anomaly(
                    kind="error_rate",
                    detail=f"Error rate {error_rate:.1%} for {provider}",
                    score=min(1.0, error_rate * 2),
                    metadata={"provider": provider, "error_rate": error_rate},
                )
            )

    if event_count >= config.limit_events:
        anomalies.append(
            Anomaly(
                kind="event_cap",
                detail="Reflection hit event fetch cap; consider increasing limit",
                score=0.4,
                metadata={"limit": config.limit_events},
            )
        )

    return anomalies


def _memory_rollup(
    window_start: str,
    window_end: str,
    recent_limit: int = 10,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    try:
        from core.memory import journal_store
    except Exception:
        return None, []

    try:
        events = journal_store.query(since=window_start, until=window_end, limit=500)
    except Exception:
        return None, []

    if not events:
        return None, []

    counts = Counter()
    for event in events:
        kind = event.get("kind") or "unknown"
        counts[str(kind)] += 1

    lessons: List[str] = []
    highlights: List[Dict[str, Any]] = []
    recent = sorted(events, key=lambda item: item.get("timestamp", ""))[-recent_limit:]
    for entry in recent:
        lesson = _format_memory_lesson(entry)
        if lesson:
            lessons.append(lesson)
        highlights.append(
            {
                "id": entry.get("id"),
                "timestamp": entry.get("timestamp"),
                "kind": entry.get("kind"),
                "rationale": entry.get("rationale"),
                "subject": entry.get("subject"),
            }
        )

    summary = {
        "counts_by_kind": dict(counts),
        "recent": highlights,
        "total": len(events),
    }

    return summary, lessons


def _format_memory_lesson(entry: Dict[str, Any]) -> Optional[str]:
    kind = str(entry.get("kind") or "memory")
    rationale = str(entry.get("rationale") or "Updated memory")
    subject = entry.get("subject")
    if subject:
        return f"memory:{kind} · {rationale} ({subject})"
    return f"memory:{kind} · {rationale}"


def _format_example(event: Dict[str, Any]) -> str:
    source = event.get("source") or "unknown"
    subject = event.get("subject") or ""
    meta = event.get("metadata") or {}
    provider = meta.get("provider") or _infer_provider(source) or ""
    detail = meta.get("rationale") or meta.get("status") or meta.get("target_file") or ""
    pieces = [source]
    if provider:
        pieces.append(f"provider={provider}")
    if subject:
        pieces.append(f"subject={subject}")
    if detail:
        pieces.append(str(detail))
    return " | ".join(pieces)


def _top_sources(events: List[Dict[str, Any]], top_n: int = 3) -> List[str]:
    counts = Counter(ev.get("source") or "unknown" for ev in events)
    return [src for src, _ in counts.most_common(top_n)]


def _infer_provider(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    if ":" in source:
        # e.g. app.chat.providers.openai
        return source.split(":", 1)[0]
    parts = source.split(".")
    if len(parts) >= 4 and parts[0] == "app" and parts[1] == "chat" and parts[2] == "providers":
        return parts[3]
    return None


def _persist_embedding(reflection: ReflectionSummary, namespace: str, embedder: EmbedFn) -> None:
    text = _build_reflection_text(reflection)
    try:
        embeddings = embedder([text])
    except Exception:
        return
    if not embeddings:
        return
    embedding = embeddings[0]
    vs: Optional[VectorStore] = None
    try:
        vs = VectorStore()
        vs.add(
            namespace=namespace,
            embedding=embedding,
            text=text,
            metadata={
                "timestamp": reflection.timestamp,
                "window_start": reflection.window_start,
                "window_end": reflection.window_end,
            },
            tags=["reflection"],
        )
    except Exception:
        pass
    finally:
        if vs is not None:
            try:
                vs.close()
            except Exception:
                pass


def _build_reflection_text(reflection: ReflectionSummary) -> str:
    providers = reflection.summary.get("providers", {})
    anomalies = reflection.anomalies
    lines = [
        f"Reflection {reflection.timestamp}",
        f"Window: {reflection.window_start} → {reflection.window_end}",
        f"Events: {reflection.event_count}",
    ]
    for provider, stats in providers.items():
        lines.append(
            f"Provider {provider}: count={stats.get('count')} avg_latency={stats.get('avg_latency_ms')}ms error_rate={stats.get('error_rate')}"
        )
    if anomalies:
        lines.append("Anomalies:")
        for anom in anomalies:
            lines.append(f"- {anom.kind}: {anom.detail}")
    return "\n".join(lines)


__all__ = ["run_reflection"]
