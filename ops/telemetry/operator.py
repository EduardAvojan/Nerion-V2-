"""Operator-facing telemetry helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ops.security.fs_guard import ensure_in_repo_auto

from .reflection import REFLECTION_DIR
from .store import TelemetryStore
from .knowledge_graph import load_knowledge_graph, knowledge_hotspots


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _safe_close(store: Optional[TelemetryStore]) -> None:
    if store is None:
        return
    try:
        store.close()
    except Exception:
        pass


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _compute_apply_metrics(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    if total == 0:
        return {
            "total": 0,
            "success": 0,
            "rolled_back": 0,
            "simulated": 0,
            "rate": 0.0,
        }
    success = 0
    rolled_back = 0
    simulated = 0
    for event in events:
        meta = event.get("metadata") or {}
        if _as_bool(meta.get("outcome")):
            success += 1
        if _as_bool(meta.get("rolled_back")):
            rolled_back += 1
        if _as_bool(meta.get("simulate")):
            simulated += 1
    rate = success / float(total) if total else 0.0
    return {
        "total": total,
        "success": success,
        "rolled_back": rolled_back,
        "simulated": simulated,
        "rate": rate,
    }


def _compute_policy_gate_counts(events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for event in events:
        meta = event.get("metadata") or {}
        decision = str(meta.get("decision") or "unknown").lower()
        counter[decision] += 1
    return dict(counter)


def _compute_governor_counts(events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for event in events:
        meta = event.get("metadata") or {}
        decision = str(meta.get("decision") or meta.get("code") or "unknown").lower()
        counter[decision] += 1
    return dict(counter)


def get_latest_reflection() -> Optional[Dict[str, Any]]:
    directory = Path(ensure_in_repo_auto(REFLECTION_DIR))
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("reflection_*.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            return {
                "path": path.as_posix(),
                "data": data,
            }
        except Exception:
            continue
    return None


def load_operator_snapshot(window_hours: int = 24) -> Dict[str, Any]:
    """Return a JSON-friendly snapshot of recent telemetry activity."""

    window_hours = max(1, int(window_hours or 24))
    now = _utc_now()
    since_dt = now - timedelta(hours=window_hours)
    since = _iso(since_dt)
    until = _iso(now)

    counts_by_kind: Dict[str, int] = {}
    top_sources: List[Dict[str, Any]] = []
    counts_total = 0

    store: Optional[TelemetryStore] = None
    apply_metrics: Dict[str, Any]
    policy_counts: Dict[str, int]
    governor_counts: Dict[str, int]
    try:
        store = TelemetryStore()
        counts_by_kind = store.counts_by_kind(since=since, until=until)
        counts_total = sum(int(v) for v in counts_by_kind.values())
        raw_sources = store.counts_by_source(since=since, until=until)
        sorted_sources = sorted(raw_sources.items(), key=lambda kv: kv[1], reverse=True)
        top_sources = [
            {"source": str(src), "count": int(count)}
            for src, count in sorted_sources[:5]
        ]
        apply_events = store.events_between(since=since, until=until, kinds=["apply"], descending=True)
        apply_metrics = _compute_apply_metrics(apply_events)
        policy_events = store.events_between(since=since, until=until, kinds=["apply_policy"], descending=True)
        policy_counts = _compute_policy_gate_counts(policy_events)
        governor_events = store.events_between(since=since, until=until, kinds=["governor"], descending=True)
        governor_counts = _compute_governor_counts(governor_events)
    except Exception:
        counts_by_kind = {}
        top_sources = []
        counts_total = 0
        apply_metrics = {
            "total": 0,
            "success": 0,
            "rolled_back": 0,
            "simulated": 0,
            "rate": 0.0,
        }
        policy_counts = {}
        governor_counts = {}
    finally:
        _safe_close(store)

    reflection_entry = get_latest_reflection()
    reflection_data = reflection_entry["data"] if reflection_entry else None

    providers: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []
    lessons: List[str] = []
    memory_block: Optional[Dict[str, Any]] = None
    if isinstance(reflection_data, dict):
        summary = reflection_data.get("summary") or {}
        provider_block = summary.get("providers") or {}
        for provider, stats in provider_block.items():
            try:
                providers.append(
                    {
                        "provider": str(provider),
                        "count": int(stats.get("count", 0)),
                        "avg_latency_ms": stats.get("avg_latency_ms"),
                        "cost_usd": stats.get("cost_usd"),
                        "error_rate": stats.get("error_rate"),
                    }
                )
            except Exception:
                continue
        providers.sort(key=lambda entry: entry.get("count", 0), reverse=True)
        raw_anomalies = reflection_data.get("anomalies") or []
        for item in raw_anomalies[:5]:
            if not isinstance(item, dict):
                continue
            anomalies.append(
                {
                    "kind": str(item.get("kind", "unknown")),
                    "detail": str(item.get("detail", "")),
                    "score": item.get("score"),
                }
            )
        lessons = [str(item) for item in reflection_data.get("lessons", []) if item]
        memory_block = summary.get("memory") if isinstance(summary, dict) else None

    knowledge_entry: Optional[Dict[str, Any]] = None
    try:
        kg_loaded = load_knowledge_graph()
        if kg_loaded:
            kg_data, kg_path = kg_loaded
            hotspots = knowledge_hotspots(kg_data, limit=5)
            knowledge_entry = {
                "path": kg_path.as_posix(),
                "generated_at": kg_data.get("generated_at"),
                "stats": kg_data.get("stats"),
                "hotspots": hotspots,
            }
    except Exception:
        knowledge_entry = None

    total_cost = 0.0
    for provider in providers:
        cost_val = provider.get("cost_usd")
        if isinstance(cost_val, (int, float)):
            total_cost += float(cost_val)

    snapshot: Dict[str, Any] = {
        "generated_at": until,
        "window": {
            "hours": window_hours,
            "since": since,
            "until": until,
        },
        "counts_by_kind": {k: int(v) for k, v in counts_by_kind.items()},
        "counts_total": int(counts_total),
        "top_sources": top_sources,
        "providers": providers,
        "anomalies": anomalies,
        "apply_metrics": apply_metrics,
        "policy_gates": policy_counts,
        "governor_decisions": governor_counts,
        "provider_cost_total": total_cost,
    }

    if reflection_entry:
        snapshot["latest_reflection"] = {
            "path": reflection_entry["path"],
            "timestamp": reflection_data.get("timestamp"),
            "event_count": reflection_data.get("event_count"),
            "summary": reflection_data.get("summary"),
            "anomalies": reflection_data.get("anomalies") or [],
        }
    else:
        snapshot["latest_reflection"] = None

    prompt_count = snapshot["counts_by_kind"].get("PROMPT", 0)
    completion_count = snapshot["counts_by_kind"].get("COMPLETION", 0)
    snapshot["prompt_completion_ratio"] = {
        "prompts": int(prompt_count),
        "completions": int(completion_count),
    }

    if lessons:
        snapshot["lessons"] = lessons
    if memory_block:
        snapshot["memory"] = memory_block
    snapshot["knowledge_graph"] = knowledge_entry

    return snapshot


def summarize_snapshot(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce UI-friendly highlights from :func:`load_operator_snapshot`."""

    if not snapshot:
        return {
            "metrics": [],
            "subtitle": "Telemetry inactive",
            "anomalies": [],
        }

    window = snapshot.get("window") or {}
    hours = window.get("hours") or 0
    total = snapshot.get("counts_total") or 0
    ratio = snapshot.get("prompt_completion_ratio") or {}
    prompts = ratio.get("prompts", 0)
    completions = ratio.get("completions", 0)

    metrics = [
        {"label": f"{hours}h events", "value": str(total)},
        {"label": "Prompts→Completions", "value": f"{prompts}→{completions}"},
    ]

    apply = snapshot.get("apply_metrics") or {}
    if apply.get("total"):
        rate = apply.get("rate")
        rate_str = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else "—"
        metrics.append(
            {
                "label": "Apply success",
                "value": f"{apply.get('success', 0)}/{apply.get('total', 0)} ({rate_str})",
            }
        )
        metrics.append(
            {
                "label": "Rollbacks",
                "value": str(apply.get("rolled_back", 0)),
            }
        )

    cost_total = snapshot.get("provider_cost_total")
    if isinstance(cost_total, (int, float)) and cost_total > 0:
        metrics.append({"label": "Cost window", "value": f"${cost_total:.4f}"})

    providers = snapshot.get("providers") or []
    if providers:
        top = providers[0]
        latency = top.get("avg_latency_ms")
        latency_str = f"{int(latency)} ms" if isinstance(latency, (int, float)) else "—"
        cost = top.get("cost_usd")
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "—"
        metrics.append(
            {
                "label": "Top provider",
                "value": f"{top.get('provider')} · {latency_str} · {cost_str}",
            }
        )
    anomalies = snapshot.get("anomalies") or []
    lessons = snapshot.get("lessons") or []
    anomaly_lines = [
        f"{item.get('kind')} – {item.get('detail')}".strip()
        for item in anomalies
        if item.get("detail")
    ]
    lessons_lines = [str(entry) for entry in lessons if entry]
    memory_block = snapshot.get("memory") or {}
    memory_counts = memory_block.get("counts_by_kind") if isinstance(memory_block, dict) else None
    knowledge_block = snapshot.get("knowledge_graph") or {}
    hotspots = knowledge_block.get("hotspots") if isinstance(knowledge_block, dict) else []
    if isinstance(memory_counts, dict) and sum(memory_counts.values()) > 0:
        total_mem = int(sum(int(v) for v in memory_counts.values()))
        metrics.append({"label": "Memory events", "value": str(total_mem)})
    if lessons_lines:
        metrics.append({"label": "Lessons", "value": lessons_lines[0][:80]})
    if anomalies:
        content = anomaly_lines[0] if anomaly_lines else "See telemetry logs"
        metrics.append({"label": f"Anomalies ({len(anomalies)})", "value": content[:80]})
    if isinstance(hotspots, list) and hotspots:
        primary = hotspots[0]
        if isinstance(primary, dict):
            comp = primary.get("component") or "unknown"
            risk = primary.get("risk_score")
            tag = str(comp)
            if isinstance(risk, (int, float)):
                tag = f"{comp} · risk {risk:.1f}"
            metrics.append({"label": "Hotspot", "value": tag[:80]})
    base_subtitle = f"Last {hours}h · {total} events"
    if apply.get("total"):
        base_subtitle = f"{apply.get('success', 0)}/{apply.get('total', 0)} applies · {base_subtitle}"
    subtitle = base_subtitle
    latest = snapshot.get("latest_reflection") or {}
    ts = latest.get("timestamp") or snapshot.get("generated_at")
    if ts:
        subtitle = f"Reflection {ts} · {subtitle}"
    if anomaly_lines:
        subtitle = f"{subtitle} · {anomaly_lines[0][:40]}"

    hotspot_lines: List[str] = []
    if isinstance(hotspots, list):
        for item in hotspots[:3]:
            if not isinstance(item, dict):
                continue
            comp = item.get("component")
            if not comp:
                continue
            risk = item.get("risk_score")
            line = f"Hotspot – {comp}"
            if isinstance(risk, (int, float)):
                line = f"Hotspot – {comp} (risk {risk:.1f})"
            hotspot_lines.append(line)

    highlight_lines = hotspot_lines + anomaly_lines + [f"Lesson – {line}" for line in lessons_lines]

    return {
        "metrics": metrics,
        "subtitle": subtitle,
        "anomalies": highlight_lines,
        "hotspots": hotspots if isinstance(hotspots, list) else [],
    }


__all__ = ["get_latest_reflection", "load_operator_snapshot", "summarize_snapshot"]
