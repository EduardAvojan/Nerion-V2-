"""Architect brief generator for Nerion Phase 3 planning."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ops.telemetry import load_operator_snapshot
from ops.telemetry.knowledge_graph import knowledge_hotspots, load_knowledge_graph

from selfcoder.analysis.smells import normalize_reports
from selfcoder.coverage_utils import compare_to_baseline, load_baseline, suggest_targets


BRIEF_VERSION = 1
DEFAULT_MAX_BRIEFS = 5
REPORTS_DIR = Path("out/analysis_reports")
COVERAGE_FILE = Path("coverage.json")
ROADMAP_FILE = Path("AGENTS.md")


@dataclass(slots=True)
class ArchitectBrief:
    """Structured upgrade brief for the architect planning loop."""

    id: str
    component: str
    title: str
    summary: str
    rationale: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    priority: float = 0.0
    signals: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    version: int = BRIEF_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "component": self.component,
            "title": self.title,
            "summary": self.summary,
            "rationale": list(self.rationale),
            "acceptance_criteria": list(self.acceptance_criteria),
            "priority": float(self.priority),
            "signals": dict(self.signals),
            "tags": list(self.tags),
            "generated_at": self.generated_at,
            "version": self.version,
        }
        return data


def generate_architect_briefs(
    *,
    max_briefs: int = DEFAULT_MAX_BRIEFS,
    telemetry_window_hours: int = 48,
    include_smells: bool = True,
    coverage_path: Optional[Path] = None,
    roadmap_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> List[ArchitectBrief]:
    """Generate architect briefs combining telemetry, coverage, smells, and roadmap goals."""

    snapshot = load_operator_snapshot(window_hours=telemetry_window_hours)
    knowledge = snapshot.get("knowledge_graph") if isinstance(snapshot, dict) else None
    knowledge_data: Optional[Dict[str, Any]] = None
    if isinstance(knowledge, dict):
        knowledge_data = knowledge
    else:
        try:
            loaded = load_knowledge_graph()
        except Exception:
            loaded = None
        if loaded:
            kg_data, _ = loaded
            knowledge_data = {
                "hotspots": knowledge_hotspots(kg_data, limit=max_briefs * 2),
                "stats": kg_data.get("stats"),
            }
    hotspots = knowledge_data.get("hotspots") if isinstance(knowledge_data, dict) else None
    hotspot_entries: List[Dict[str, Any]] = []
    if isinstance(hotspots, list):
        hotspot_entries = [item for item in hotspots if isinstance(item, dict)]

    coverage_source = coverage_path or COVERAGE_FILE
    roadmap_source = roadmap_path or ROADMAP_FILE
    reports_source = reports_dir or REPORTS_DIR

    coverage_info = _load_coverage_signals(coverage_source)
    smell_map = _load_smell_signals(reports_source) if include_smells else {}
    roadmap_goals = _load_roadmap_goals(roadmap_source)
    anomaly_lines = snapshot.get("anomalies") if isinstance(snapshot, dict) else []

    briefs: List[ArchitectBrief] = []
    seen_components: set[str] = set()
    for entry in hotspot_entries[: max_briefs * 2]:
        component = str(entry.get("component") or "").strip()
        if not component or component in seen_components:
            continue
        seen_components.add(component)
        brief = _build_component_brief(
            component=component,
            hotspot=entry,
            coverage=coverage_info.get(component),
            smell=smell_map.get(component),
            anomalies=anomaly_lines,
        )
        briefs.append(brief)
        if len(briefs) >= max_briefs:
            break

    # Add roadmap-driven briefs if space remains
    if len(briefs) < max_briefs and roadmap_goals:
        remaining = max_briefs - len(briefs)
        for goal in roadmap_goals[:remaining]:
            briefs.append(goal)

    return briefs


def _build_component_brief(
    *,
    component: str,
    hotspot: Dict[str, Any],
    coverage: Optional[Dict[str, Any]],
    smell: Optional[Dict[str, Any]],
    anomalies: Optional[Iterable[Any]],
) -> ArchitectBrief:
    risk_score = float(hotspot.get("risk_score") or 0.0)
    test_failures = int(hotspot.get("test_failures") or 0)
    apply_failures = int(hotspot.get("apply_failures") or 0)
    fix_commits = int(hotspot.get("recent_fix_commits") or 0)

    rationale: List[str] = []
    acceptance: List[str] = []
    tags: List[str] = ["component", component]

    summary_parts = [
        f"Risk score {risk_score:.1f}",
        f"recent fixes {fix_commits}",
    ]
    if test_failures:
        summary_parts.append(f"{test_failures} recent test failures")
        rationale.append(
            f"Telemetry shows {test_failures} failing test events for {component} in the last window."
        )
        acceptance.append("Latest telemetry window shows zero failing TEST events for this component.")
    if apply_failures:
        summary_parts.append(f"{apply_failures} apply failures")
        rationale.append(
            f"Apply attempts against {component} have failed {apply_failures} time(s)."
        )
        acceptance.append("Auto-apply attempts succeed without rollback for this component.")
    if fix_commits:
        rationale.append(
            f"Git history recorded {fix_commits} recent fix commit(s) touching {component}."
        )

    coverage_missing = None
    coverage_delta = None
    if coverage:
        coverage_missing = coverage.get("missing_lines")
        coverage_delta = coverage.get("delta_vs_baseline")
        if coverage_missing:
            rationale.append(
                f"Coverage report lists {coverage_missing} missing line(s) in {coverage.get('example_file')}"
            )
            acceptance.append("Reduce missing lines by adding targeted tests for key paths.")
        if coverage_delta is not None:
            delta_desc = f"{coverage_delta:+.2f}pp vs. baseline"
            rationale.append(f"Coverage trend: {delta_desc} for overall repository.")

    smell_summary = None
    if smell:
        smell_count = smell.get("count", 0)
        smell_top = smell.get("top")
        if smell_count:
            smell_summary = f"{smell_count} static analysis finding(s)"
            rationale.append(
                f"Static analyzers flagged {smell_count} issue(s) under {component} (e.g., {smell_top})."
            )
            acceptance.append("Address outstanding static analysis findings (lint, security, complexity).")

    if anomalies:
        for line in anomalies:
            text = str(line)
            if component in text:
                rationale.append(f"Reflection anomaly references component: {text}")
                acceptance.append("Reflection anomalies mentioning this component are resolved.")
                break

    if not acceptance:
        acceptance.append("Document the mitigation steps and link post-change telemetry showing risk reduction.")

    summary = f"Stabilise {component}: " + ", ".join(summary_parts)
    priority = risk_score + (test_failures * 2) + apply_failures + (fix_commits * 0.5)
    if coverage_missing:
        priority += min(coverage_missing / 20.0, 2.0)
    if smell_summary:
        summary += f", {smell_summary}"

    brief = ArchitectBrief(
        id=f"brief-{component}-{uuid.uuid4().hex[:8]}",
        component=component,
        title=f"Reduce risk in {component}",
        summary=summary,
        rationale=rationale,
        acceptance_criteria=acceptance,
        priority=priority,
        signals={
            "hotspot": hotspot,
            "coverage": coverage,
            "smells": smell,
        },
        tags=tags,
    )
    return brief


def _load_coverage_signals(path: Path) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    cov_path = Path(path)
    if not cov_path.exists():
        return data
    try:
        payload = json.loads(cov_path.read_text(encoding="utf-8"))
    except Exception:
        return data
    baseline_raw = load_baseline()
    baseline = baseline_raw if isinstance(baseline_raw, dict) else None
    current_pct, delta = compare_to_baseline(payload, baseline)
    suggestions = suggest_targets(payload, top_n=10)
    for filename, missing in suggestions:
        component = _component_from_path(filename)
        entry = data.setdefault(component, {
            "component": component,
            "missing_lines": 0,
            "files": [],
            "coverage_pct": current_pct,
            "delta_vs_baseline": delta,
        })
        entry["missing_lines"] += missing
        files: List[str] = entry.setdefault("files", [])  # type: ignore[assignment]
        if filename not in files:
            files.append(filename)
        if not entry.get("example_file"):
            entry["example_file"] = filename
    return data


def _load_smell_signals(reports_dir: Path) -> Dict[str, Dict[str, Any]]:
    report = _latest_report_path(reports_dir)
    if report is None or not report.exists():
        return {}
    try:
        raw = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return {}
    smells = normalize_reports(raw)
    buckets: Dict[str, Dict[str, Any]] = {}
    for smell in smells:
        component = _component_from_path(smell.path)
        entry = buckets.setdefault(component, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append({
                "tool": smell.tool,
                "code": smell.code,
                "message": smell.message,
                "path": smell.path,
                "line": smell.line,
            })
    for component, entry in buckets.items():
        examples = entry.get("examples") or []
        if examples:
            top = examples[0]
            entry["top"] = f"{top.get('tool')} {top.get('code')}"
    return buckets


def _latest_report_path(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_roadmap_goals(path: Path) -> List[ArchitectBrief]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return []

    phase_block = _extract_phase_block(text, "Phase 3")
    tasks = _parse_unchecked_items(phase_block)
    briefs: List[ArchitectBrief] = []
    for idx, line in enumerate(tasks):
        clean = re.sub(r"^[-*]\s*\[\s*\]\s*", "", line).strip()
        if not clean:
            continue
        rationale = ["Roadmap item from Phase 3 remains outstanding."]
        acceptance = ["Roadmap task is completed and marked with [x] in AGENTS.md."]
        briefs.append(
            ArchitectBrief(
                id=f"roadmap-{idx}-{uuid.uuid4().hex[:6]}",
                component="roadmap",
                title=f"Roadmap: {clean}",
                summary=clean,
                rationale=rationale,
                acceptance_criteria=acceptance,
                priority=1.0,
                signals={"task": clean, "source": str(p)},
                tags=["roadmap", "phase-3"],
            )
        )
    return briefs


def _extract_phase_block(text: str, phase_heading: str) -> str:
    pattern = re.compile(rf"###\s+{re.escape(phase_heading)}(?P<body>[\s\S]*?)(?=\n###\s|\Z)", re.IGNORECASE)
    match = pattern.search(text)
    return match.group("body") if match else ""


def _parse_unchecked_items(section_text: str) -> List[str]:
    lines = section_text.splitlines()
    unchecked = [line for line in lines if re.search(r"[-*]\s*\[\s*\]\s*", line)]
    return unchecked


def _component_from_path(path: Optional[str]) -> str:
    if not path:
        return "root"
    normalized = str(path).strip().lstrip("./")
    parts = normalized.split("/")
    if not parts:
        return "root"
    first = parts[0]
    return first or "root"


__all__ = ["ArchitectBrief", "generate_architect_briefs"]
