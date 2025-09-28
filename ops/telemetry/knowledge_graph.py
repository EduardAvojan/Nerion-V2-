"""Knowledge graph construction for Nerion telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import ast
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ops.security.fs_guard import ensure_in_repo_auto

from .schema import EventKind
from .store import TelemetryStore


SOURCE_SUFFIXES: Set[str] = {".py", ".ts", ".tsx", ".js", ".jsx"}
SKIP_COMPONENT_PARTS: Set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
    "tmp",
    "out",
    "export",
    "coverage",
    ".venv",
    "venv",
    "env",
}
GIT_FIX_KEYWORDS: Tuple[str, ...] = ("fix", "bug", "regress", "hotfix", "patch", "repair")
JS_IMPORT_RE = re.compile(r"import\s+(?:[\w*{}$,\s]+\s+from\s+)?[\"\']([^\"\']+)[\"\']", re.MULTILINE)
JS_EXPORT_RE = re.compile(r"export\s+[^;]*?\s+from\s+[\"\']([^\"\']+)[\"\']", re.MULTILINE)
JS_REQUIRE_RE = re.compile(r"require\(\s*[\"\']([^\"\']+)[\"\']\s*\)")
JS_DYNAMIC_IMPORT_RE = re.compile(r"import\(\s*[\"\']([^\"\']+)[\"\']\s*\)")
DEFAULT_OUTPUT_PATH = Path("out/telemetry/knowledge_graph.json")
MAX_EDGE_EXAMPLES = 5
DEFAULT_GIT_SINCE = "120.days"
DEFAULT_GIT_MAX_COUNT = "500"


@dataclass
class KnowledgeNode:
    """Represents a component or entity in the knowledge graph."""

    id: str
    kind: str
    label: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """Relationship between nodes in the knowledge graph."""

    source: str
    target: str
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Container for Nerion's knowledge graph snapshot."""

    generated_at: str
    root: str
    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a JSON-serialisable dictionary."""

        return {
            "generated_at": self.generated_at,
            "root": self.root,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "stats": dict(self.stats),
        }


def build_knowledge_graph(
    root: Optional[str | Path] = None,
    *,
    include_git: bool = True,
    telemetry_events_limit: int = 5000,
    telemetry_store: Optional[TelemetryStore] = None,
    git_runner: Optional[Callable[[Sequence[str]], Optional[str]]] = None,
) -> KnowledgeGraph:
    """Build a component-level knowledge graph for the Nerion repository."""

    repo_root = ensure_in_repo_auto(Path(root or Path.cwd()))
    known_components = _discover_components(repo_root)

    component_data: Dict[str, Dict[str, Any]] = {}
    _scan_sources(repo_root, known_components, component_data)

    git_stats = _collect_git_stats(repo_root, known_components, include_git, git_runner)
    telemetry_stats = _collect_telemetry_signals(
        repo_root,
        known_components,
        telemetry_events_limit,
        telemetry_store,
    )

    for comp in set(git_stats["changes"]) | set(git_stats["fixes"]) | set(telemetry_stats):
        component_data.setdefault(comp, _empty_component_entry())

    generated_at = _utcnow_iso()
    nodes: List[KnowledgeNode] = []
    edges: List[KnowledgeEdge] = []
    metrics_lookup: Dict[str, Dict[str, Any]] = {}

    for comp in sorted(component_data.keys()):
        entry = component_data[comp]
        dependencies = sorted(entry["dependencies"]) if entry["dependencies"] else []
        external_deps = sorted(entry["external_dependencies"]) if entry["external_dependencies"] else []
        dependency_sources = {
            dep: sorted(paths)[:MAX_EDGE_EXAMPLES]
            for dep, paths in entry["dependency_sources"].items()
        }
        files = sorted(entry["files"])
        loc = int(entry["loc"])
        git_fix_count = int(git_stats["fixes"].get(comp, 0))
        git_change_count = int(git_stats["changes"].get(comp, 0))
        telemetry = telemetry_stats.get(comp, {})
        test_failures = int(telemetry.get("test_failures", 0))
        apply_failures = int(telemetry.get("apply_failures", 0))
        recent_events = int(telemetry.get("recent_events", 0))
        risk_score = (
            git_fix_count * 2
            + git_change_count
            + test_failures * 3
            + apply_failures * 2
        )

        metrics = {
            "file_count": len(files),
            "loc": loc,
            "recent_fix_commits": git_fix_count,
            "recent_changes": git_change_count,
            "test_failures": test_failures,
            "apply_failures": apply_failures,
            "recent_events": recent_events,
            "risk_score": risk_score,
        }
        metrics_lookup[comp] = metrics

        attributes = {
            "dependencies": dependencies,
            "external_dependencies": external_deps,
            "example_files": files[:MAX_EDGE_EXAMPLES],
        }
        if dependency_sources:
            attributes["dependency_sources"] = dependency_sources

        node = KnowledgeNode(
            id=f"component:{comp}",
            kind="component",
            label=comp,
            metrics=metrics,
            attributes=attributes,
        )
        nodes.append(node)

        for dep in dependencies:
            edge_meta: Dict[str, Any] = {}
            examples = dependency_sources.get(dep)
            if examples:
                edge_meta["examples"] = examples
            edges.append(
                KnowledgeEdge(
                    source=f"component:{comp}",
                    target=f"component:{dep}",
                    relation="depends_on",
                    metadata=edge_meta,
                )
            )

    stats = _build_stats(nodes, edges, metrics_lookup)

    return KnowledgeGraph(
        generated_at=generated_at,
        root=repo_root.as_posix(),
        nodes=nodes,
        edges=edges,
        stats=stats,
    )


def write_knowledge_graph(graph: KnowledgeGraph, path: Optional[str | Path] = None) -> Path:
    """Persist the knowledge graph to disk and return the target path."""

    target = ensure_in_repo_auto(Path(path) if path else DEFAULT_OUTPUT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(graph.to_dict(), ensure_ascii=False, indent=2)
    target.write_text(payload, encoding="utf-8")
    return target


def summarize_knowledge_graph(graph: KnowledgeGraph) -> List[str]:
    """Produce CLI-friendly summary lines for the knowledge graph."""

    stats = graph.stats or {}
    component_count = stats.get("component_count", len(graph.nodes))
    edge_count = stats.get("edge_count", len(graph.edges))
    lines = [
        f"[telemetry] knowledge graph built: {component_count} components, {edge_count} dependency edges",
    ]

    top_risky = stats.get("top_risky") or []
    if top_risky:
        first = top_risky[0]
        lines.append(
            "[telemetry] hotspot: "
            f"{first.get('component')} (risk={first.get('risk_score')}, "
            f"fixes={first.get('recent_fix_commits')}, tests={first.get('test_failures')})"
        )
        if len(top_risky) > 1:
            others = ", ".join(
                f"{entry.get('component')} ({entry.get('risk_score')})"
                for entry in top_risky[1:3]
            )
            if others:
                lines.append(f"[telemetry] other hotspots: {others}")

    top_dependencies = stats.get("top_dependencies") or []
    if top_dependencies:
        peak = top_dependencies[0]
        lines.append(
            "[telemetry] most connected: "
            f"{peak.get('component')} → {peak.get('dependency_count')} components"
        )

    return lines


def load_knowledge_graph(path: Optional[str | Path] = None) -> Optional[Tuple[Dict[str, Any], Path]]:
    """Load a previously exported knowledge graph."""

    target = ensure_in_repo_auto(Path(path) if path else DEFAULT_OUTPUT_PATH)
    if not target.exists():
        return None
    try:
        raw = target.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return data, target
    except Exception:
        return None


def knowledge_hotspots(graph: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Return the highest-risk components from the graph snapshot."""

    limit = max(1, limit)
    stats = graph.get("stats") if isinstance(graph, dict) else None
    entries: List[Dict[str, Any]] = []
    if isinstance(stats, dict):
        top_risky = stats.get("top_risky")
        if isinstance(top_risky, list):
            for item in top_risky:
                if not isinstance(item, dict):
                    continue
                component = str(item.get("component") or "").strip()
                if not component:
                    continue
                entries.append(
                    {
                        "component": component,
                        "risk_score": float(item.get("risk_score", 0.0) or 0.0),
                        "recent_fix_commits": int(item.get("recent_fix_commits", 0) or 0),
                        "test_failures": int(item.get("test_failures", 0) or 0),
                        "apply_failures": int(item.get("apply_failures", 0) or 0),
                    }
                )
    if not entries:
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                component = str(node.get("label") or "").strip()
                if not component:
                    continue
                metrics = node.get("metrics")
                if not isinstance(metrics, dict):
                    metrics = {}
                entries.append(
                    {
                        "component": component,
                        "risk_score": float(metrics.get("risk_score", 0.0) or 0.0),
                        "recent_fix_commits": int(metrics.get("recent_fix_commits", 0) or 0),
                        "test_failures": int(metrics.get("test_failures", 0) or 0),
                        "apply_failures": int(metrics.get("apply_failures", 0) or 0),
                    }
                )
    entries.sort(
        key=lambda item: (
            float(item.get("risk_score", 0.0) or 0.0),
            int(item.get("recent_fix_commits", 0) or 0),
            int(item.get("test_failures", 0) or 0),
            int(item.get("apply_failures", 0) or 0),
        ),
        reverse=True,
    )
    return entries[:limit]


def _discover_components(root: Path) -> Set[str]:
    components: Set[str] = set()
    for item in root.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if name.startswith('.') or name in SKIP_COMPONENT_PARTS:
            continue
        components.add(name)
    return components


def _scan_sources(root: Path, known_components: Set[str], component_data: Dict[str, Dict[str, Any]]) -> None:
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in SKIP_COMPONENT_PARTS for part in relative.parts):
            continue
        comp = _component_for_parts(relative.parts, known_components)
        if not comp:
            continue
        entry = component_data.setdefault(comp, _empty_component_entry())
        rel_str = relative.as_posix()
        if rel_str not in entry["files"]:
            entry["files"].append(rel_str)
        text = _read_text(path)
        entry["loc"] += text.count('\n') + 1 if text else 0
        internal, external = _extract_dependencies(path, text, comp, known_components)
        if internal:
            entry["dependencies"].update(internal)
            for dep in internal:
                entry["dependency_sources"].setdefault(dep, set()).add(rel_str)
        if external:
            entry["external_dependencies"].update(external)


def _extract_dependencies(
    path: Path,
    text: str,
    component: str,
    known_components: Set[str],
) -> Tuple[Set[str], Set[str]]:
    suffix = path.suffix
    if suffix == '.py':
        modules = _python_modules(text)
        return _map_modules(modules, component, known_components)
    if suffix in {'.ts', '.tsx', '.js', '.jsx'}:
        modules = _javascript_modules(text)
        return _map_modules(modules, component, known_components)
    return set(), set()


def _python_modules(text: str) -> Set[str]:
    modules: Set[str] = set()
    try:
        tree = ast.parse(text or "")
    except SyntaxError:
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            prefix = "." * level
            if module:
                modules.add(prefix + module)
            elif prefix:
                modules.add(prefix)
    return modules


def _javascript_modules(text: str) -> Set[str]:
    modules: Set[str] = set()
    for pattern in (JS_IMPORT_RE, JS_EXPORT_RE, JS_REQUIRE_RE, JS_DYNAMIC_IMPORT_RE):
        for match in pattern.findall(text or ""):
            if match:
                modules.add(match)
    return modules


def _map_modules(
    modules: Iterable[str],
    current_component: str,
    known_components: Set[str],
) -> Tuple[Set[str], Set[str]]:
    internal: Set[str] = set()
    external: Set[str] = set()
    for module in modules:
        target, is_external = _module_to_component(module, current_component, known_components)
        if not target:
            continue
        if is_external:
            external.add(target)
        elif target != current_component:
            internal.add(target)
    return internal, external


def _module_to_component(
    module: str,
    current_component: str,
    known_components: Set[str],
) -> Tuple[Optional[str], bool]:
    if not module:
        return None, False
    mod = module.strip()
    if not mod:
        return None, False
    mod = mod.replace('\\', '/')
    if mod.startswith('.'):
        return current_component, False
    mod = mod.lstrip('/')
    token = re.split(r"[./]", mod, maxsplit=1)[0]
    if not token:
        return None, False
    if token in known_components:
        return token, False
    if token in SKIP_COMPONENT_PARTS:
        return None, False
    return token, True


def _collect_git_stats(
    root: Path,
    known_components: Set[str],
    enabled: bool,
    git_runner: Optional[Callable[[Sequence[str]], Optional[str]]],
) -> Dict[str, Dict[str, int]]:
    fixes: Dict[str, int] = defaultdict(int)
    changes: Dict[str, int] = defaultdict(int)
    if not enabled:
        return {"fixes": fixes, "changes": changes}

    args = [
        "git",
        "-C",
        root.as_posix(),
        "log",
        f"--since={DEFAULT_GIT_SINCE}",
        "--no-merges",
        f"--max-count={DEFAULT_GIT_MAX_COUNT}",
        "--pretty=%H\t%s",
        "--name-only",
    ]
    output: Optional[str] = None
    if git_runner:
        try:
            output = git_runner(args)
        except Exception:
            output = None
    if output is None:
        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if result.returncode == 0:
                output = result.stdout
        except Exception:
            output = None
    if not output:
        return {"fixes": fixes, "changes": changes}

    current_is_fix = False
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            current_is_fix = False
            continue
        if '\t' in stripped:
            try:
                _, subject = stripped.split('\t', 1)
            except ValueError:
                subject = stripped
            lowered = subject.lower()
            current_is_fix = any(keyword in lowered for keyword in GIT_FIX_KEYWORDS)
            continue
        file_rel = _normalise_git_path(stripped)
        comp = _component_from_string(file_rel, known_components)
        if not comp:
            continue
        changes[comp] += 1
        if current_is_fix:
            fixes[comp] += 1

    return {"fixes": fixes, "changes": changes}


def _collect_telemetry_signals(
    root: Path,
    known_components: Set[str],
    limit: int,
    store: Optional[TelemetryStore],
) -> Dict[str, Dict[str, int]]:
    signals: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "test_failures": 0,
        "apply_failures": 0,
        "recent_events": 0,
    })
    provided = store is not None
    store_obj: Optional[TelemetryStore] = store
    if store_obj is None:
        try:
            store_obj = TelemetryStore()
        except Exception:
            store_obj = None
    if store_obj is None:
        return {}
    try:
        events = store_obj.events_between(
            kinds=[EventKind.TEST.value, EventKind.APPLY.value],
            limit=max(1, min(limit, 20000)),
            descending=True,
        )
    except Exception:
        events = []
    finally:
        if not provided:
            try:
                store_obj.close()
            except Exception:
                pass

    for event in events:
        comp = _component_from_event(event, root, known_components)
        if not comp:
            continue
        entry = signals[comp]
        entry["recent_events"] += 1
        if _is_failure_event(event):
            kind = str(event.get("kind") or "").lower()
            if kind == EventKind.TEST.value:
                entry["test_failures"] += 1
            elif kind == EventKind.APPLY.value:
                entry["apply_failures"] += 1
    return dict(signals)


def _component_from_event(
    event: Dict[str, Any],
    root: Path,
    known_components: Set[str],
) -> Optional[str]:
    subject = event.get("subject")
    comp = _component_from_string(subject, known_components)
    if comp:
        return comp

    metadata = event.get("metadata") or {}
    comp = _component_from_mapping(metadata, known_components)
    if comp:
        return comp

    payload = event.get("payload")
    if isinstance(payload, dict):
        comp = _component_from_mapping(payload, known_components)
        if comp:
            return comp

    tags = event.get("tags") or []
    for tag in tags:
        comp = _component_from_string(tag, known_components)
        if comp:
            return comp
    return None


def _component_from_mapping(
    data: Dict[str, Any],
    known_components: Set[str],
) -> Optional[str]:
    for key in ("component", "module", "target", "file", "path"):
        value = data.get(key)
        comp = _component_from_string(value, known_components)
        if comp:
            return comp
    return None


def _component_from_string(value: Any, known_components: Set[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace('\\', '/')
    if text.startswith('repo://'):
        text = text[len('repo://') :]
    if text.startswith('./'):
        text = text[2:]
    if text.startswith('../'):
        text = text.split('../')[-1]
    if text.startswith('/'):
        text = text.lstrip('/')
    token = re.split(r"[./]", text, maxsplit=1)[0]
    return token if token in known_components else None


def _component_for_parts(parts: Sequence[str], known_components: Set[str]) -> Optional[str]:
    if not parts:
        return None
    first = parts[0]
    if first in SKIP_COMPONENT_PARTS or first.startswith('.'):
        return None
    return first if first in known_components else None


def _normalise_git_path(path: str) -> str:
    cleaned = path.strip().replace('{', '').replace('}', '')
    if '=>' in cleaned:
        cleaned = cleaned.split('=>', 1)[-1]
    return cleaned.strip()


def _build_stats(
    nodes: List[KnowledgeNode],
    edges: List[KnowledgeEdge],
    metrics_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    component_count = len(nodes)
    edge_count = len(edges)
    top_risky = sorted(
        (
            {
                "component": node.label,
                "risk_score": metrics_lookup[node.label]["risk_score"],
                "recent_fix_commits": metrics_lookup[node.label]["recent_fix_commits"],
                "test_failures": metrics_lookup[node.label]["test_failures"],
                "apply_failures": metrics_lookup[node.label]["apply_failures"],
            }
            for node in nodes
        ),
        key=lambda item: item["risk_score"],
        reverse=True,
    )[:5]

    dependency_counts = sorted(
        (
            {
                "component": node.label,
                "dependency_count": len(node.attributes.get("dependencies", [])),
            }
            for node in nodes
        ),
        key=lambda item: item["dependency_count"],
        reverse=True,
    )[:5]

    return {
        "component_count": component_count,
        "edge_count": edge_count,
        "top_risky": top_risky,
        "top_dependencies": dependency_counts,
    }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_failure_event(event: Dict[str, Any]) -> bool:
    for container in (event.get("metadata") or {}, event.get("payload") or {}):
        if not isinstance(container, dict):
            continue
        status = container.get("status")
        if isinstance(status, str) and status.lower() in {"fail", "failed", "failure", "error", "timeout"}:
            return True
        if container.get("success") is False or container.get("ok") is False:
            return True
    tags = event.get("tags") or []
    return any(str(tag).lower() in {"fail", "error", "timeout"} for tag in tags)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""


def _empty_component_entry() -> Dict[str, Any]:
    return {
        "files": [],
        "loc": 0,
        "dependencies": set(),
        "external_dependencies": set(),
        "dependency_sources": defaultdict(set),
    }


__all__ = [
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "build_knowledge_graph",
    "write_knowledge_graph",
    "summarize_knowledge_graph",
    "load_knowledge_graph",
    "knowledge_hotspots",
]
