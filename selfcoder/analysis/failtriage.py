"""Failing-test triage utilities for code-repair/benchmark tasks.

Capabilities:
- Run pytest to identify failing tests and capture the first failure traceback.
- Rank suspect source files using heuristics:
  • Failure file hints from traceback frames (prefer non-test, in-repo files)
  • Keyword matches from error message against filenames/symbols
  • Optional coverage JSON if produced externally (light integration)
- Build a compact context pack (top-N files with focused windows).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re

try:
    from ops.security.safe_subprocess import safe_run
except Exception:
    import subprocess
    def safe_run(argv, **kwargs):
        return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd","timeout","check","capture_output","text")})


@dataclass
class TraceFrame:
    file: str
    line: int
    func: Optional[str]
    src: Optional[str]


@dataclass
class Failure:
    test: str
    message: str
    frames: List[TraceFrame]


@dataclass
class Triage:
    failed_tests: List[str]
    failures: List[Failure]
    suspects: List[Tuple[str, float]]
    notes: Dict[str, Any]


PY_FILE_RE = re.compile(r"(\S+\.py)(?::(\d+))?")


def _run_pytest_once(task_dir: Path, extra_args: Optional[List[str]] = None, timeout: int = 300) -> str:
    import sys
    args = [sys.executable, "-m", "pytest", "-ra"]
    if extra_args:
        args.extend(extra_args)
    res = safe_run(args, cwd=task_dir, timeout=timeout, check=False, capture_output=True)
    def _to_str(b):
        if b is None:
            return ""
        return b.decode('utf-8', errors='ignore') if isinstance(b, (bytes, bytearray)) else str(b)
    out = _to_str(res.stdout) + "\n" + _to_str(res.stderr)
    return out


def _extract_failed_tests(output: str) -> List[str]:
    nodeids: List[str] = []
    for ln in output.splitlines():
        # Patterns like: FAILED tests/test_x.py::test_foo - AssertionError: ...
        if ln.startswith("FAILED ") and " - " in ln:
            part = ln.split(" - ", 1)[0]
            node = part[len("FAILED "):].strip()
            nodeids.append(node)
    return nodeids


def _parse_first_traceback(output: str) -> Tuple[str, List[TraceFrame]]:
    # Naive parse: find last block of 'File "...", line N, in func' lines
    frames: List[TraceFrame] = []
    msg = ""
    lines = output.splitlines()
    acc: List[TraceFrame] = []
    for i, ln in enumerate(lines):
        m = re.search(r'File "([^"]+)", line (\d+), in (\S+)', ln)
        if m:
            acc.append(TraceFrame(file=m.group(1), line=int(m.group(2)), func=m.group(3), src=None))
        elif ln.strip().startswith("E ") and acc:
            # error message lines typically start with 'E '
            msg = ln.strip()[2:]
    if acc:
        frames = acc
    return msg, frames


def _score_suspects(frames: List[TraceFrame], error_msg: str, repo_root: Path, cov_hits: Optional[Dict[str, int]] = None) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    def add(path: str, w: float):
        try:
            p = Path(path)
            # de-prepend task dir if present
            rel = p.as_posix()
            scores[rel] = scores.get(rel, 0.0) + w
        except Exception:
            pass
    # Frame-based weighting
    for idx, fr in enumerate(frames):
        w = 2.0 if ("site-packages" not in fr.file and "/pytest/" not in fr.file) else 0.25
        add(fr.file, w)
    # Error keyword match against filenames
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", error_msg)
    toks = [t.lower() for t in toks if len(t) >= 3]
    if toks:
        for py in repo_root.rglob("*.py"):
            name = py.name.lower()
            hit = any(t in name for t in toks)
            if hit:
                add(py.as_posix(), 0.5)
    # Coverage-assisted weighting (failing tests only)
    if cov_hits:
        for fpath, hits in cov_hits.items():
            try:
                p = Path(fpath)
                if not str(p).endswith('.py'):
                    continue
                # Weight by log-scale of executed lines
                boost = min(10.0, 1.0 + (hits / 20.0))
                add(p.as_posix(), boost)
            except Exception:
                continue
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:20]


def _extract_windows(path: Path, points: List[int], radius: int = 30, max_bytes: int = 20_000) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""
    lines = text.splitlines()
    chunks: List[str] = []
    for pt in (points or [1]):
        start = max(1, pt - radius)
        end = min(len(lines), pt + radius)
        seg = lines[start-1:end]
        chunks.append(f"# Window {start}-{end}\n" + "\n".join(seg))
    out = "\n\n".join(chunks)
    if len(out.encode("utf-8")) > max_bytes:
        return out[:max_bytes]
    return out


def build_context_pack(repo_root: Path, suspects: List[Tuple[str, float]], frames: List[TraceFrame], top_n: int = 6) -> Dict[str, Any]:
    pack: Dict[str, Any] = {"files": []}
    pts_by_file: Dict[str, List[int]] = {}
    for fr in frames:
        pts_by_file.setdefault(fr.file, []).append(fr.line)
    chosen = suspects[:top_n]
    if not chosen:
        # Fallback to files from traceback frames
        seen = set()
        for fr in frames:
            try:
                if 'site-packages' in fr.file or '/pytest/' in fr.file:
                    continue
                if fr.file in seen:
                    continue
                seen.add(fr.file)
                chosen.append((fr.file, 0.1))
                if len(chosen) >= top_n:
                    break
            except Exception:
                continue
    for rel, score in chosen:
        try:
            p = Path(rel)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            win = _extract_windows(p, pts_by_file.get(rel, []))
            pack["files"].append({"path": str(p), "score": score, "window": win})
        except Exception:
            continue
    return pack


def _coverage_hits_inprocess(task_dir: Path, nodeids: List[str]) -> Dict[str, int]:
    """Best-effort: run failing tests with coverage and return hit counts per file."""
    covmap: Dict[str, int] = {}
    try:
        import coverage  # type: ignore
        import pytest  # type: ignore
        import sys as _sys
        cov = coverage.Coverage(source=[str(task_dir)])
        cov.start()
        cwd = Path.cwd()
        try:
            if str(task_dir) not in _sys.path:
                _sys.path.insert(0, str(task_dir))
            import os as __os
            __os.chdir(task_dir)
            pytest.main(['-q', '-x', *nodeids])
        finally:
            import os as __os
            __os.chdir(cwd)
        cov.stop()
        cov.save()
        data = cov.get_data()
        for f in data.measured_files():
            try:
                if not str(f).startswith(str(task_dir)):
                    continue
                covmap[str(Path(f))] = len(list(data.lines(f) or []))
            except Exception:
                continue
    except Exception:
        return {}
    return covmap


def triage_task(task_dir: Path) -> Triage:
    out = _run_pytest_once(task_dir)
    failed = _extract_failed_tests(out)
    msg, frames = _parse_first_traceback(out)
    cov_hits = {}
    if failed:
        cov_hits = _coverage_hits_inprocess(task_dir, failed)
    suspects = _score_suspects(frames, msg, task_dir, cov_hits=cov_hits)
    if not suspects:
        # Fallback suspects: non-test python files in task dir
        try:
            for py in sorted(task_dir.rglob('*.py')):
                if '/tests/' in py.as_posix() or py.name.startswith('test_'):
                    continue
                suspects.append((py.as_posix(), 0.1))
        except Exception:
            pass
    notes = {"raw_lines": len(out.splitlines()), "coverage_files": len(cov_hits)}
    return Triage(failed_tests=failed, failures=[Failure(test=failed[0] if failed else "", message=msg, frames=frames)], suspects=suspects, notes=notes)


def write_triage_artifacts(task_dir: Path, triage: Triage, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    triage_path = out_dir / "triage.json"
    suspects_path = out_dir / "suspects.json"
    ctx_path = out_dir / "context.json"
    triage_path.write_text(json.dumps({
        "failed_tests": triage.failed_tests,
        "failures": [
            {"test": f.test, "message": f.message, "frames": [asdict(fr) for fr in f.frames]}
            for f in triage.failures
        ]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    suspects_path.write_text(json.dumps(triage.suspects, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx = build_context_pack(task_dir, triage.suspects, triage.failures[0].frames if triage.failures else [])
    # Include failure summary for proposer heuristics
    ctx["failures"] = [
        {"test": f.test, "message": f.message}
        for f in triage.failures
    ]
    ctx_path.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir
