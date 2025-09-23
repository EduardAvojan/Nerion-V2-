# selfcoder/artifacts.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import io
import contextlib

# Prefer project security helpers; fall back to local-safe shims in isolated environments (e.g., tests)
try:
    from ops.security.io_safe import write_text  # type: ignore
    from ops.security import fs_guard  # type: ignore
except Exception:  # pragma: no cover - exercised in shadow test envs
    def write_text(path: Path, content: str) -> None:
        """Safely write UTF-8 text ensuring parent directories exist.
        Fallback used only when ops.security is not importable (e.g., test shadow trees).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    class _FSGuard:
        @staticmethod
        def ensure_in_repo(root: Path, path_str: str) -> Path:
            """Conservative path resolution restricted under root.
            If the resolved path escapes `root`, re-root under `root` using the basename.
            """
            root_res = Path(root).resolve()
            p = (root_res / path_str).resolve() if not str(path_str).startswith("/") else Path(path_str).resolve()
            try:
                p.relative_to(root_res)
            except Exception:
                p = root_res / Path(path_str).name
            return p

    fs_guard = _FSGuard()

ARTIFACT_DIR = Path(".nerion/artifacts")

@dataclass
class SimResult:
    pytest_rc: Optional[int] = None
    health_ok: Optional[bool] = None
    pytest_out: Optional[str] = None
    health_out: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlanArtifact:
    ts: str
    origin: str                  # "cli", "voice", "self-improve", ...
    score: int
    rationale: str               # short “why this score”
    plan: Dict[str, Any]         # the validated plan dict (not raw)
    files_touched: List[str] = field(default_factory=list)
    sim: Optional[SimResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

def artifact_path(prefix: str = "plan") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return ARTIFACT_DIR / f"{prefix}_{stamp}.json"

def save_artifact(artifact: PlanArtifact, out: Optional[Path] = None) -> Path:
    out = out or artifact_path()
    safe = fs_guard.ensure_in_repo(Path("."), str(out))
    safe.parent.mkdir(parents=True, exist_ok=True)
    # Suppress any stdout/stderr prints from underlying writer implementations
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        write_text(safe, json_dump(artifact))
    return safe

def json_dump(artifact: PlanArtifact) -> str:
    import json
    d = asdict(artifact)
    return json.dumps(d, indent=2, ensure_ascii=False)
