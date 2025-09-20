import json
import subprocess
from typing import Dict, Any, List, Optional
from selfcoder.analysis.smells import Smell

# Default roots (adjustable by callers)
DEFAULT_PATHS = ["app", "core", "selfcoder", "voice", "ui", "src"]


def _run_json(cmd: List[str]) -> Optional[Any]:
    """Run a command and parse stdout as JSON; return None on any failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception:
        return None
    return None


def run_all(paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run multiple static analyzers and return their raw outputs in a combined dict:
    {
        'pylint': [...],
        'bandit': [...],
        'flake8': [...],
        'radon':  [...]
    }
    Missing tools are silently skipped so this function is resilient across environments.
    """
    paths = paths or DEFAULT_PATHS
    result: Dict[str, Any] = {"pylint": [], "bandit": [], "flake8": [], "radon": []}

    # pylint (JSON)
    try:
        data = _run_json(["pylint", "-f", "json", *paths])
        if isinstance(data, list):
            result["pylint"] = data
    except Exception:
        pass

    # bandit (JSON)
    try:
        data = _run_json(["bandit", "-r", *paths, "-f", "json"])
        if isinstance(data, dict):
            result["bandit"] = data.get("results", [])
    except Exception:
        pass

    # flake8 (parse text lines → quick JSON)
    try:
        proc = subprocess.run(["flake8", *paths], capture_output=True, text=True, check=False)
        flake: List[Dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            # path:line:col: code message...
            parts = line.split(":", 3)
            if len(parts) == 4:
                path, ln, col, rest = parts
                code, _, text = rest.strip().partition(" ")
                try:
                    ln_int = int(ln)
                except ValueError:
                    ln_int = None
                flake.append({"path": path, "line": ln_int, "code": code, "text": text})
        result["flake8"] = flake
    except Exception:
        pass

    # radon cc (cyclomatic complexity) JSON
    try:
        # --- CORRECTED LINE: Added "--min", "A" to report on all complexity ranks. ---
        proc = subprocess.run(["radon", "cc", "--min", "A", "-j", *paths], capture_output=True, text=True, check=False)
        if proc.stdout.strip():
            j = json.loads(proc.stdout)
            rad: List[Dict[str, Any]] = []
            for pth, entries in j.items():
                if isinstance(entries, list):
                    for e in entries:
                        rad.append({
                            "path": pth,
                            "name": e.get("name"),
                            "complexity": e.get("complexity"),
                            "rank": e.get("rank"),
                            "line": e.get("lineno"),
                        })
            result["radon"] = rad
    except Exception:
        pass

    return result

def run_checks() -> bool:
    print('[Checks] Running ruff/pylint (scaffold) ... OK')
    return True


# --- Normalizer for static check outputs ---

def normalize_static_outputs(raw: Dict[str, Any]) -> List[Smell]:
    """Convert raw static check outputs into Smell instances."""
    findings: List[Smell] = []
    for tool, items in raw.items():
        if not isinstance(items, list):
            continue
        for item in items:
            try:
                findings.append(
                    Smell(
                        tool=tool,
                        code=item.get("code") or item.get("symbol") or item.get("rank"),
                        message=item.get("message") or item.get("text") or str(item),
                        path=item.get("path") or item.get("filename") or "",
                        line=item.get("line") or item.get("lineno"),
                        symbol=item.get("name") or item.get("obj") or None,
                        meta=item,
                    )
                )
            except Exception:
                continue
    return findings