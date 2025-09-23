from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from selfcoder.diagnostics import analyze_exception, persist_analysis


_OPTIONAL_CHECK_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("smoke", "NERION_VERIFY_SMOKE_CMD", "NERION_VERIFY_SMOKE_TIMEOUT"),
    ("integration", "NERION_VERIFY_INTEGRATION_CMD", "NERION_VERIFY_INTEGRATION_TIMEOUT"),
    ("ui_build", "NERION_VERIFY_UI_CMD", "NERION_VERIFY_UI_TIMEOUT"),
    ("regression", "NERION_VERIFY_REG_CMD", "NERION_VERIFY_REG_TIMEOUT"),
)

_OPTIONAL_TIMEOUT_DEFAULTS: Dict[str, int] = {
    "smoke": 240,
    "integration": 480,
    "ui_build": 420,
    "regression": 420,
}

_SKIP_TOKENS = {"0", "false", "off", "skip", "none", "no"}


def _now() -> float:
    return time.monotonic()


def _default_command_for(name: str, root: Path) -> Optional[List[str]]:
    if name == "smoke":
        smoke_dir = root / "tests" / "smoke"
        if smoke_dir.exists():
            return ["pytest", "tests/smoke", "-q"]
    if name == "integration":
        integ_dir = root / "tests" / "integration"
        if integ_dir.exists():
            return ["pytest", "tests/integration", "-q"]
    if name == "ui_build":
        pkg = root / "app" / "ui" / "holo-app" / "package.json"
        if pkg.exists():
            return ["npm", "run", "build", "--prefix", "app/ui/holo-app"]
    if name == "regression":
        reg_dir = root / "tests" / "regression"
        if reg_dir.exists():
            return ["pytest", "tests/regression", "-q"]
    return None


def _resolve_command(name: str, env_var: str, root: Path) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    raw = os.getenv(env_var)
    if raw:
        token = raw.strip()
        if token.lower() in _SKIP_TOKENS:
            return None, f"disabled via {env_var}", f"env:{env_var}"
        return shlex.split(token), None, f"env:{env_var}"
    default_cmd = _default_command_for(name, root)
    if not default_cmd:
        return None, "no command configured", None
    if shutil.which(default_cmd[0]) is None:
        return None, f"command '{default_cmd[0]}' not available", None
    return default_cmd, None, "default"


def _resolve_timeout(name: str, env_var: str) -> Optional[int]:
    raw = os.getenv(env_var)
    if raw:
        try:
            return int(raw)
        except ValueError:
            return _OPTIONAL_TIMEOUT_DEFAULTS.get(name)
    return _OPTIONAL_TIMEOUT_DEFAULTS.get(name)


def _run_named_check(
    name: str,
    command: List[str],
    *,
    cwd: Path,
    timeout: Optional[int] = None,
) -> Dict[str, object]:
    start = _now()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration": duration,
            "skipped": False,
            "reason": None,
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        analysis = analyze_exception(exc)
        persist_analysis(analysis)
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": 124,
            "stdout": f"[verify] {name} timed out after {timeout} seconds\n",
            "stderr": exc.stderr or "",
            "duration": duration,
            "skipped": False,
            "reason": "timeout",
            "ok": False,
            "analysis": analysis,
        }
    except FileNotFoundError as exc:
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": None,
            "stdout": "",
            "stderr": str(exc),
            "duration": duration,
            "skipped": True,
            "reason": f"command not found: {command[0]}",
            "ok": True,
        }
    except PermissionError as exc:
        analysis = analyze_exception(exc)
        persist_analysis(analysis)
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": 124,
            "stdout": f"[verify] {name} timeout/kill not permitted\n",
            "stderr": str(exc),
            "duration": duration,
            "skipped": False,
            "reason": "timeout",
            "ok": False,
            "analysis": analysis,
        }
    except Exception as exc:
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": 1,
            "stdout": "",
            "stderr": str(exc),
            "duration": duration,
            "skipped": False,
            "reason": "exception",
            "ok": False,
        }


def run_post_apply_checks(root: Path | None = None) -> Dict[str, Dict[str, object]]:
    """Run optional post-apply verification commands defined via env vars."""

    root = Path(root or Path.cwd())
    results: Dict[str, Dict[str, object]] = {}
    for name, env_var, timeout_var in _OPTIONAL_CHECK_SPECS:
        cmd, skip_reason, origin = _resolve_command(name, env_var, root)
        if cmd is None:
            results[name] = {
                "name": name,
                "command": None,
                "rc": None,
                "stdout": "",
                "stderr": "",
                "duration": 0.0,
                "skipped": True,
                "reason": skip_reason,
                "ok": True,
                "origin": origin,
            }
            continue
        timeout = _resolve_timeout(name, timeout_var)
        entry = _run_named_check(name, cmd, cwd=root, timeout=timeout)
        entry["origin"] = origin
        results[name] = entry
    return results


def failed_checks(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Return the list of check names that failed."""

    failed: List[str] = []
    for name, entry in results.items():
        if entry.get("skipped"):
            continue
        rc = entry.get("rc")
        if rc not in (0, None):
            failed.append(name)
    return failed
