from __future__ import annotations
import os
import shlex
import tempfile
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

from selfcoder.diagnostics import analyze_exception, persist_analysis

# Directories/files we intentionally ignore both when copying and when diffing
DEFAULT_IGNORES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "node_modules",
    "dist",
    "build",
    "backups",
    ".DS_Store",
}

# Cap diff text to avoid dumping huge outputs when many files differ
MAX_DIFF_CHARS = 20000

# Optional simulation checks (beyond pytest/healthcheck)
_OPTIONAL_CHECK_SPECS: Tuple[Tuple[str, str], ...] = (
    ("lint", "NERION_SIM_LINT_CMD"),
    ("typecheck", "NERION_SIM_TYPE_CMD"),
    ("ui_build", "NERION_SIM_UI_CMD"),
    ("regression", "NERION_SIM_REG_CMD"),
)

_OPTIONAL_TIMEOUT_DEFAULTS: Dict[str, int] = {
    "lint": 180,
    "typecheck": 240,
    "ui_build": 420,
    "regression": 300,
}

_SKIP_TOKENS = {"0", "false", "off", "skip", "none", "no"}


def _now() -> float:
    return time.monotonic()


def _skip_result(name: str, reason: str, *, stdout: str = "") -> Dict[str, Union[None, str, bool, float]]:
    return {
        "name": name,
        "command": None,
        "rc": None,
        "stdout": stdout,
        "stderr": "",
        "skipped": True,
        "reason": reason,
        "duration": 0.0,
        "ok": True,
        "origin": None,
    }


def _default_command_for(name: str, root: Path) -> Optional[List[str]]:
    # Defaults are intentionally off; enable via NERION_SIM_*_CMD.
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
        return None, "no default command configured", None
    if shutil.which(default_cmd[0]) is None:
        return None, f"command '{default_cmd[0]}' not available", None
    return default_cmd, None, "default"


def _resolve_timeout(name: str, env_var: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(env_var)
    if raw:
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _run_named_check(
    name: str,
    command: List[str],
    *,
    cwd: Path,
    timeout: Optional[int] = None,
) -> Dict[str, Union[int, str, bool, float, List[str]]]:
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
            "skipped": False,
            "reason": None,
            "duration": duration,
            "ok": proc.returncode == 0,
            "origin": None,
        }
    except subprocess.TimeoutExpired as exc:
        analysis = analyze_exception(exc)
        persist_analysis(analysis)
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": 124,
            "stdout": f"[simulate] {name} timed out after {timeout} seconds\n",
            "stderr": exc.stderr or "",
            "analysis": analysis,
            "skipped": False,
            "reason": "timeout",
            "duration": duration,
            "ok": False,
            "origin": None,
        }
    except PermissionError as exc:
        analysis = analyze_exception(exc)
        if isinstance(analysis, dict):
            analysis["root_cause"] = "TimeoutExpired"
        persist_analysis(analysis)
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": 124,
            "stdout": f"[simulate] {name} timeout/kill not permitted\n",
            "stderr": "",
            "analysis": analysis,
            "skipped": False,
            "reason": "timeout",
            "duration": duration,
            "ok": False,
            "origin": None,
        }
    except FileNotFoundError as exc:
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": None,
            "stdout": "",
            "stderr": str(exc),
            "skipped": True,
            "reason": f"command not found: {command[0]}",
            "duration": duration,
            "ok": True,
            "origin": None,
        }
    except Exception as exc:
        duration = round(_now() - start, 3)
        return {
            "name": name,
            "command": command,
            "rc": 1,
            "stdout": "",
            "stderr": str(exc),
            "skipped": False,
            "reason": "exception",
            "duration": duration,
            "ok": False,
            "origin": None,
        }

def _rewrite_paths_for_shadow(argv: list[Union[str, list]], real_root: Path, shadow_root: Path) -> list[str]:
    """
    Replace any path-like args that live under real_root with their corresponding
    path under shadow_root. Now correctly handles nested lists of paths.
    """
    out: list[str] = []
    
    def translate_path(p_str: str) -> str:
        try:
            p = Path(p_str)
            if not p.is_absolute():
                p = (real_root / p_str).resolve()
            
            if p.exists() and (p == real_root or real_root in p.parents):
                rel = p.relative_to(real_root)
                return str((shadow_root / rel).resolve())
        except (ValueError, OSError):
            pass
        return p_str

    i = 0
    while i < len(argv):
        tok = argv[i]

        # Handle known flags that take a path argument
        if isinstance(tok, str) and tok in {"--root", "-f", "--file", "--actions-file"} and i + 1 < len(argv):
            val = argv[i + 1]
            if isinstance(val, str):
                out.extend([tok, translate_path(val)])
            else:
                 # Should not happen for these flags, but handle gracefully
                out.extend([tok, str(val)])
            i += 2
            continue

        # --- CORRECTED LOGIC BLOCK ---
        # Handle the case where the argument is a list of paths
        if isinstance(tok, list):
            translated_list = [translate_path(p) for p in tok if isinstance(p, str)]
            out.extend(translated_list)
            i += 1
            continue
        # --- END CORRECTION ---

        # Handle standalone path-like strings
        if isinstance(tok, str) and not tok.startswith('-'):
            out.append(translate_path(tok))
        else:
            # It's a flag or some other argument, pass it through
            out.append(str(tok))
        
        i += 1
    return out

def make_shadow_copy(src_root: Path, dest: Path = None) -> Path:
    """Creates a clean, temporary copy of the project for simulation."""
    shadow = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="nerion-sim-"))

    def _ignore(_dir, names):
        return DEFAULT_IGNORES & set(names)

    shutil.copytree(src_root, shadow, dirs_exist_ok=True, ignore=_ignore)
    return shadow

def run_tests_and_healthcheck(
    shadow_root: Path,
    skip_pytest: bool = False,
    skip_healthcheck: bool = False,
    pytest_timeout: Optional[int] = None,
    healthcheck_timeout: Optional[int] = None,
) -> dict:
    """Run required + optional simulation checks inside the shadow workspace."""

    results: Dict[str, Dict[str, Union[str, int, bool, float, List[str]]]] = {}
    py_executable = sys.executable

    # pytest --------------------------------------------------------------
    if skip_pytest:
        tests = _skip_result("pytest", "skipped via --skip-pytest", stdout="[simulate] pytest skipped by --skip-pytest\n")
    else:
        tests_dir = shadow_root / "selfcoder" / "tests"
        has_tests = tests_dir.exists() and any(tests_dir.rglob("test_*.py"))
        if not has_tests:
            tests = _skip_result("pytest", "no tests discovered", stdout="[simulate] no tests found, skipping\n")
        else:
            test_cmd = [py_executable, "-m", "pytest", "-q", "selfcoder/tests"]
            tests = _run_named_check("pytest", test_cmd, cwd=shadow_root, timeout=pytest_timeout)
            tests.setdefault("origin", "default")
    tests["timeout"] = pytest_timeout
    results["pytest"] = tests

    # healthcheck ---------------------------------------------------------
    if skip_healthcheck:
        hc = _skip_result("healthcheck", "skipped via --skip-healthcheck", stdout="[simulate] healthcheck skipped by --skip-healthcheck\n")
    else:
        hc_cmd = [py_executable, "-m", "selfcoder.cli", "healthcheck"]
        hc = _run_named_check("healthcheck", hc_cmd, cwd=shadow_root, timeout=healthcheck_timeout)
        hc.setdefault("origin", "default")
    hc["timeout"] = healthcheck_timeout
    results["healthcheck"] = hc

    # optional checks -----------------------------------------------------
    for name, env_var in _OPTIONAL_CHECK_SPECS:
        cmd, skip_reason, origin = _resolve_command(name, env_var, shadow_root)
        timeout_env = f"{env_var}_TIMEOUT"
        timeout_default = _OPTIONAL_TIMEOUT_DEFAULTS.get(name)
        timeout = _resolve_timeout(name, timeout_env, timeout_default)
        if not cmd:
            entry = _skip_result(name, skip_reason or "disabled")
            entry["origin"] = origin
            entry["timeout"] = timeout
            results[name] = entry
            continue
        entry = _run_named_check(name, cmd, cwd=shadow_root, timeout=timeout)
        entry["origin"] = origin or "default"
        entry["timeout"] = timeout
        results[name] = entry

    return results

def compute_diff(old_root: Path, new_root: Path) -> dict:
    """Computes the difference between two directories.
    Returns dict: { "text": diff_text, "files": changed_files }
    """
    def _decode(out_bytes: bytes) -> str:
        try:
            return out_bytes.decode("utf-8", errors="replace")
        except Exception:
            return out_bytes.decode("latin-1", errors="replace")

    diff_text = ""
    changed_files: list[str] = []
    
    try:
        cmd = [
            "diff",
            "-ruN",
        ]
        # Add exclusions
        for pat in sorted(DEFAULT_IGNORES):
            cmd.extend(["--exclude", pat])
        
        cmd.extend([str(old_root), str(new_root)])

        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        diff_text = _decode(p.stdout or b"")
        
        # Now find the names of the changed files
        changed_files_set = set()
        for line in diff_text.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                path_part = line.split("\t")[0][4:]
                # Resolve the path relative to one of the roots to get a canonical name
                potential_path = Path(path_part).resolve()
                try:
                    rel_path = potential_path.relative_to(new_root)
                    changed_files_set.add(str(old_root / rel_path))
                except ValueError:
                    try:
                        rel_path = potential_path.relative_to(old_root)
                        changed_files_set.add(str(old_root / rel_path))
                    except ValueError:
                        pass
        changed_files = sorted(list(changed_files_set))

    except FileNotFoundError:
        # Fallback to git if diff command is not available
        try:
            p = subprocess.run(
                [
                    "git", "--no-pager", "diff", "--no-index",
                    *[f"--exclude={ignore}" for ignore in DEFAULT_IGNORES],
                    "--", str(old_root), str(new_root),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            diff_text = _decode(p.stdout or b"")
            changed_files = [line.split("/")[-1] for line in diff_text.splitlines() if line.startswith("+++ b/")]

        except FileNotFoundError:
            diff_text = "[diff tools not found, cannot compute diff]"
            changed_files = []

    if not diff_text.strip():
        diff_text = "[No changes detected]"

    # Truncate to keep console tidy and avoid extremely large dumps
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n… [truncated]"

    return {"text": diff_text, "files": changed_files}
