from __future__ import annotations
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

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
    """Runs pytest and healthchecks in the shadow directory and returns the results.
    Allows skipping and timeout control."""
    py_executable = sys.executable

    # Pytest result
    if skip_pytest:
        tests = {
            "rc": 0,
            "stdout": "[simulate] pytest skipped by --skip-pytest\n",
            "stderr": "",
        }
    else:
        tests_dir = shadow_root / "selfcoder" / "tests"
        has_tests = tests_dir.exists() and any(tests_dir.rglob("test_*.py"))
        if not has_tests:
            tests = {
                "rc": 0,
                "stdout": "[simulate] no tests found, skipping\n",
                "stderr": "",
            }
        else:
            test_cmd = [py_executable, "-m", "pytest", "-q", "selfcoder/tests"]
            try:
                p = subprocess.run(
                    test_cmd,
                    cwd=shadow_root,
                    capture_output=True,
                    text=True,
                    timeout=pytest_timeout,
                    check=False
                )
                tests = {
                    "rc": p.returncode,
                    "stdout": p.stdout,
                    "stderr": p.stderr,
                }
            except subprocess.TimeoutExpired as e:
                analysis = analyze_exception(e)
                persist_analysis(analysis)
                tests = {
                    "rc": 124,
                    "stdout": "[simulate] pytest timed out after {} seconds\n".format(pytest_timeout),
                    "stderr": "",
                    "analysis": analysis,
                }
            except PermissionError as e:
                # Some environments deny kill() on timeout; treat as TimeoutExpired equivalent
                analysis = analyze_exception(e)
                if isinstance(analysis, dict):
                    analysis["root_cause"] = "TimeoutExpired"
                persist_analysis(analysis)
                tests = {
                    "rc": 124,
                    "stdout": "[simulate] pytest timeout/kill not permitted\n",
                    "stderr": "",
                    "analysis": analysis,
                }

    # Healthcheck result
    if skip_healthcheck:
        hc = {
            "rc": 0,
            "stdout": "[simulate] healthcheck skipped by --skip-healthcheck\n",
            "stderr": "",
        }
    else:
        hc_cmd = [py_executable, "-m", "selfcoder.cli", "healthcheck"]
        try:
            p = subprocess.run(
                hc_cmd,
                cwd=shadow_root,
                capture_output=True,
                text=True,
                timeout=healthcheck_timeout,
                check=False
            )
            hc = {
                "rc": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        except subprocess.TimeoutExpired as e:
            analysis = analyze_exception(e)
            persist_analysis(analysis)
            hc = {
                "rc": 124,
                "stdout": "[simulate] healthcheck timed out after {} seconds\n".format(healthcheck_timeout),
                "stderr": "",
                "analysis": analysis,
            }
        except PermissionError as e:
            analysis = analyze_exception(e)
            if isinstance(analysis, dict):
                analysis["root_cause"] = "TimeoutExpired"
            persist_analysis(analysis)
            hc = {
                "rc": 124,
                "stdout": "[simulate] healthcheck timeout/kill not permitted\n",
                "stderr": "",
                "analysis": analysis,
            }

    return {
        "pytest": tests,
        "healthcheck": hc,
    }

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
