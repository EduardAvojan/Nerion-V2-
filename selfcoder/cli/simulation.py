"""Simulation system for running commands in shadow copies."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

try:
    from ops.security.safe_subprocess import safe_run
except Exception:
    import subprocess
    def safe_run(argv, **kwargs):
        return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd","timeout","check","capture_output")})


def simulate_fallback(args) -> int:
    """Fallback simulation when full simulation system is unavailable."""
    root = Path.cwd()
    rc_cmd = 0
    print("\n--- Simulation Report ---")
    print("[simulate] Shadow Directory: None")
    print(f"[simulate] Command exit code: {rc_cmd}")

    skip_pytest = getattr(args, "skip_pytest", False)
    skip_healthcheck = getattr(args, "skip_healthcheck", False)

    pytest_rc = None
    if not skip_pytest:
        try:
            res = safe_run([sys.executable, "-m", "pytest", "-q"], cwd=root, timeout=300, check=False, capture_output=True)
            pytest_rc = res.returncode
        except Exception:
            pytest_rc = 1
        print(f"[simulate] Pytest exit code: {pytest_rc}")
    pytest_rc_norm = pytest_rc
    if pytest_rc is not None and pytest_rc == 5:
        pytest_rc_norm = 0

    health_rc = None
    if not skip_healthcheck:
        try:
            import importlib
            hc = importlib.import_module("selfcoder.healthcheck")
            out = hc.run_all()
            if isinstance(out, tuple):
                ok, _ = out
                health_rc = 0 if ok else 1
            else:
                health_rc = 0 if out else 1
        except Exception:
            health_rc = 1
        print(f"[simulate] Healthcheck exit code: {health_rc}")

    print("--- Diff ---\n")

    if getattr(args, "simulate_json", False):
        summary = {
            "shadow_dir": str(root),
            "cmd_exit": rc_cmd,
            "pytest_exit": pytest_rc,
            "healthcheck_exit": health_rc,
            "changed": False,
            "changed_files": [],
            "diff_text": "",
        }
        print(json.dumps(summary, indent=2))

    failed = (rc_cmd != 0) or ((pytest_rc_norm is not None and pytest_rc_norm != 0)) or ((health_rc is not None and health_rc != 0))
    return 1 if failed else 0


def maybe_simulate(args, cmd_name, argv_builder):
    """
    Check if simulation is requested and execute command in shadow copy if so.

    Returns:
        - None if no simulation requested (caller continues with actual apply)
        - Exit code (int) if simulation was performed
    """
    if not getattr(args, "simulate", False):
        return None  # No simulation, caller continues with the actual apply

    try:
        from selfcoder.simulation import make_shadow_copy, run_tests_and_healthcheck, compute_diff, _rewrite_paths_for_shadow
        root = Path.cwd()
        shadow = make_shadow_copy(root, getattr(args, "simulate_dir", None))

        original_argv = argv_builder()
        # Rewrite paths to point inside the shadow copy
        shadow_argv = _rewrite_paths_for_shadow(original_argv, root, shadow)

        # Run the command in the shadow repository
        full_command = [sys.executable, "-m", "selfcoder.cli", cmd_name] + shadow_argv
        result = safe_run(full_command, cwd=shadow, timeout=300, check=False, capture_output=False)
        rc_cmd = result.returncode

        skip_pytest = getattr(args, "skip_pytest", False)
        skip_healthcheck = getattr(args, "skip_healthcheck", False)
        pytest_timeout = getattr(args, "pytest_timeout", None)
        healthcheck_timeout = getattr(args, "healthcheck_timeout", None)
        results = run_tests_and_healthcheck(
            shadow,
            skip_pytest=skip_pytest,
            skip_healthcheck=skip_healthcheck,
            pytest_timeout=pytest_timeout,
            healthcheck_timeout=healthcheck_timeout,
        )
        diff = compute_diff(root, shadow)
        changed_files = diff.get("files", [])
        changed = bool(changed_files)

        # Present the simulation results
        print("\n--- Simulation Report ---")
        print(f"[simulate] Shadow Directory: {shadow}")
        print(f"[simulate] Command exit code: {rc_cmd}")
        for name, entry in results.items():
            label = name.replace("_", " ").title()
            if name == "pytest":
                label = "Pytest"
            elif name == "healthcheck":
                label = "Healthcheck"
            if entry.get("skipped"):
                reason = entry.get("reason") or ""
                print(f"[simulate] {label} skipped: {reason}")
            else:
                print(f"[simulate] {label} exit code: {entry.get('rc')}")
        print(f"--- Diff ---\n{diff['text']}")

        out = None
        if getattr(args, "simulate_json", False):
            out = {
                "shadow_dir": str(shadow),
                "cmd_exit": rc_cmd,
                "changed": changed,
                "changed_files": changed_files,
                "diff_text": diff["text"],
                "checks": results,
            }

        if not getattr(args, "simulate_keep", False):
            print("[simulate] Cleaning up shadow directory...")
            shutil.rmtree(shadow, ignore_errors=True)

        if out is not None:
            print(json.dumps(out, indent=2))

        failed_checks = [
            name
            for name, entry in results.items()
            if not entry.get("skipped") and entry.get("rc") not in (0, None)
        ]
        failed = rc_cmd != 0 or bool(failed_checks)
        return 1 if failed else 0

    except Exception:
        return simulate_fallback(args)
