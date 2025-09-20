

"""Healthcheck CLI command (extracted from selfcoder.cli).

Provides:
- run(args): executes the health checks and prints a summary
- register(subparsers): wires the `healthcheck` subcommand
"""
from __future__ import annotations

import logging
import sys

from selfcoder import healthcheck

# Ensure safe_subprocess is importable for future CLI checks
from ops.security.safe_subprocess import safe_run
def _check_safe_subprocess_cli() -> tuple[bool, str]:
    try:
        result = safe_run([sys.executable, "-c", "print('hc_cli')"])
        return (b"hc_cli" in result.stdout, "safe_subprocess (cli) basic run")
    except Exception as e:
        return False, f"safe_subprocess (cli) failed: {e}"


def _run_checks(verbose: bool) -> tuple[bool, str]:
    """Run all health checks and build a printable report tuple (ok, message)."""
    try:
        import importlib
        from selfcoder import logging_config as _lc
        _orig_setup = _lc.setup_logging

        def _setup_logging_shim(*, level: int = logging.INFO, stream = sys.stderr):
            try:
                return _orig_setup(level=level, stream=stream)
            except TypeError:
                try:
                    return _orig_setup(level, stream)
                except TypeError:
                    return _orig_setup()

        _lc.setup_logging = _setup_logging_shim
        importlib.reload(healthcheck)
        try:
            res = healthcheck.run_all()
        finally:
            _lc.setup_logging = _orig_setup
    except Exception:
        res = healthcheck.run_all()

    if isinstance(res, tuple) and len(res) >= 2:
        ok, report = bool(res[0]), str(res[1])
    else:
        ok, report = bool(res), "healthcheck completed"

    if verbose:
        # Build a per-check breakdown
        details = []
        for name in sorted(n for n in dir(healthcheck) if n.startswith("_check_")):
            fn = getattr(healthcheck, name, None)
            if callable(fn):
                try:
                    r = fn()
                    details.append((bool(r[0]), str(r[1])) if isinstance(r, tuple) else (bool(r), f"{name}"))
                except Exception as e:
                    details.append((False, f"{name} raised: {e!r}"))
        # Also run CLI-side safe_subprocess check
        try:
            r = _check_safe_subprocess_cli()
            details.append((bool(r[0]), str(r[1])) if isinstance(r, tuple) else (bool(r), "_check_safe_subprocess_cli"))
        except Exception as e:
            details.append((False, f"_check_safe_subprocess_cli raised: {e!r}"))
        lines = []
        for ok_i, msg_i in details:
            prefix = " - [OK] " if ok_i else " - [FAIL] "
            lines.append(prefix + msg_i)
        if lines:
            report = report + "\n" + "\n".join(lines)

    return ok, report


def run(args) -> int:
    ok, report = _run_checks(verbose=getattr(args, "verbose", False))
    status = "OK" if ok else "FAIL"
    print(f"[healthcheck] {status}: {report}")
    return 0 if ok else 1


def register(subparsers) -> None:
    """Register the `healthcheck` subcommand on the given subparsers object."""
    sc = subparsers.add_parser("healthcheck", help="run health checks")
    sc.add_argument("--verbose", action="store_true", help="print per-check breakdown")
    sc.set_defaults(func=run)