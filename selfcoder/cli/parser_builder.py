"""CLI parser builder with extension loading."""
from __future__ import annotations

import argparse

from .autotest import register_autotest

# --- Optional plugin system (safe if plugins/ is absent) -------------------
try:
    from plugins.registry import transformer_registry as _xf_reg, cli_registry as _cli_reg
    from plugins.loader import load_plugins as _load_plugins
except Exception:
    _xf_reg = None
    _cli_reg = None
    _load_plugins = None

# --- Optional plugin hot-reload (safe if watchdog/loader absent) ----------
try:
    from plugins.hot_reload import start_watcher as _plugins_start_watcher, stop_watcher as _plugins_stop_watcher
except Exception:
    _plugins_start_watcher = None
    _plugins_stop_watcher = None


def build_parser() -> argparse.ArgumentParser:
    """Build the main CLI argument parser with all extensions."""
    p = argparse.ArgumentParser(prog="nerion", description="Nerion Selfcoder CLI")
    p.add_argument("-V", "--version", action="store_true", help="print build version and exit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Register split-out built-ins
    try:
        from selfcoder.cli_ext import voice as _voice_cli
        _voice_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import chat as _chat_cli
        _chat_cli.register(sub)
    except Exception:
        pass
    # Prefer the snapshot-aware plan command from commands/ if available
    try:
        from selfcoder.cli.commands import plan as _plan_cmd
        _plan_cmd.register(sub)
    except Exception:
        # Keep going; fallback will be registered later if needed
        pass
    try:
        from selfcoder.cli_ext import journal as _journal_cli
        _journal_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import memory as _memory_cli
        _memory_cli.register(sub)
    except Exception:
        pass
    try:
        # Self-learn (LoRA dataset stub & schedule helpers)
        from selfcoder.cli_ext.self_learn import add_self_learn_subparser as _add_self_learn
        _add_self_learn(sub)
    except Exception:
        pass

    # Register autotest command
    register_autotest(sub)

    # Let modular CLI extenders register their subcommands (e.g., healthcheck)
    try:
        from selfcoder.cli_ext import parser as _cli_parser
        _cli_parser.extend_parser(sub)
    except Exception:
        # Do not fail the CLI if extender is missing or broken
        pass

    # Register plan and health subcommands
    try:
        from selfcoder.cli_ext import plan as _plan_cli
        _plan_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import health as _health_cli
        _health_cli.register(sub)
    except Exception:
        pass

    # Register healthcheck fallback if not already present
    choices = sub.choices if hasattr(sub, 'choices') else {}
    if "healthcheck" not in choices:
        from .fallbacks import register_health_fallback
        register_health_fallback(sub)

    # Register split-out subcommands
    try:
        from selfcoder.cli_ext import docstring as _doc_cli
        _doc_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import snapshots as _snap_cli
        _snap_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import batch as _batch_cli
        _batch_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import profile as _prof_cli
        _prof_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import bench as _bench_cli
        _bench_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import scan as _scan_cli
        _scan_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import rename as _ren_cli
        _ren_cli.register(sub)
    except Exception:
        pass

    # Register fallbacks if commands are missing
    try:
        choices = getattr(sub, "choices", {})
    except Exception:
        choices = {}

    if "rename" not in choices:
        from .fallbacks import register_rename_fallback
        register_rename_fallback(sub)

    try:
        from selfcoder.cli_ext import diagnose as _diag_cli
        _diag_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import self_improve as _si_cli
        _si_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import telemetry as _telemetry_cli
        _telemetry_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import architect as _architect_cli
        _architect_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import plugins as _pl_cli
        _pl_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import deps_cli as _deps_cli
        _deps_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import net as _net_cli
        _net_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import lint as _lint_cli
        _lint_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import doctor as _doctor_cli
        _doctor_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import graph as _graph_cli
        _graph_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import policy_cli as _pol_cli
        _pol_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import models as _models_cli
        _models_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import docs_cli as _docs_cli
        _docs_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import artifacts as _art_cli
        _art_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import review as _review_cli
        _review_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import patch as _patch_cli
        _patch_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import serv as _serve_cli
        _serve_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import learn as _learn_cli
        _learn_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import package as _package_cli
        _package_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import preflight as _pf_cli
        _pf_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import coverage_cli as _cov_cli
        _cov_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import main as _main_cli
        _main_cli.register(sub)
    except Exception:
        pass

    # Load plugin-based commands if plugin system is present
    try:
        if _load_plugins is not None:
            _load_plugins()  # type: ignore[misc]
        if _cli_reg is not None:
            try:
                _cli_reg.extend_parser(sub)
            except Exception:
                pass
        # Additionally, attempt to extend from the live registry module to
        # handle rare cases where multiple imports created distinct singletons.
        try:
            import plugins.registry as _preg  # type: ignore
            if hasattr(_preg, "cli_registry"):
                try:
                    _preg.cli_registry.extend_parser(sub)  # type: ignore[attr-defined]
                except Exception:
                    # Ignore duplicate parser names or any extension-time errors
                    pass
        except Exception:
            pass
    except Exception:
        # Do not fail the CLI if plugins misbehave
        pass
    return p
