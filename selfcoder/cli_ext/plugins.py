

from __future__ import annotations

import argparse
import os
import time
from core.ui.progress import progress
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes

# Optional plugin system (safe if plugins/ is absent)
try:
    from plugins.registry import transformer_registry as _xf_reg, cli_registry as _cli_reg
    from plugins.loader import load_plugins as _load_plugins, discover_plugins as _discover_plugins
    from plugins.security import load_allowlist_data as _load_allowlist, _sha256 as _sha256
except Exception:  # pragma: no cover - optional
    _xf_reg = None
    _cli_reg = None
    _load_plugins = None

# Optional plugin hot-reload (safe if watchdog/loader absent)
try:
    from plugins.hot_reload import start_watcher as _plugins_start_watcher, stop_watcher as _plugins_stop_watcher
except Exception:  # pragma: no cover - optional
    _plugins_start_watcher = None
    _plugins_stop_watcher = None


def _cmd_watch(args: argparse.Namespace) -> int:
    plugins_dir = getattr(args, "plugins_dir", None) or os.getenv("NERION_PLUGINS_DIR", "plugins")

    # Initial load so newly started watcher sees latest state
    try:
        if _load_plugins and _xf_reg and _cli_reg:
            _load_plugins(_xf_reg, _cli_reg, plugins_dir=plugins_dir)
    except Exception:
        pass

    if not _plugins_start_watcher:
        print("[plugins] hot-reload unavailable (install watchdog or check plugins.hot_reload)")
        return 0

    handle = _plugins_start_watcher(plugins_dir)
    if handle is None:
        # Either directory missing or watchdog not installed; message already printed by start_watcher
        return 0

    try:
        print("[plugins] press Ctrl+C to stop watching…")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[plugins] stopping watcher…")
    finally:
        try:
            if _plugins_stop_watcher:
                _plugins_stop_watcher(handle)
        except Exception:
            pass
    return 0


def register(subparsers) -> None:
    """Register the `plugins` command group and subcommands."""
    p_plugins = subparsers.add_parser("plugins", help="manage local plugins")
    p_plugins_sub = p_plugins.add_subparsers(dest="plugins_cmd", required=True)

    p_plugins_watch = p_plugins_sub.add_parser("watch", help="watch plugins dir and hot-reload on changes")
    p_plugins_watch.add_argument("--plugins-dir", default=None, help="plugins directory (default: $NERION_PLUGINS_DIR or ./plugins)")
    p_plugins_watch.set_defaults(func=_cmd_watch)

    def _cmd_verify(args: argparse.Namespace) -> int:
        plugins_dir = getattr(args, "plugins_dir", None) or os.getenv("NERION_PLUGINS_DIR", "plugins")
        from pathlib import Path
        root = Path('.')
        with progress("plugins: verify"):
            data = _load_allowlist(root)
        allowed = set(data.get('allowed', []) or [])
        hashes = dict(data.get('hashes', {}) or {})
        items = []
        for name, path in _discover_plugins(plugins_dir):
            p = Path(path)
            resolved = p.resolve() if p.exists() else p
            is_symlink = p.is_symlink()
            inside = str(resolved).startswith(str((root / 'plugins').resolve()))
            allowed_ok = name in allowed
            actual = ''
            try:
                actual = _sha256(resolved)
            except Exception:
                actual = ''
            expected = hashes.get(name)
            hash_ok = (expected is None) or (actual.lower() == str(expected).strip().lower())
            status = 'ok'
            reasons = []
            if not allowed_ok:
                status = 'blocked'
                reasons.append('not allowlisted')
            if is_symlink:
                status = 'blocked'
                reasons.append('symlink')
            if not inside:
                status = 'blocked'
                reasons.append('outside plugins dir')
            if expected and not hash_ok:
                status = 'blocked'
                reasons.append('hash mismatch')
            items.append({
                'name': name,
                'path': str(p),
                'allowed': allowed_ok,
                'symlink': is_symlink,
                'inside_plugins_dir': inside,
                'expected_hash': expected,
                'actual_hash': actual,
                'status': status,
                'reasons': reasons,
            })
        import json
        print(json.dumps({'plugins': items}, indent=2))
        ok = all((it.get('status') == 'ok') for it in items)
        print(_fmt_msg('plugins', 'verify', _MsgRes.OK if ok else _MsgRes.FAIL, f"count={len(items)}"))
        return 0

    p_plugins_verify = p_plugins_sub.add_parser("verify", help="verify plugins against allowlist and path policy")
    p_plugins_verify.add_argument("--plugins-dir", default=None, help="plugins directory (default: $NERION_PLUGINS_DIR or ./plugins)")
    p_plugins_verify.set_defaults(func=_cmd_verify)
