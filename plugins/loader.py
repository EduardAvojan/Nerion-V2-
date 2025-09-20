import os
import importlib.util
import logging
from typing import Optional
from pathlib import Path
from plugins.security import assert_plugin_allowed, plugin_sandbox

logger = logging.getLogger(__name__)

try:
    from core.memory.journal import log_event as _log_event  # type: ignore
except Exception:  # pragma: no cover
    def _log_event(*_a, **_kw):
        return None

def discover_plugins(plugins_dir="plugins"):
    plugins = []
    if not os.path.isdir(plugins_dir):
        return plugins
    for entry in os.listdir(plugins_dir):
        plugin_path = os.path.join(plugins_dir, entry)
        if os.path.isdir(plugin_path):
            plugin_file = os.path.join(plugin_path, "plugin.py")
            if os.path.isfile(plugin_file):
                plugins.append((entry, plugin_file))
    return plugins

def load_plugins(transformer_registry, cli_registry, plugins_dir="plugins"):
    loaded = []
    errors = []
    discovered = []
    plugins = discover_plugins(plugins_dir)
    discovered = [name for name, _ in plugins]
    for plugin_name, plugin_path in plugins:
        # Security: ensure plugin is explicitly allowlisted
        try:
            assert_plugin_allowed(Path("."), plugin_name, Path(plugin_path))
        except Exception as e:
            logger.error(f"Plugin {plugin_name} not allowed: {e}")
            errors.append(plugin_name)
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"{plugin_name}.plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugin_class = getattr(module, "Plugin", None)
            if plugin_class is None:
                logger.error(f"No Plugin class found in {plugin_path}")
                continue
            plugin = plugin_class()
            if hasattr(plugin, "register_transformers"):
                plugin.register_transformers(transformer_registry)
            if hasattr(plugin, "register_cli"):
                # Wrap CLI extension callbacks so that any registered command's
                # handler runs inside a runtime FS sandbox.
                def _add_cli_ext(cb):
                    def _proxied(subparsers):
                        # Proxy add_parser to wrap set_defaults(func=...)
                        orig_add = getattr(subparsers, 'add_parser')
                        def _add_parser_proxy(*a, **k):
                            p = orig_add(*a, **k)
                            orig_setdef = getattr(p, 'set_defaults')
                            def _set_defaults_proxy(**kwargs):
                                fn = kwargs.get('func')
                                if callable(fn):
                                    def _wrapped(args):
                                        from pathlib import Path as _P
                                        with plugin_sandbox(_P('.').resolve()):
                                            return fn(args)
                                    kwargs['func'] = _wrapped
                                return orig_setdef(**kwargs)
                            setattr(p, 'set_defaults', _set_defaults_proxy)
                            return p
                        setattr(subparsers, 'add_parser', _add_parser_proxy)
                        # Invoke original callback with the proxied subparsers
                        return cb(subparsers)
                    cli_registry.add_cli_extension(_proxied)
                plugin.register_cli(_add_cli_ext)
            loaded.append(plugin_name)
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name} from {plugin_path}: {e}")
            errors.append(plugin_name)
            try:
                _log_event("plugin_load", rationale="error", plugin=plugin_name, error=str(e), plugins_dir=plugins_dir)
            except Exception:
                pass
    try:
        _log_event(
            "plugin_load",
            rationale="load_plugins",
            plugins_dir=plugins_dir,
            discovered=discovered,
            loaded=loaded,
            errors=len(errors),
        )
    except Exception:
        pass

def reload_plugins(transformer_registry, cli_registry, plugins_dir="plugins"):
    transformer_registry.clear()
    cli_registry.clear()
    load_plugins(transformer_registry, cli_registry, plugins_dir)
    try:
        _log_event("plugin_reload", rationale="reload_plugins", plugins_dir=plugins_dir)
    except Exception:
        pass


# --- Convenience APIs for programmatic use ---------------------------------

def load_plugins_auto(plugins_dir: Optional[str] = None) -> None:
    """Load plugins using the singleton registries. Safe no-op on failure."""
    try:
        from plugins.registry import transformer_registry as _xf_reg, cli_registry as _cli_reg
        load_plugins(_xf_reg, _cli_reg, plugins_dir=(plugins_dir or os.getenv("NERION_PLUGINS_DIR", "plugins")))
    except Exception:
        # Never raise from loader in programmatic contexts
        pass


def reload_plugins_auto(plugins_dir: Optional[str] = None) -> None:
    """Reload plugins using the singleton registries. Safe no-op on failure."""
    try:
        from plugins.registry import transformer_registry as _xf_reg, cli_registry as _cli_reg
        reload_plugins(_xf_reg, _cli_reg, plugins_dir=(plugins_dir or os.getenv("NERION_PLUGINS_DIR", "plugins")))
    except Exception:
        # Never raise from loader in programmatic contexts
        pass
