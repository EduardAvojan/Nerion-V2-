

from __future__ import annotations
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Optional dependency: watchdog
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
    _HAS_WATCHDOG = True
except Exception:
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore
    _HAS_WATCHDOG = False


# Default on_change callback uses our loader + registries if caller doesn't provide one
def _default_on_change(plugins_dir: str) -> None:
    try:
        from plugins.registry import transformer_registry as _xf_reg, cli_registry as _cli_reg
        from plugins.loader import load_plugins as _load_plugins
        _load_plugins(_xf_reg, _cli_reg, plugins_dir=plugins_dir)
    except Exception:
        # Never crash the watcher due to plugin issues
        pass


@dataclass
class WatchHandle:
    observer: Optional["Observer"]
    path: Path
    stop_event: threading.Event


class _DebouncedHandler(FileSystemEventHandler):  # type: ignore[misc]
    def __init__(self, plugins_dir: str, on_change: Callable[[str], None], debounce_sec: float = 0.3):
        super().__init__()
        self.plugins_dir = plugins_dir
        self.on_change = on_change
        self.debounce_sec = debounce_sec
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _schedule(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_sec, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self):
        try:
            self.on_change(self.plugins_dir)
        except Exception:
            pass

    # Any FS event should trigger a debounced reload
    def on_any_event(self, event):  # type: ignore[no-redef]
        self._schedule()


def start_watcher(plugins_dir: str = "plugins",
                  on_change: Optional[Callable[[str], None]] = None) -> Optional[WatchHandle]:
    """
    Start a hot-reload file watcher on the plugins directory.

    Args:
        plugins_dir: Directory to watch (default: "plugins").
        on_change: Callback invoked after a short debounce when any file in
                   the directory changes. Signature: (plugins_dir: str) -> None.
                   If omitted, a safe default that reloads plugins is used.

    Returns:
        WatchHandle if a watcher is running, or None if watchdog is unavailable
        or the directory does not exist.
    """
    path = Path(plugins_dir).resolve()
    if not path.exists() or not path.is_dir():
        # Nothing to watch
        return None

    if not _HAS_WATCHDOG:
        # Graceful no-op if watchdog isn't installed
        print("[plugins] watchdog not installed; hot-reload disabled (pip install watchdog)")
        return None

    cb = on_change or _default_on_change
    stop_event = threading.Event()
    handler = _DebouncedHandler(str(path), cb)
    observer = Observer()
    observer.daemon = True
    observer.schedule(handler, str(path), recursive=True)
    observer.start()

    print(f"[plugins] watching {path} for changes…")
    return WatchHandle(observer=observer, path=path, stop_event=stop_event)


def stop_watcher(handle: Optional[WatchHandle]) -> None:
    """Stop a previously started watcher. Safe to call with None."""
    if handle is None:
        return
    try:
        if handle.observer is not None:
            handle.observer.stop()
            handle.observer.join(timeout=2.0)
    except Exception:
        pass