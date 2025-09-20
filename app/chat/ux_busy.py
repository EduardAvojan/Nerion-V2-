"""Lightweight terminal busy indicator (spinner).

- Prints a friendly message with a spinner only when work takes > start_delay_s.
- Auto-clears the line when stopped.
- No-op when stdout is not a TTY.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional


class BusyIndicator:
    def __init__(self, message: str = "Working…", *, interval: float = 0.12, start_delay_s: float = 0.35) -> None:
        self.message = message
        self.interval = float(interval)
        self.start_delay_s = float(start_delay_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._shown = False
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        # Braille spinner frames, fallback to ASCII if unsupported terminals
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        try:
            # crude capability check: encode then decode
            " ".join(self._frames).encode(sys.stdout.encoding or "utf-8", errors="strict")
        except Exception:
            self._frames = ["-", "\\", "|", "/"]

    def start(self) -> None:
        if not self._tty:
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="nerion_busy", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t0 = time.time()
        i = 0
        while not self._stop.is_set():
            if not self._shown and (time.time() - t0) < self.start_delay_s:
                time.sleep(min(self.interval, self.start_delay_s))
                continue
            self._shown = True
            frame = self._frames[i % len(self._frames)]
            i += 1
            try:
                sys.stdout.write(f"\r\x1b[2K{frame} {self.message}")
                sys.stdout.flush()
            except Exception:
                pass
            time.sleep(self.interval)
        # clear line on exit
        try:
            if self._shown:
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()
        except Exception:
            pass

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._thread.join(timeout=0.3)
        except Exception:
            pass
        self._thread = None
        self._shown = False

    # Context manager helpers
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

