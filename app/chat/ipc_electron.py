"""StdIO IPC bridge for the Electron HOLO app.

When NERION_UI_CHANNEL=holo-electron, this module:
- emits JSON events to stdout (one per line)
- reads JSON commands from stdin and exposes a queue for the chat loop

It lets Electron spawn Python and communicate without launching Electron from Python.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from queue import SimpleQueue
from typing import Any, Callable, Optional

_ENABLED = (os.getenv("NERION_UI_CHANNEL") or "").strip().lower() == "holo-electron"
_chat_queue: "SimpleQueue[str]" = SimpleQueue()
_command_queue: "SimpleQueue[dict[str, Any]]" = SimpleQueue()
_handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
_handler_lock = threading.RLock()
_reader_started = False


def enabled() -> bool:
    return _ENABLED


def emit(event_type: str, payload: Optional[dict[str, Any]] = None) -> None:
    if not _ENABLED:
        return
    try:
        msg = {"type": event_type}
        if payload is not None:
            msg["payload"] = payload
        # Debug logging for chat_turn events
        try:
            if event_type == "chat_turn":
                with open("/tmp/nerion_ipc_debug.log", "a") as f:
                    import datetime
                    role = payload.get("role") if payload else None
                    text_preview = (payload.get("text") if payload else "")[:50]
                    f.write(f"[{datetime.datetime.now()}] Emitting chat_turn: role={role}, text={text_preview}...\n")
        except Exception:
            pass
        # Use stdout line protocol consumed by Electron main process
        stdout = getattr(sys, '__stdout__', sys.stdout)
        stdout.write(json.dumps(msg) + "\n")
        stdout.flush()
    except Exception:
        # Never crash the runtime due to UI wiring issues
        pass


def get_nowait() -> Optional[str]:
    """Return next chat text from UI queue if available, else None."""
    try:
        return _chat_queue.get_nowait()
    except Exception:
        return None


def get_command_nowait() -> Optional[dict[str, Any]]:
    try:
        return _command_queue.get_nowait()
    except Exception:
        return None


def enqueue_chat(text: str) -> None:
    if not text:
        return
    try:
        _chat_queue.put(str(text))
    except Exception:
        pass


def register_handler(command_type: str, handler: Callable[[dict[str, Any]], None]) -> None:
    if not _ENABLED or not command_type or not callable(handler):
        return
    key = str(command_type).strip().lower()
    if not key:
        return
    with _handler_lock:
        _handlers.setdefault(key, []).append(handler)


def _reader_loop():
    # Read JSON lines from stdin and dispatch supported commands
    while True:
        try:
            line = sys.stdin.readline()
            if line == "":
                # EOF or no data; small backoff
                try:
                    import time
                    time.sleep(0.05)
                except Exception:
                    pass
                if sys.stdin.closed:
                    break
                continue
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            mtype = str(data.get("type") or "").strip().lower()
            payload = data.get("payload") or {}
            # Debug logging for chat messages
            try:
                if mtype == "chat":
                    with open("/tmp/nerion_ipc_debug.log", "a") as f:
                        import datetime
                        f.write(f"[{datetime.datetime.now()}] Received chat: {payload}\n")
            except Exception:
                pass
            if mtype == "chat":
                text = str((payload.get("text") if isinstance(payload, dict) else payload) or "").strip()
                if text:
                    _chat_queue.put(text)
                continue

            entry = {"type": mtype, "payload": payload if isinstance(payload, dict) else {}}
            handled = False
            if mtype:
                key = mtype
                with _handler_lock:
                    callbacks = list(_handlers.get(key, []))
                for cb in callbacks:
                    try:
                        cb(entry["payload"])
                        handled = True
                    except Exception:
                        # Keep bridge resilient; fall back to queueing below
                        handled = False
            if not handled:
                try:
                    _command_queue.put(entry)
                except Exception:
                    pass
        except Exception:
            # Keep the reader resilient
            try:
                import time
                time.sleep(0.05)
            except Exception:
                pass


def start_reader_once() -> bool:
    """Start stdin reader thread exactly once (noop if disabled)."""
    global _reader_started
    if not _ENABLED or _reader_started:
        return False
    try:
        th = threading.Thread(target=_reader_loop, name="ipc_stdio_reader", daemon=True)
        th.start()
        _reader_started = True
        return True
    except Exception:
        return False
