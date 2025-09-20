

"""Thin bridge to the Electron HOLO app via stdin.

This module isolates all Electron/stdio wiring from nerion_chat.py to keep the
chat loop lean. It is safe to import on systems without Electron installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Callable, Optional
from typing import Any

_holo_proc: Optional[subprocess.Popen] = None
_holo_stdin = None


def maybe_launch(base_dir: Optional[str] = None) -> bool:
    """Launch the Electron HOLO app if available. Returns True on success.

    Looks for `app/ui/holo-app` relative to `base_dir` (or this file's dir).
    If `electron` is not installed it prints a helpful message and returns False.
    """
    global _holo_proc, _holo_stdin
    if _holo_proc and _holo_proc.poll() is None:
        return True
    try:
        here = os.path.dirname(__file__)
        # Default to app/… root (two levels up from this file)
        root = base_dir or os.path.dirname(here)
        app_dir = os.path.join(root, "ui", "holo-app")
        pkg = os.path.join(app_dir, "package.json")
        if not os.path.exists(pkg):
            return False
        electron_bin = os.path.join(app_dir, "node_modules", ".bin", "electron")
        if os.path.exists(electron_bin):
            cmd = [electron_bin, "."]
        elif shutil.which("electron"):
            cmd = ["electron", "."]
        else:
            print("[HOLO] Electron not found. Run: cd app/ui/holo-app && npm install")
            return False

        proc = subprocess.Popen(
            cmd,
            cwd=app_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _holo_proc = proc
        _holo_stdin = proc.stdin
        print("[HOLO] Launched Electron holo (stdin bridge)")
        return True
    except Exception as e:
        print(f"[HOLO] Launch failed: {e}")
        _holo_proc = None
        _holo_stdin = None
        return False


def send_event(payload: dict) -> None:
    """Send a single JSON line event to the HOLO app, if running."""
    global _holo_stdin
    if _holo_stdin:
        try:
            _holo_stdin.write(json.dumps(payload) + "\n")
            _holo_stdin.flush()
        except Exception:
            # If the pipe is broken, ignore gracefully.
            pass


def send_patch_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
    data: dict[str, Any] = {"type": event_type}
    if payload is not None:
        data["payload"] = payload
    send_event(data)


def on_tts_word(length: int) -> None:
    """Helper used by TTS 'on_word' callback to visualize tokens."""
    token = "x" * max(1, int(length or 1))
    send_event({"type": "word", "token": token})


def wire_tts_callbacks(set_callbacks: Callable) -> None:
    """Connect TTS callbacks to HOLO event stream. Safe on failure."""
    try:
        def _on_start(_payload):
            send_event({"type": "speak_start"})
            if (os.getenv('NERION_VOICE_STATUS') or '0').strip().lower() in {'1','true','yes','on'}:
                print('🔊 Speaking…')
        def _on_stop(_payload):
            send_event({"type": "speak_stop"})
            if (os.getenv('NERION_VOICE_STATUS') or '0').strip().lower() in {'1','true','yes','on'}:
                print('🔈 Done.')
        set_callbacks(
            on_start=_on_start,
            on_stop=_on_stop,
            on_word=lambda n: on_tts_word(n),
        )
    except Exception:
        # TTS backends may not support callbacks; ignore quietly.
        pass
