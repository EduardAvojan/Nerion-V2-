from __future__ import annotations
from typing import Optional, Callable
import threading
import time
import sys
from queue import SimpleQueue

# Guarded import for pynput
try:
    from pynput import keyboard as _kb
    _HAS_PYNPUT = True
except Exception:
    _kb = None
    _HAS_PYNPUT = False

# --- Key resolution helpers -------------------------------------------------
def _resolve_key(name: str):
    """Translate a simple key name to a pynput key object when available.
    Supports: 'space', 'f1', 'f2', ..., single letters like 'q'.
    Falls back to the original string if pynput is unavailable.
    """
    if not _HAS_PYNPUT:
        return name
    n = (name or '').strip().lower()
    if not n:
        return None
    # Function keys
    if n.startswith('f') and n[1:].isdigit():
        try:
            return getattr(_kb.Key, n)
        except Exception:
            return None
    # Named keys
    named = {
        'space': _kb.Key.space,
        'esc': _kb.Key.esc,
        'escape': _kb.Key.esc,
        'enter': _kb.Key.enter,
        'return': _kb.Key.enter,
        'capslock': getattr(_kb.Key, 'caps_lock', None),
        'caps_lock': getattr(_kb.Key, 'caps_lock', None),
        'pause': getattr(_kb.Key, 'pause', None),
    }
    if n in named:
        return named[n]
    # Single char
    if len(n) == 1:
        class _Char:
            def __init__(self, c):
                self.char = c
        return _Char(n)
    return None

# --- Optional module initializer for config injection ---
_cfg_ref = None

_keymap = {
    'ptt': 'space',
    # Multiple options to avoid host/app conflicts (VS Code reserves F1/F2)
    # Parsed as comma-separated preference order.
    'toggle': 'caps_lock,f9,f12,f1,f2',
    # Optional modifier+key combos for environments where plain F-keys
    # are intercepted. Comma-separated list.
    'toggle_combo': 'ctrl+f9,ctrl+f12,cmd+f9,cmd+f12',
    'quit': 'q',
    'debounce_ms': 65,
}

def _load_keymap_from_cfg(cfg) -> None:
    global _keymap
    try:
        km = (cfg or {}).get('keymap') if isinstance(cfg, dict) else None
        if hasattr(cfg, 'get') and km is None:
            # allow nested like cfg['chat']['keymap']
            chat = cfg.get('chat') if isinstance(cfg.get('chat'), dict) else None
            km = chat.get('keymap') if chat else None
        if km is None and isinstance(chat, dict):
            km = chat.get('keys')
        if isinstance(km, dict):
            _keymap.update({k: v for k, v in km.items() if k in _keymap})
    except Exception:
        pass

def init(cfg=None) -> None:
    """
    Optional initializer for PTT module. Currently stores a config reference
    for future use (e.g., debounce tuning, keybinds). No behavior change.
    """
    global _cfg_ref
    _cfg_ref = cfg
    _load_keymap_from_cfg(cfg)

class PttController:
    """
    Minimal PTT controller abstraction.

    - `press()` triggers the provided `on_press` callback exactly once when
      transitioning from up→down.
    - `release()` triggers `on_release` on down→up.
    - Thread-safe; can be called from UI threads, key hooks, or tests.

    The actual keyboard binding (e.g., Space) remains in `nerion_chat.py` via
    `KeyWatcher`. This controller simply offers a stable API for external UIs
    (Electron, StreamDeck, etc.) to drive push-to-talk without importing
    pynput.
    """

    def __init__(self,
                 on_press: Callable[[], None],
                 on_release: Callable[[], None]):
        self.on_press = on_press
        self.on_release = on_release
        self._pressed = False
        self._lock = threading.RLock()

    def press(self) -> None:
        with self._lock:
            if not self._pressed:
                self._pressed = True
                try:
                    self.on_press()
                except Exception:
                    pass

    def release(self) -> None:
        with self._lock:
            if self._pressed:
                self._pressed = False
                try:
                    self.on_release()
                except Exception:
                    pass

    def is_pressed(self) -> bool:
        with self._lock:
            return self._pressed

    def toggle(self) -> None:
        # convenience for non-momentary switches
        if self.is_pressed():
            self.release()
        else:
            self.press()

    def __enter__(self):
        self.press()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


# ---- KeyWatcher and ChatInput: UI helpers for PTT and terminal chat ----

class KeyWatcher:
    """Keyboard listener for Space hold-to-talk (PTT) with debounce.

    Pure UI helper:
    - Exposes .space_down/.quit_pressed/.toggled events
    - Calls optional callbacks instead of importing app.nerion_chat
    """
    def __init__(self,
                 debounce_ms: int = None,
                 on_first_space: Optional[Callable[[], None]] = None,
                 toggle_speech: Optional[Callable[[], None]] = None,
                 on_quit: Optional[Callable[[], None]] = None,
                 ptt_key: Optional[str] = None,
                 toggle_key: Optional[str] = None,
                 quit_key: Optional[str] = None):
        # Events
        self.space_down = threading.Event()
        self.quit_pressed = threading.Event()
        self.toggled = threading.Event()
        self._listener = None
        # Resolve debounce and keys (prefer explicit args, else cfg keymap, else defaults)
        try:
            dm = debounce_ms if debounce_ms is not None else _keymap.get('debounce_ms', 65)
        except Exception:
            dm = 65
        self._debounce_s = max(0, dm) / 1000.0
        self._last_press_ts = 0.0
        self._on_first_space = on_first_space
        self._toggle_speech = toggle_speech
        self._on_quit = on_quit
        # Store configured key names for debugging
        self._ptt_key_name = (ptt_key or _keymap.get('ptt', 'space'))
        self._toggle_key_name = (toggle_key or _keymap.get('toggle', 'f9'))
        self._quit_key_name = (quit_key or _keymap.get('quit', 'q'))
        # Pre-resolve keys to speed comparisons when pynput is present
        self._ptt_key = _resolve_key(self._ptt_key_name)
        # Support multiple toggle keys (comma-separated)
        try:
            _toggles = [s.strip() for s in str(self._toggle_key_name).split(',') if s.strip()]
        except Exception:
            _toggles = [str(self._toggle_key_name)]
        self._toggle_keys = [_resolve_key(k) for k in _toggles if k]
        self._quit_char = (self._quit_key_name or '').lower()
        # Parse combo toggles
        try:
            combo_str = None
            if isinstance(_cfg_ref, dict):
                combo_str = (_cfg_ref.get('keymap') or {}).get('toggle_combo')
            if combo_str is None:
                combo_str = _keymap.get('toggle_combo', '')
            combos = [s.strip() for s in str(combo_str).split(',') if s.strip()]
        except Exception:
            combos = []
        self._toggle_combos = []  # list of (mods:set[str], key)
        for c in combos:
            parts = [p.strip().lower() for p in c.split('+') if p.strip()]
            if not parts:
                continue
            key_name = parts[-1]
            mods = set(parts[:-1])
            self._toggle_combos.append((mods, _resolve_key(key_name)))
        # Modifier state
        self._mod_ctrl = False
        self._mod_alt = False
        self._mod_cmd = False

    def start(self) -> bool:
        if not _HAS_PYNPUT:
            return False
        if _kb is None:
            return False

        def _is_key(k, target):
            # Compare against Key (e.g., Key.space) or a char-like object
            try:
                if hasattr(target, 'char'):
                    return getattr(k, 'char', '').lower() == target.char
                return k == target
            except Exception:
                return False

        def _is_toggle_key(k) -> bool:
            try:
                for t in self._toggle_keys:
                    if t is None:
                        continue
                    if hasattr(t, 'char'):
                        if getattr(k, 'char', '').lower() == getattr(t, 'char', '').lower():
                            return True
                    else:
                        if k == t:
                            return True
            except Exception:
                pass
            return False

        def _is_toggle_combo(k) -> bool:
            try:
                # Current modifier snapshot
                mods_now = set()
                if self._mod_ctrl:
                    mods_now.add('ctrl')
                if self._mod_alt:
                    mods_now.add('alt')
                if self._mod_cmd:
                    mods_now.add('cmd')
                for mods_req, key_req in self._toggle_combos:
                    if key_req is None:
                        continue
                    # key match
                    matched = False
                    if hasattr(key_req, 'char'):
                        matched = getattr(k, 'char', '').lower() == getattr(key_req, 'char', '').lower()
                    else:
                        matched = (k == key_req)
                    if not matched:
                        continue
                    # mods subset
                    if mods_req.issubset(mods_now):
                        return True
            except Exception:
                pass
            return False

        def on_press(key):
            try:
                # Track modifiers
                if getattr(key, 'ctrl', False) or key in {getattr(_kb.Key, 'ctrl', None), getattr(_kb.Key, 'ctrl_l', None), getattr(_kb.Key, 'ctrl_r', None)}:
                    self._mod_ctrl = True
                if key in {getattr(_kb.Key, 'alt', None), getattr(_kb.Key, 'alt_l', None), getattr(_kb.Key, 'alt_r', None), getattr(_kb.Key, 'alt_gr', None)}:
                    self._mod_alt = True
                if key in {getattr(_kb.Key, 'cmd', None), getattr(_kb.Key, 'cmd_l', None), getattr(_kb.Key, 'cmd_r', None), getattr(_kb.Key, 'super', None)}:
                    self._mod_cmd = True
                # Quit key: single char like 'q'
                if getattr(key, 'char', '').lower() == self._quit_char:
                    self.quit_pressed.set()
                    if self._on_quit:
                        try:
                            self._on_quit()
                        except Exception:
                            pass
                    return

                # Toggle key(s) (prefer non-conflicting keys like F9/F12) or combos
                if _is_toggle_key(key) or _is_toggle_combo(key):
                    if self._toggle_speech:
                        try:
                            self._toggle_speech()
                        except Exception:
                            pass
                    self.toggled.set()
                    return

                # PTT key (default: Space)
                if self._ptt_key is not None and _is_key(key, self._ptt_key):
                    now = time.time()
                    if self.space_down.is_set() and (now - self._last_press_ts) < self._debounce_s:
                        return
                    self._last_press_ts = now
                    if not self.space_down.is_set():
                        self.space_down.set()
                        # barge-in immediately on first press
                        if self._on_first_space:
                            try:
                                self._on_first_space()
                            except Exception:
                                pass
            except Exception:
                pass

        def on_release(key):
            try:
                # Track modifiers
                if getattr(key, 'ctrl', False) or key in {getattr(_kb.Key, 'ctrl', None), getattr(_kb.Key, 'ctrl_l', None), getattr(_kb.Key, 'ctrl_r', None)}:
                    self._mod_ctrl = False
                if key in {getattr(_kb.Key, 'alt', None), getattr(_kb.Key, 'alt_l', None), getattr(_kb.Key, 'alt_r', None), getattr(_kb.Key, 'alt_gr', None)}:
                    self._mod_alt = False
                if key in {getattr(_kb.Key, 'cmd', None), getattr(_kb.Key, 'cmd_l', None), getattr(_kb.Key, 'cmd_r', None), getattr(_kb.Key, 'super', None)}:
                    self._mod_cmd = False
                if self._ptt_key is not None and (key == self._ptt_key or getattr(key, 'char', '').lower() == getattr(self._ptt_key, 'char', '').lower()):
                    self.space_down.clear()
            except Exception:
                pass

        self._listener = _kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        return True

    def stop(self):
        try:
            if self._listener:
                self._listener.stop()
                # Give it a moment to finish callbacks
                try:
                    self._listener.join(0.2)
                except Exception:
                    pass
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


class ChatInput:
    """Background reader for terminal lines; pushes into a queue without blocking the main loop."""
    def __init__(self):
        self.q = SimpleQueue()
        self._th = None
        self._stop = threading.Event()

    def start(self) -> bool:
        # Only start if stdin is a TTY
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return False
        except Exception:
            return False

        def _reader():
            while not self._stop.is_set():
                try:
                    line = sys.stdin.readline()
                    if line == '':
                        # EOF or no data; backoff briefly
                        time.sleep(0.05)
                        if sys.stdin.closed:
                            break
                        continue
                    self.q.put(line.rstrip('\n'))
                except Exception:
                    time.sleep(0.05)
                    continue

        self._th = threading.Thread(target=_reader, name='chat_input', daemon=True)
        self._th.start()
        return True

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except Exception:
            return None

    def stop(self):
        try:
            self._stop.set()
        except Exception:
            pass
