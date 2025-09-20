
from __future__ import annotations
from typing import Callable, Optional, Tuple
import re
import difflib

# Supported slash commands:
#   /speech on|off   (alias: /voice on|off, /toggle)
#   /mute on|off
#   /say <text>
#   /q | /quit       (skip this interaction)
#   /exit            (treated like /q by default; caller may override)
#   /stt <backend> [model]   (set offline STT profile; backends: whisper|vosk|sphinx|auto)

def try_parse_command(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()
    if not t.startswith("/"):
        return None, None
    low = t.lower()
    if low.startswith("/speech "):
        return "speech", t.split(maxsplit=1)[1]
    if low.startswith("/voice "):
        return "speech", t.split(maxsplit=1)[1]
    if low.strip() in {"/toggle", "/voice", "/speech"}:
        return "toggle", ""
    if low.startswith("/mute "):
        return "mute", t.split(maxsplit=1)[1]
    if low.startswith("/say "):
        return "say", t[5:]
    if low.startswith("/stt "):
        return "stt", t.split(maxsplit=1)[1]
    if low in {"/q", "/quit"}:
        return "quit", ""
    if low in {"/exit", "/shutdown"}:
        return "exit", ""
    return None, None


def handle_command(
    cmd: str,
    arg: str,
    *,
    speak_fn: Callable[[str], None],
    set_speech_enabled: Callable[[bool], None],
    get_speech_enabled: Optional[Callable[[], bool]] = None,
    set_mute_fn: Optional[Callable[[bool], None]] = None,
    set_tts_params: Optional[Callable[[Optional[int], Optional[str]], None]] = None,
    set_stt_profile_fn: Optional[Callable[[Optional[str], Optional[str]], None]] = None,
    on_exit: Optional[Callable[[], None]] = None,
) -> bool:
    """Handle a parsed slash command.

    Returns True to continue the current loop, False to *skip this turn*.
    (Callers may interpret False as "do not capture mic now" – like /q.)
    """
    if cmd == "speech":
        on = (arg or "").strip().lower() == "on"
        set_speech_enabled(on)
        speak_fn("🔊 Speech ON" if on else "🔇 Speech OFF")
        return True
    if cmd == "toggle":
        cur = False
        try:
            if get_speech_enabled is not None:
                cur = bool(get_speech_enabled())
        except Exception:
            cur = False
        set_speech_enabled(not cur)
        speak_fn("🔊 Speech ON" if (not cur) else "🔇 Speech OFF")
        return True

    if cmd == "mute":
        if set_mute_fn is not None:
            on = (arg or "").strip().lower() == "on"
            set_mute_fn(on)
            speak_fn("🔇 Nerion muted." if on else "🔊 Nerion unmuted.")
        return True

    if cmd == "say":
        a = (arg or "").strip()
        low = a.lower()
        if low.startswith("rate ") and set_tts_params is not None:
            try:
                val = int(a.split(maxsplit=1)[1])
                set_tts_params(val, None)
                speak_fn(f"TTS rate set to {val}.")
                return True
            except Exception:
                speak_fn("Couldn't set TTS rate.")
                return True
        if low.startswith("voice ") and set_tts_params is not None:
            try:
                name = a.split(maxsplit=1)[1].strip()
                set_tts_params(None, name)
                speak_fn(f"Voice set to {name}.")
                return True
            except Exception:
                speak_fn("Couldn't set voice.")
                return True
        speak_fn(a)
        return True

    if cmd == "stt":
        raw = (arg or "").strip()
        if not raw:
            speak_fn("Usage: /stt <whisper|vosk|sphinx|auto> [model]")
            return True
        parts = raw.split()
        backend = parts[0].strip().lower()
        model = parts[1].strip().lower() if len(parts) > 1 else None
        if backend in {"default", "auto"}:
            backend = "auto"
        elif backend not in {"whisper", "vosk", "sphinx"}:
            speak_fn("Unknown STT backend. Use whisper, vosk, sphinx, or auto.")
            return True
        if set_stt_profile_fn is not None:
            try:
                set_stt_profile_fn(backend, model)
                if model:
                    speak_fn(f"STT set to {backend} {model}.")
                else:
                    speak_fn(f"STT set to {backend}.")
            except Exception:
                speak_fn("Couldn't set STT profile.")
        else:
            speak_fn("STT profile change is not available in this mode.")
        return True

    if cmd == "quit":
        # skip this turn (do not proceed to mic capture)
        return False

    if cmd == "exit":
        if on_exit is not None:
            on_exit()
        # by default, treat as skip-this-turn so caller decides
        return False

    return True

# --- Robust command matching and core controls (ASR/voice) ---

MATCH_THRESHOLD = 0.82

def _normalize(text: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9\s]', ' ', (text or '').lower()).split())

def matches_command(heard: str, phrases: set[str]) -> bool:
    """
    Robustly match full command phrases with word boundaries, ignoring punctuation/casing.
    Avoids substring bugs like 'mute' matching 'unmute'.
    """
    norm = ' ' + _normalize(heard) + ' '
    for p in phrases:
        p_norm = ' ' + _normalize(p) + ' '
        if p_norm in norm:
            return True
    return False

def fuzzy_matches_command(heard: str, phrases: set[str], threshold: float = MATCH_THRESHOLD) -> bool:
    """
    Fuzzy match for short commands to survive ASR errors (e.g., 'mewt' ~ 'mute').
    Token-level comparison using SequenceMatcher ratio.
    """
    norm = _normalize(heard)
    heard_tokens = norm.split()
    for p in phrases:
        p_tokens = _normalize(p).split()
        if len(p_tokens) <= 3:
            for i in range(0, max(1, len(heard_tokens) - len(p_tokens) + 1)):
                window = ' '.join(heard_tokens[i:i + len(p_tokens)])
                if difflib.SequenceMatcher(None, window, ' '.join(p_tokens)).ratio() >= threshold:
                    return True
    return False

def is_command(heard: str, phrases: set[str]) -> bool:
    return matches_command(heard, phrases) or fuzzy_matches_command(heard, phrases)

def _has_token(text: str, token: str) -> bool:
    return re.search(rf'\b{re.escape(token)}\b', _normalize(text)) is not None

def handle_core_controls(
    heard: str,
    *,
    set_mute: Callable[[bool], None],
    set_speech_enabled: Callable[[bool], None],
) -> Optional[str]:
    """
    Returns one of {"mute","unmute","sleep","exit","interrupt"} if a core control was handled,
    otherwise None. This is a HARD check that runs before any LLM/memory.
    """
    u = _normalize(heard)
    if _has_token(u, 'unmute') or u in {'voice on', 'sound on', 'speak'}:
        print('[CTRL] HARD mute=OFF')
        set_mute(False)
        return 'unmute'
    if _has_token(u, 'mute') and (not _has_token(u, 'unmute')):
        print('[CTRL] HARD mute=ON')
        set_mute(True)
        return 'mute'
    if u in {'stop listening', 'sleep', 'go to sleep', 'talk later', 'see you'}:
        print('[CTRL] HARD sleep')
        return 'sleep'
    if u in {'exit', 'quit', 'shutdown', 'goodbye', 'good bye', 'bye', 'terminate'}:
        print('[CTRL] HARD exit')
        return 'exit'
    if u in {'hold on', 'stop', 'pause'}:
        print('[CTRL] HARD interrupt')
        return 'interrupt'
    if u in {'speech off', 'voice off'}:
        set_speech_enabled(False)
        return 'mute'
    if u in {'speech on', 'voice on'}:
        set_speech_enabled(True)
        return 'unmute'
    return None
