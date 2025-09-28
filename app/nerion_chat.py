"""Voice/chat runner for Nerion.

Responsibilities:
- wire up speech input/output, key controls, memory bridge, and self-coding triggers
- host the interactive loop (press-to-talk or VAD)
- expose `run_self_coding_pipeline` wrapper and `run_loop()` entrypoint
"""

# Import main loop engine
from .chat.engine import run_main_loop
from .chat.self_coding import run_self_coding_pipeline as _core_run_self_coding
from .chat.commands import MATCH_THRESHOLD as MATCH_THRESHOLD
from .chat.commands import matches_command as _matches_command
from .chat.commands import fuzzy_matches_command as _fuzzy_matches_command
from .config import load_config
import logging
from ops.security.net_gate import NetworkGate
import os
import datetime as dt
import sys
from pathlib import Path
from .chat.session_state import (
    _load_session_state_if_fresh,
    set_state_accessors,
)

from .chat.state import ChatState
from .chat.ptt import init as ptt_init, PttController
import warnings
import time
import threading
from typing import Optional
from .chat.tts_router import init as tts_configure, init_tts as tts_init, set_callbacks as tts_set_callbacks
from .chat.memory_bridge import init as mem_init, LongTermMemory
from selfcoder.config import allow_network as _allow_net
from .chat.ui_bridge import maybe_launch as holo_maybe_launch, wire_tts_callbacks as holo_wire_tts_callbacks
from .chat import ipc_electron as _ipc
from .chat.llm import build_chain
from .chat.voice_io import (
    safe_speak, cancel_speech, initial_calibration, set_device_index, set_voice_state,
)
from .chat.context import (
    _auto_title,
)

# When running under the Electron stdio bridge, route arbitrary prints to stderr so
# stdout remains reserved for JSON events consumed by the renderer.
if _ipc.enabled():
    sys.stdout = sys.__stderr__

# Load local environment variables from .env (if present)
try:
    from dotenv import load_dotenv  # type: ignore

    _DOTENV_OVERRIDE = os.getenv('NERION_DOTENV_PATH')
    if _DOTENV_OVERRIDE:
        candidates = [Path(_DOTENV_OVERRIDE)]
    else:
        base = Path(__file__).resolve().parents[1]
        candidates = [base / '.env', Path.cwd() / '.env']
    loaded = False
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate)
            loaded = True
            break
    if not loaded:
        load_dotenv()
except Exception:
    # If python-dotenv is not installed or load fails, continue silently
    pass

# --- Search environment normalization helper ---
def _ensure_search_env():
    """Normalize search env via shared helper and print a one-line status."""
    try:
        from selfcoder.analysis.search_api import normalize_search_env as _norm
        prov, has_key = _norm()
    except Exception:
        prov = os.getenv('NERION_SEARCH_PROVIDER', '').strip().lower() or 'duck'
        has_key = bool(os.getenv('NERION_SEARCH_API_KEY', '').strip())
    net = _allow_net()
    key_status = 'present' if has_key else 'missing'
    DEBUG = bool((os.getenv('NERION_DEBUG') or '').strip())
    if DEBUG:
        logging.getLogger(__name__).debug("[Search] provider=%s key=%s network=%s", prov, key_status, 'on' if net else 'off')
# Optional SpeechRecognition import for CI/servers without audio stack
try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None  # type: ignore

# --- Voice I/O helpers (speak, listen, etc.) and context imported above ---

# --- Backward-compat self-coding pipeline wrapper ---
def run_self_coding_pipeline(instruction: str, speak=None, listen_once=None) -> bool:
    """
    Back-compat wrapper for self-coding pipeline.

    - If `speak` / `listen_once` are not provided, use module-level functions so tests
      can stub `nc.speak` / `nc.listen_once` as done in test_voice_pipeline.
    - If they are provided (e.g., from the runtime path that passes safe_speak and listen_once),
      delegate through as-is.
    """
    if speak is None:
        # Prefer plain speak if present (tests stub this), otherwise fall back to safe_speak.
        speak = globals().get("speak") or globals().get("safe_speak")
    if listen_once is None:
        listen_once = globals().get("listen_once")
    # Profile hint for self-coding tasks
    try:
        from selfcoder.policy.profile_resolver import decide as _dec, apply_env_scoped as _apply_env_scoped
        dec = _dec('self_coding')
        if dec and dec.name:
            try:
                print(f"[profile] hint: {dec.name} ({dec.why})")
            except Exception:
                pass
            try:
                scope = _apply_env_scoped(dec)
            except Exception:
                scope = None
            try:
                return _core_run_self_coding(instruction, speak, listen_once)
            finally:
                if scope and hasattr(scope, '__exit__'):
                    try:
                        scope.__exit__(None, None, None)
                    except Exception:
                        pass
    except Exception:
        pass
    return _core_run_self_coding(instruction, speak, listen_once)


# Guarded import for pynput
try:
    import importlib
    _kb = importlib.import_module('pynput.keyboard')
    _HAS_PYNPUT = True
except Exception:
    _kb = None
    _HAS_PYNPUT = False
INITIAL_RETRY_DELAY = 1
MAX_RETRIES = 3

def with_retries(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = MAX_RETRIES
        delay = INITIAL_RETRY_DELAY
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f'Retry {attempt + 1} failed: {e}')
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
    return wrapper
logger = logging.getLogger(__name__)
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings('ignore', category=NotOpenSSLWarning)
except ImportError:
    pass
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
try:
    from .version import BUILD_TAG  # prefer centralized version tag
except Exception:
    BUILD_TAG = '2025-08-11 control-fuzzy v3'
EXIT_COMMANDS = {'exit', 'quit', 'shutdown', 'goodbye', 'good bye', 'bye', 'terminate'}
SLEEP_COMMANDS = {'stop listening', 'sleep', 'go to sleep', 'talk later', 'see you'}
INTERRUPT_COMMANDS = {'hold on', 'stop', 'pause'}
MUTE_COMMANDS = {'mute', 'be quiet', 'silence', 'stop speaking', 'mute yourself'}
UNMUTE_COMMANDS = {'unmute', 'speak', 'sound on', 'voice on'}
CONTROL_ALL = EXIT_COMMANDS | SLEEP_COMMANDS | INTERRUPT_COMMANDS | MUTE_COMMANDS | UNMUTE_COMMANDS
MEMORY_FILE = 'memory_db.json'
VOICE_NAME_PREFERRED = 'Daniel'
TTS_RATE = 190



# Centralized chat state (speech/mute/voice settings)
STATE = ChatState()

# Optional external PTT controller (for UI integrations)
_ptt_controller: Optional[PttController] = None

def ptt_press() -> None:
    """Programmatically press-to-talk (for external UI bindings)."""
    global _ptt_controller
    if _ptt_controller is not None:
        _ptt_controller.press()


def ptt_release() -> None:
    """Programmatically release press-to-talk (for external UI bindings)."""
    global _ptt_controller
    if _ptt_controller is not None:
        _ptt_controller.release()

def is_speech_enabled() -> bool:
    # Speech enabled flag (not considering mute)
    return bool(STATE.voice.enabled)

def toggle_speech() -> bool:
    return STATE.toggle_speech()

def is_muted() -> bool:
    return bool(STATE.muted)

def set_muted(on: bool) -> None:
    STATE.set_mute(on)

# Voice configuration (loaded from app/settings.yaml)
_voice_mode: str = 'ptt'
_device_hint: Optional[str] = None
_barge_in: bool = False
_sr_device_index: Optional[int] = None  # speech_recognition device index

# Runtime speech enable/disable (hard split: mic + TTS)
_speech_enabled: bool = True

def set_speech_enabled(state: bool) -> None:
    """Toggle full speech stack (mic + TTS)."""
    global _speech_enabled
    prev = bool(STATE.voice.enabled)
    STATE.set_speech(bool(state))
    _speech_enabled = bool(STATE.voice.enabled)
    if not _speech_enabled:
        try:
            cancel_speech()
        except Exception:
            pass
        print("🔇 Speech OFF (voice + mic disabled)")
    elif prev is False and _speech_enabled:
        print("🔊 Speech ON (PTT enabled; replies will be spoken)")

_pending_intent = {
    'active': False,
    'question': None,
    'ts': 0.0,
    'pending_new_convo': False,
}




def _load_last_artifact_from_state(state: ChatState, max_chars: int = 12000) -> Optional[str]:
    try:
        active = state.active
        if not active:
            return None
        path = getattr(active, 'last_artifact_path', None)
        if not path or not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()
            if len(data) > max_chars:
                data = data[:max_chars] + '\n…'
            return data
    except Exception:
        return None


# --- Lightweight task + slot parser (rules; no LLM) -------------------------
# Shared task/slot parser imported from .chat.slots as _parse_task_slots



# --- Settings and device helpers ---

def _resolve_sr_device(name_substr: Optional[str]) -> Optional[int]:
    """Best-effort map a device name substring to a speech_recognition device index."""
    try:
        if not name_substr:
            return None
        names = sr.Microphone.list_microphone_names()
        low = name_substr.lower()
        for idx, nm in enumerate(names):
            if low in (nm or '').lower():
                return idx
    except Exception:
        pass
    return None



# --- Streaming mic frames (PCM16 @16kHz, ~20ms) -----------------------------



@with_retries
def main(custom_state: Optional[ChatState] = None) -> None:
    logger.info('main() called')
    try:
        # If a custom state is provided, use it. Otherwise, use the global STATE.
        global STATE
        if custom_state:
            STATE = custom_state
        
        # Route chat model if not pinned by profile/env
        try:
            from selfcoder.llm_router import apply_router_env as _route_llm
            _route_llm(instruction=None, file=None, task='chat')
        except Exception:
            pass
        _chain = build_chain()
        # Load unified app config
        cfg = load_config()
        # Initialize network gate from config/env for consistent policy
        try:
            NetworkGate.init({
                "allow_network_access": bool(_allow_net()),
                "net": {"idle_revoke_after": os.getenv("NERION_NET_IDLE_REVOKE", "15m")},
                "paths": {"net_audit_log": os.getenv("NERION_NET_AUDIT", os.path.join('out','security_audit','net_gate.log'))}
            })
        except Exception:
            pass
        # --- Normalize search env and print status ---
        _ensure_search_env()
        # Be tolerant to both dict and object forms
        if isinstance(cfg, dict):
            voice_cfg = cfg.get('voice', {}) or {}
        else:
            # assume dataclass-like with `.voice` attr and maybe `.to_dict()`
            voice_cfg = getattr(cfg, 'voice', {}) or {}
            if hasattr(voice_cfg, 'to_dict'):
                voice_cfg = voice_cfg.to_dict()
            elif hasattr(voice_cfg, '__dict__'):
                voice_cfg = dict(voice_cfg.__dict__)
            elif not isinstance(voice_cfg, dict):
                voice_cfg = {}
        CFG = {'voice': voice_cfg}
        try:
            ptt_init(CFG)
        except Exception:
            pass
        try:
            mem_init(CFG)
        except Exception:
            pass
        global _voice_mode, _device_hint, _barge_in, _sr_device_index
        _voice_mode = str(voice_cfg.get('mode', 'ptt')).strip().lower()
        _device_hint = voice_cfg.get('device')
        _barge_in = bool(voice_cfg.get('barge_in', False))
        _sr_device_index = _resolve_sr_device(_device_hint)
        try:
            set_voice_state(STATE)
            set_device_index(_sr_device_index)
        except Exception:
            pass
        # Seed centralized state from settings
        try:
            STATE.set_voice(
                ptt=(_voice_mode == 'ptt'),
                device_hint=_device_hint,
                barge_in=bool(_barge_in),
            )
            STATE.set_speech(bool(voice_cfg.get('always_speak', True)))
            # Keep legacy flag in sync for older call sites
            global _speech_enabled
            _speech_enabled = bool(STATE.voice.enabled)
        except Exception:
            pass
        try:
            set_state_accessors(STATE, _auto_title)
        except Exception:
            pass
        try:
            tts_configure({'voice': voice_cfg})
        except Exception:
            # Fallback to legacy path if configure fails
            tts_init(voice_cfg.get('tts_backend') if isinstance(voice_cfg, dict) else None,
                     rate=TTS_RATE, preferred_voice=VOICE_NAME_PREFERRED)
        # Try to restore recent short-term conversation state (48h TTL)
        try:
            restored, n_turns = _load_session_state_if_fresh()
            if restored:
                print(f"[SESSION] Restored recent context (last {n_turns} turn(s)).")
        except Exception:
            pass
        # Default to concise mode unless explicitly disabled
        try:
            cmode_env = (os.getenv('NERION_CONCISE_DEFAULT') or '1').strip().lower()
            cmode = cmode_env in {'1','true','yes','on'}
            setattr(STATE, '_concise_mode', cmode)
        except Exception:
            pass
        # Initialize memory store (after config so path overrides apply)
        _mem = LongTermMemory(MEMORY_FILE)
        print(f"[VOICE] device hint: {repr(_device_hint)} -> index {_sr_device_index}")
        print(f"[VOICE] mode: {_voice_mode}")
        print(f'✅ Nerion build: {BUILD_TAG}')
        print('🎧 Calibrating microphone…')
        try:
            dur = float(os.getenv('NERION_CALIBRATE_SECS', '2.0'))
        except Exception:
            dur = 2.0
        initial_calibration(max(0.5, min(10.0, dur)))
        if not _ipc.enabled():
            holo_maybe_launch()
            # Only wire stdin callbacks for the spawned Electron bridge
            holo_wire_tts_callbacks(tts_set_callbacks)
        # (Upgrade prompt is offered inside the engine loop when appropriate)
        now = dt.datetime.now()
        hr = now.hour
        if hr < 12:
            greeting = "Good morning, I'm ready for your commands."
        elif hr < 18:
            greeting = "Good afternoon, I'm ready for your commands."
        else:
            greeting = "Good evening, I'm ready for your commands."
        print('Nerion:', greeting)
        safe_speak(greeting)
        # Run learn review on start by default (best-effort, non-blocking).
        # Set NERION_LEARN_ON_START=0 to disable.
        try:
            _los = (os.getenv('NERION_LEARN_ON_START') or '').strip().lower()
            _enable = True
            if _los in {'0','false','no','off'}:
                _enable = False
            elif _los in {'1','true','yes','on'}:
                _enable = True
            if _enable:
                def _bg_learn():
                    try:
                        from selfcoder.learning.continuous import review_outcomes as _rev
                        _rev()
                    except Exception:
                        pass
                threading.Thread(target=_bg_learn, name='learn_on_start', daemon=True).start()
        except Exception:
            pass
        run_main_loop(STATE, voice_cfg)
    except Exception:
        logger.exception('Unhandled error in main')
        raise
if __name__ == '__main__':
    main()

def run_loop():
    """Public runner shim for external callers/tests."""
    return main()

# --- Test-facing wrappers (exported for compatibility) ----------------------
def matches_command(heard: str, phrases: set[str]) -> bool:
    try:
        return _matches_command(heard, phrases)
    except Exception:
        return False

def fuzzy_matches_command(heard: str, phrases: set[str], threshold: float = MATCH_THRESHOLD) -> bool:
    try:
        return _fuzzy_matches_command(heard, phrases, threshold)
    except Exception:
        return False
