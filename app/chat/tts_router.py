from __future__ import annotations
import os
import sys
import threading
import subprocess
import time
from queue import SimpleQueue
from typing import Optional, Callable

try:
    import pyttsx3  # type: ignore
except Exception:  # pragma: no cover
    pyttsx3 = None  # pyttsx3 may be unavailable in some environments

# --- Public callbacks (wired by app.nerion_chat) ---------------------------
_on_start: Optional[Callable[[dict], None]] = None
_on_stop: Optional[Callable[[dict], None]] = None
_on_word: Optional[Callable[[int], None]] = None  # receives token length

def set_callbacks(on_start=None, on_stop=None, on_word=None) -> None:
    """Register optional callbacks used by the UI layer (e.g., HOLO app)."""
    global _on_start, _on_stop, _on_word
    _on_start = on_start
    _on_stop = on_stop
    _on_word = on_word

# --- Backend selection -----------------------------------------------------
_backend_choice: Optional[str] = None  # 'say' | 'pyttsx3' | 'piper' | 'coqui'
_DEF_TRUE = {"1", "true", "yes", "on"}

# Current TTS parameters (shared by both backends)
_current_rate: int = 190
_current_voice: str = "Daniel"

def resolve_backend(preference: Optional[str]) -> str:
    """Resolve backend to one of {'say','pyttsx3','piper','coqui'}.

    Priority: explicit env → config preference → platform default.
    Piper/Coqui are used only when explicitly selected.
    """
    env = (os.getenv('NERION_TTS_BACKEND', '') or '').strip().lower()
    if env in {'say', 'pyttsx3', 'piper', 'coqui'}:
        return env
    if os.getenv('NERION_FORCE_SAY', '').strip().lower() in _DEF_TRUE:
        return 'say'
    pref = (preference or '').strip().lower() if preference else 'auto'
    if pref in {'say','pyttsx3','piper','coqui'}:
        return pref
    # Platform default
    return 'say' if sys.platform == 'darwin' else 'pyttsx3'


# --- Unified config-based initializer (for app) ---
def init(cfg=None) -> None:
    """
    Configure TTS from a unified config object.

    Accepts either:
      - cfg['voice'] with keys: 'tts_backend', 'rate', 'preferred_voice'
      - or cfg['tts']       with keys: 'backend', 'rate', 'voice'
    Falls back to environment and defaults handled by init_tts().
    This reads cfg['voice'] or cfg['tts']; other sections are ignored.
    """
    pref = None
    rate = 190
    voice = 'Daniel'
    try:
        if isinstance(cfg, dict):
            v = cfg.get('voice') or {}
            if isinstance(v, dict):
                pref = v.get('tts_backend', pref)
                try:
                    rate = int(v.get('rate', rate))
                except Exception:
                    pass
                voice = v.get('preferred_voice', voice)
            tts_cfg = cfg.get('tts') or {}
            if isinstance(tts_cfg, dict):
                pref = tts_cfg.get('backend', pref)
                try:
                    rate = int(tts_cfg.get('rate', rate))
                except Exception:
                    pass
                voice = tts_cfg.get('voice', voice)
    except Exception:
        # Defensive: never let config parsing break TTS
        pass
    # persist for runtime (used by both backends)
    global _current_rate, _current_voice
    _current_rate, _current_voice = rate, voice
    init_tts(pref, rate=rate, preferred_voice=voice)

# --- pyttsx3 engine + worker ----------------------------------------------
_engine = None  # lazily initialized
_say_proc = None
_play_proc = None  # external WAV player for piper/coqui
_say_lock = threading.Lock()

_tts_queue: SimpleQueue[str] = SimpleQueue()
_tts_cancel = threading.Event()
_tts_thread: Optional[threading.Thread] = None
_init_lock = threading.RLock()

def _emit_start():
    if callable(_on_start):
        try:
            _on_start({"type": "speak_start"})
        except Exception:
            pass

def _emit_stop():
    if callable(_on_stop):
        try:
            _on_stop({"type": "speak_stop"})
        except Exception:
            pass

def _emit_word(n: int):
    if callable(_on_word):
        try:
            _on_word(n)
        except Exception:
            pass

def _on_tts_word(name, location, length):
    # pyttsx3 callback signature -> forward approximate token signal
    try:
        _emit_word(max(1, int(length or 1)))
    except Exception:
        pass

def _tts_worker_loop():
    """Consume queued text and speak via pyttsx3. Preemptible via _tts_cancel."""
    global _engine
    while True:
        try:
            text = _tts_queue.get()
            if text is None:
                break
            if _engine is None:
                # No engine -> nothing to do
                continue
            _tts_cancel.clear()
            _emit_start()
            try:
                import re
                parts = [p.strip() for p in re.split(r'([.!?]+\s+)', text)]
                chunks = []
                buf = ''
                for p in parts:
                    buf += p
                    if re.search(r'[.!?]\s*$', p):
                        if buf.strip():
                            chunks.append(buf.strip())
                        buf = ''
                if buf.strip():
                    chunks.append(buf.strip())
                for seg in chunks:
                    if _tts_cancel.is_set():
                        break
                    try:
                        _engine.say(seg)
                        _engine.runAndWait()
                    except Exception:
                        break
            finally:
                _emit_stop()
        except Exception:
            try:
                _emit_stop()
            except Exception:
                pass

def _init_pyttsx3_engine(rate: int, preferred_voice: str):
    """Initialize pyttsx3 engine and background worker."""
    global _engine, _tts_thread
    if pyttsx3 is None:
        _engine = None
        return
    try:
        with _init_lock:
            _engine = pyttsx3.init()
        try:
            _engine.setProperty('rate', rate)
            voices = _engine.getProperty('voices')
            selected_id = None
            for v in voices:
                if preferred_voice.lower() in (getattr(v, 'name', '') or '').lower():
                    selected_id = getattr(v, 'id', None)
                    break
            if selected_id is None and voices:
                for v in voices:
                    if 'english' in (getattr(v, 'name', '') or '').lower():
                        selected_id = getattr(v, 'id', None)
                        break
            if selected_id:
                _engine.setProperty('voice', selected_id)
        except Exception:
            pass
        try:
            _engine.connect('started-word', _on_tts_word)
        except Exception:
            pass
        if _tts_thread is None or not _tts_thread.is_alive():
            _tts_thread = threading.Thread(target=_tts_worker_loop, name='nerion_tts', daemon=True)
            _tts_thread.start()
    except Exception:
        _engine = None  # fall back to say

# --- Public init/speak/cancel ---------------------------------------------
def init_tts(preference: Optional[str], *, rate: int = 190, preferred_voice: str = 'Daniel') -> None:
    """Initialize TTS according to settings/env/platform."""
    global _backend_choice, _engine
    global _current_rate, _current_voice
    _current_rate, _current_voice = rate, preferred_voice
    if os.environ.get('NERION_TESTING'):
        _backend_choice = 'pyttsx3'
        return
    choice = resolve_backend(preference)
    _backend_choice = choice
    if choice == 'pyttsx3':
        _init_pyttsx3_engine(rate=rate, preferred_voice=preferred_voice)
        print('[TTS] backend: pyttsx3')
    elif choice == 'say':
        _engine = None
        print('[TTS] backend: macOS say' + (' (forced)' if (os.getenv('NERION_TTS_BACKEND','').strip().lower()=='say' or os.getenv('NERION_FORCE_SAY','').strip().lower() in _DEF_TRUE) else ' (auto)'))
    elif choice in {'piper','coqui'}:
        _engine = None
        print(f"[TTS] backend: {choice} (experimental)")
        # Best-effort env validation
        if choice == 'piper':
            m = os.getenv('PIPER_MODEL_PATH') or os.getenv('NERION_PIPER_MODEL')
            if not m:
                print('[TTS:piper] warning: PIPER_MODEL_PATH not set (local model required)')
            if not _which('piper'):
                print('[TTS:piper] warning: piper CLI not found on PATH')
        if choice == 'coqui':
            m = os.getenv('COQUI_MODEL_PATH') or os.getenv('NERION_COQUI_MODEL')
            if not m:
                print('[TTS:coqui] warning: COQUI_MODEL_PATH not set (local model required)')
            cmd = os.getenv('COQUI_TTS_CMD') or 'tts'
            if not _which(cmd):
                print(f"[TTS:coqui] warning: '{cmd}' CLI not found on PATH")
    else:
        _engine = None
        print('[TTS] backend: unknown; falling back to platform default')

def _speak_via_say(text: str) -> None:
    """Use macOS 'say' for quick one-shot speaking."""
    global _say_proc
    if not text:
        return
    try:
        with _say_lock:
            try:
                if _say_proc and _say_proc.poll() is None:
                    _say_proc.terminate()
            except Exception:
                pass
            _emit_start()
            # Build say command with current params
            cmd = ["/usr/bin/say"]
            try:
                if _current_rate:
                    cmd += ["-r", str(int(_current_rate))]
            except Exception:
                pass
            try:
                if _current_voice and isinstance(_current_voice, str):
                    cmd += ["-v", _current_voice]
            except Exception:
                pass
            cmd.append(text)
            _say_proc = subprocess.Popen(cmd)

            # Watcher to emit stop when the say process exits
            def _watch_say(p):
                try:
                    p.wait()
                finally:
                    _emit_stop()
                    # Clear handle once done
                    with _say_lock:
                        _say_proc = None
            t = threading.Thread(target=_watch_say, args=(_say_proc,), daemon=True)
            t.start()
    except Exception:
        _emit_stop()
        try:
            with _say_lock:
                _say_proc = None
        except Exception:
            pass


def _which(cmd: str) -> Optional[str]:
    try:
        import shutil as _sh
        return _sh.which(cmd)
    except Exception:
        return None


def _play_wav(path: str) -> None:
    """Play a WAV file using a simple system player (afplay/aplay/ffplay)."""
    global _play_proc
    player = None
    if sys.platform == 'darwin':
        player = _which('afplay')
    else:
        player = _which('aplay') or _which('ffplay')
    if not player:
        # As a last resort, do nothing but emit stop
        _emit_stop()
        return
    try:
        if player.endswith('ffplay'):
            _play_proc = subprocess.Popen([player, '-nodisp', '-autoexit', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            _play_proc = subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def _watch_play(p):
            try:
                p.wait()
            finally:
                _emit_stop()
                try:
                    os.remove(path)
                except Exception:
                    pass
        threading.Thread(target=_watch_play, args=(_play_proc,), daemon=True).start()
    except Exception:
        _emit_stop()
        try:
            os.remove(path)
        except Exception:
            pass


def _speak_via_piper(text: str) -> None:
    """Synthesize speech via piper CLI to a temp WAV, then play.

    Requires: `piper` on PATH and PIPER_MODEL_PATH env set to a local model.
    """
    if not text:
        return
    model = os.getenv('PIPER_MODEL_PATH') or os.getenv('NERION_PIPER_MODEL')
    if not model:
        print('[TTS:piper] missing PIPER_MODEL_PATH; falling back')
        _speak_via_say(text) if sys.platform == 'darwin' else _tts_queue.put(text)
        return
    if not _which('piper'):
        print('[TTS:piper] piper CLI not found; falling back')
        _speak_via_say(text) if sys.platform == 'darwin' else _tts_queue.put(text)
        return
    import tempfile
    try:
        _emit_start()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            wav_path = tmp.name
        # Build command: piper -m <model> -f <wav> -t <text>
        cmd = ['piper', '-m', model, '-f', wav_path, '-t', text]
        # Best-effort rate mapping via speaking rate not supported directly; ignore
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(wav_path):
            _play_wav(wav_path)
        else:
            try:
                os.remove(wav_path)
            except Exception:
                pass
            if proc.returncode != 0:
                try:
                    err = (proc.stderr or b'').decode('utf-8', errors='ignore').strip()
                except Exception:
                    err = ''
                print(f"[TTS:piper] synthesis failed (rc={proc.returncode}) {(': ' + err) if err else ''}")
            _emit_stop()
    except Exception:
        _emit_stop()


def _speak_via_coqui(text: str) -> None:
    """Synthesize via Coqui TTS CLI if available, else no-op.

    Requires: `tts` on PATH and COQUI_MODEL_PATH pointing to a local model.
    """
    if not text:
        return
    model = os.getenv('COQUI_MODEL_PATH') or os.getenv('NERION_COQUI_MODEL')
    tts_cmd = os.getenv('COQUI_TTS_CMD') or 'tts'
    if not model:
        print('[TTS:coqui] missing COQUI_MODEL_PATH; falling back')
        _speak_via_say(text) if sys.platform == 'darwin' else _tts_queue.put(text)
        return
    if not _which(tts_cmd):
        print(f"[TTS:coqui] '{tts_cmd}' CLI not found; falling back")
        _speak_via_say(text) if sys.platform == 'darwin' else _tts_queue.put(text)
        return
    import tempfile
    try:
        _emit_start()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            wav_path = tmp.name
        # Example CLI: tts --text "hello" --model_path <model> --out_path out.wav
        # Map _current_rate -> length_scale (lower = faster, clamp to [0.6,1.6])
        try:
            length_scale = max(0.6, min(1.6, 190.0 / float(_current_rate or 190)))
        except Exception:
            length_scale = 1.0
        cmd = [tts_cmd, '--text', text, '--model_path', model, '--out_path', wav_path, '--length_scale', str(length_scale)]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(wav_path):
            _play_wav(wav_path)
        else:
            try:
                os.remove(wav_path)
            except Exception:
                pass
            if proc.returncode != 0:
                try:
                    err = (proc.stderr or b'').decode('utf-8', errors='ignore').strip()
                except Exception:
                    err = ''
                print(f"[TTS:coqui] synthesis failed (rc={proc.returncode}) {(': ' + err) if err else ''}")
            _emit_stop()
    except Exception:
        _emit_stop()

def speak(text: str) -> None:
    """Enqueue text for speech (pyttsx3) or delegate to 'say'."""
    if not text:
        return
    try:
        # Resolve backend lazily if needed
        if _backend_choice is None:
            init_tts(None, rate=_current_rate, preferred_voice=_current_voice)
        if _backend_choice == 'say':
            _speak_via_say(text)
            return
        if _backend_choice == 'piper':
            _speak_via_piper(text)
            return
        if _backend_choice == 'coqui':
            _speak_via_coqui(text)
            return
        # pyttsx3 (default)
        if _engine is not None:
            _tts_queue.put(text)
        else:
            # Fallback to say
            _speak_via_say(text)
    except Exception:
        pass

def cancel_speech() -> None:
    """Hard-stop any ongoing TTS and drain queued utterances."""
    global _say_proc
    try:
        _tts_cancel.set()
    except Exception:
        pass
    try:
        if _engine:
            _engine.stop()
    except Exception:
        pass
    try:
        while True:
            _ = _tts_queue.get_nowait()
    except Exception:
        pass
    try:
        with _say_lock:
            if _say_proc and _say_proc.poll() is None:
                try:
                    _say_proc.terminate()
                    time.sleep(0.05)
                except Exception:
                    pass
                if _say_proc.poll() is None:
                    try:
                        _say_proc.kill()
                    except Exception:
                        pass
            _say_proc = None
            # Stop external player if running
            global _play_proc
            try:
                if _play_proc and _play_proc.poll() is None:
                    _play_proc.terminate()
            except Exception:
                pass
            _play_proc = None
    except Exception:
        pass
    try:
        _emit_stop()
    except Exception:
        pass


def reset() -> None:
    """Tear down TTS state (engine, threads, processes). Safe for hot-reload."""
    global _engine, _tts_thread, _say_proc
    try:
        cancel_speech()
    except Exception:
        pass
    try:
        if _engine:
            _engine.stop()
    except Exception:
        pass
    _engine = None
    _tts_thread = None
    with _say_lock:
        _say_proc = None


def set_params(*, rate: Optional[int] = None, voice: Optional[str] = None) -> None:
    """Dynamically adjust TTS params for both backends.

    If pyttsx3 is active, update engine properties. For 'say', the next
    utterance will pick up the new params.
    """
    global _current_rate, _current_voice
    if rate is not None:
        _current_rate = int(rate)
        try:
            if _engine:
                _engine.setProperty('rate', _current_rate)
        except Exception:
            pass
    if voice is not None:
        _current_voice = voice
        try:
            if _engine and pyttsx3 is not None:
                # Try to select a matching voice id
                voices = _engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in (getattr(v, 'name', '') or '').lower():
                        _engine.setProperty('voice', getattr(v, 'id', None))
                        break
        except Exception:
            pass


def get_backend() -> Optional[str]:
    """Return the active backend ('say' or 'pyttsx3') or None if uninitialized."""
    return _backend_choice


def shutdown() -> None:
    """Gracefully stop TTS worker threads and processes."""
    try:
        cancel_speech()
    except Exception:
        pass
    try:
        if _tts_thread and _tts_thread.is_alive():
            _tts_queue.put(None)  # sentinel to end worker
    except Exception:
        pass
