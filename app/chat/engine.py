"""Nerion chat engine: the main interaction loop.
This module owns the while-loop and per-turn handlers so the runner stays thin.
It imports the same helper functions the runner used, avoiding circular imports.
"""

from __future__ import annotations
import os
import re
import time
from typing import Optional
import concurrent.futures as _futures
import threading
from contextlib import suppress
from types import SimpleNamespace

# ---- Parent + Notebook wiring (Incubator Phase 1) -----------------------------------
from app.parent.driver import ParentDriver, ParentLLM
from app.parent.tools_manifest import ToolsManifest as _ToolsManifest
from app.parent.tools_manifest import load_tools_manifest_from_yaml as _load_manifest_from_yaml
from app.logging.experience import ExperienceLogger as _ExperienceLogger

# Commands / IO / intents
from .commands import try_parse_command, handle_command, is_command, handle_core_controls
from .voice_io import (
    speak, safe_speak, cancel_speech, listen_once, _ptt_stream_transcribe,
    set_stt_profile as _set_stt_profile,
)
from .voice_io import tts_set_params as _tts_set_params
from .context import _normalize, _relation_to_context, _needs_disambiguation, _auto_title
from .llm import (
    _build_followup_prompt,
    _make_clarifying_question,
    _strip_think_blocks,
    build_chain_with_temp,
)
from .session_state import _save_session_state
from .slots import parse_task_slots as _parse_task_slots
from .net_access import _fmt_time_left as _fmt_time_left
from .net_access import status_chip as _status_chip  # noqa: F401 - re-exported for tests and external importers
from .net_access import NetworkGate as NetworkGate
from .ptt import PttController, KeyWatcher, ChatInput


# Site‑query intent
from .routes_web import run_site_query as _run_site_query, run_web_search as _run_web_search
# Data-driven intents (rules, local/web)
from .intents import load_intents as _load_intent_rules, detect_intent as _detect_intent, call_handler as _call_intent_handler

# Semantic fallback (offline-first)
from app.chat.semantic_parser import (
    configure_intents as _sem_cfg,
    configure_model as _sem_model,
    parse_intent_by_similarity as _sem_match,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
from .help import get_help as _help_text
from .parent_exec import build_executor as _build_parent_executor
from .offline_tools import run_healthcheck as _hc_offline, run_diagnostics as _diag_offline
from .ux_busy import BusyIndicator as _Busy
from app.learning.upgrade_agent import maybe_offer_upgrade, handle_choice as _upgrade_handle_choice
from .offline_tools import (
    get_current_time as _off_time,
    get_current_date as _off_date,
    recall_memory as _off_recall,
)
try:
    from selfcoder.upgrade.shadow import should_shadow as _shadow_should, schedule_shadow_replay as _shadow_schedule
except Exception:  # pragma: no cover
    def _shadow_should() -> bool:
        return False
    def _shadow_schedule() -> None:
        return None

# Memory
from .memory_bridge import LongTermMemory
from .memory_session import SessionCache
from . import ipc_electron as _ipc
from .ui_controller import ElectronCommandRouter


# Lightweight debug helper (enabled when NERION_DEBUG is truthy)
DEBUG = bool((os.getenv('NERION_DEBUG') or '').strip())
def _dbg(msg: str) -> None:
    try:
        if DEBUG:
            print('[DEBUG]', msg)
    except Exception:
        pass


TRUTHY = {'1', 'true', 'yes', 'on'}
_SESSION_CACHE: SessionCache | None = None
_MEMORY_STORE: LongTermMemory | None = None


class _VirtualPTTWatcher(SimpleNamespace):
    """Event-driven watcher used when Electron handles PTT externally."""

    def __init__(self):
        super().__init__()
        self.space_down = threading.Event()
        self.quit_pressed = threading.Event()

    def start(self) -> bool:
        return True

    def stop(self) -> None:
        self.space_down.clear()
        self.quit_pressed.clear()


def _bootstrap_memory_state() -> None:
    global _SESSION_CACHE, _MEMORY_STORE
    if _SESSION_CACHE is not None and _MEMORY_STORE is not None:
        return
    try:
        ns = {
            'user': os.getenv('USER', 'global'),
            'workspace': os.getenv('NERION_SCOPE_WS', 'default'),
            'project': os.getenv('NERION_SCOPE_PROJECT', 'default'),
        }
        mem_path = os.getenv('NERION_MEMORY_PATH', 'memory_db.json')
        _MEMORY_STORE = LongTermMemory(path=mem_path, ns=ns)
        session_path = os.getenv('NERION_SESSION_FILE', 'out/memory_session.json')
        try:
            max_turns = int(os.getenv('NERION_SESSION_MAX_TURNS', '50'))
        except Exception:
            max_turns = 50
        _SESSION_CACHE = SessionCache(path=session_path, ns=ns, max_turns=max_turns)
        _SESSION_CACHE.load()
        try:
            decay = float(os.getenv('NERION_MEMORY_DECAY', '0.15'))
        except Exception:
            decay = 0.15
        try:
            ttl_days = int(os.getenv('NERION_MEMORY_TTL_DAYS', '14'))
        except Exception:
            ttl_days = 14
        try:
            promotion_threshold = float(os.getenv('NERION_MEMORY_PROMOTION', '2.5'))
        except Exception:
            promotion_threshold = 2.5
        promos = _SESSION_CACHE.decay_and_prune(decay_per_day=decay, default_ttl_days=ttl_days, promotion_threshold=promotion_threshold)
        for promo in promos:
            fact = promo.get('fact')
            if not fact:
                continue
            _MEMORY_STORE.add_fact(
                fact,
                tags=promo.get('tags') or [],
                scope='long',
                provenance='session_promote',
                confidence=0.8,
                ttl_days=promo.get('ttl_days'),
            )
    except Exception as exc:
        _dbg(f"[session] bootstrap failed: {exc!r}")
        _SESSION_CACHE = None
        if _MEMORY_STORE is None:
            try:
                _MEMORY_STORE = LongTermMemory('memory_db.json')
            except Exception:
                _MEMORY_STORE = None


def _session_shutdown() -> None:
    if _SESSION_CACHE is None:
        return
    try:
        _SESSION_CACHE.save()
    except Exception as exc:
        _dbg(f"[session] save failed: {exc!r}")


def _infer_session_tags(text: str) -> list[str]:
    low = (text or '').lower()
    tags: set[str] = set()
    if any(w in low for w in ["pizza", "food", "restaurant", "cook", "eat", "coffee", "tea", "lunch", "dinner"]):
        tags.add('food')
    if any(w in low for w in ["python", "code", "git", "refactor", "test", "build", "deploy", "tool"]):
        tags.add('tools')
    if any(w in low for w in ["work", "job", "office", "meeting", "project"]):
        tags.add('work')
    if any(w in low for w in ["like", "love", "enjoy"]):
        tags.add('positive')
    if any(w in low for w in ["dislike", "hate", "annoyed", "angry"]):
        tags.add('negative')
    return sorted(tags)


def _session_record_user(text: str, mem: LongTermMemory) -> None:
    if _SESSION_CACHE is None:
        return
    try:
        _SESSION_CACHE.record_turn('user', text)
        extracted = mem.extract_from_utterance(text)
        ttl_hint = getattr(mem, '_last_ttl_days', None)
        try:
            promote_n = max(1, int(os.getenv('NERION_SESSION_PROMOTE_N', '3')))
        except Exception:
            promote_n = 3
        for idx, fact in enumerate(extracted):
            ttl = ttl_hint if (ttl_hint is not None and idx == len(extracted) - 1) else None
            item = _SESSION_CACHE.upsert_short_fact(fact, _infer_session_tags(fact), score=0.5, ttl_days=ttl)
            if item.get('count', 0) >= promote_n:
                mem.add_fact(fact, tags=_infer_session_tags(fact), scope='long', provenance='session_promote', confidence=0.8, ttl_days=ttl)
                item['count'] = 0
    except Exception as exc:
        _dbg(f"[session] user record failed: {exc!r}")


def _session_record_assistant(reply: str) -> None:
    if _SESSION_CACHE is None:
        return
    try:
        _SESSION_CACHE.record_turn('assistant', reply)
        if (os.getenv('NERION_SESSION_SUMMARY', '1') or '').strip().lower() in TRUTHY:
            turns = _SESSION_CACHE.state.get('turns', [])
            _SESSION_CACHE.update_summary(turns)
    except Exception as exc:
        _dbg(f"[session] assistant record failed: {exc!r}")


def sanitize_for_prompt(text: str) -> str:
    clean = re.sub(r"(?i)(system:|developer:|ignore previous|override)", "", text or "")
    return clean.strip()[:512]


def _render_memory_block(items):
    lines = [f"- {item.get('text')}" for item in items if item.get('text')]
    if not lines:
        return ''
    return "Here are things about the user that might help:\n" + "\n".join(lines) + "\n"


def _process_typed_line(text: str, STATE, watcher):
    if text is None:
        return None, False, False
    txt = text.strip()
    if not txt:
        return None, False, False
    c, a = try_parse_command(txt)
    if c is not None:
        ok = handle_command(
            c,
            a or "",
            speak_fn=lambda s: safe_speak(s, watcher),
            set_speech_enabled=lambda v: STATE.set_speech(bool(v)),
            get_speech_enabled=lambda: bool(STATE.voice.enabled),
            set_mute_fn=lambda m: STATE.set_mute(bool(m)),
            set_tts_params=lambda rate, voice: _tts_set_params(rate=rate, voice=voice),
            set_stt_profile_fn=lambda b, m: _set_stt_profile(b, m),
        )
        return None, True, not ok
    if txt.startswith('>'):
        return txt[1:].strip(), True, False
    if not txt.startswith('/'):
        # Forward user turn to UI when running under Electron
        try:
            if _ipc.enabled():
                _ipc.emit('chat_turn', {'role': 'user', 'text': txt})
        except Exception:
            pass
        return txt, True, False
    return None, True, False


def _poll_chat_queue(chat, STATE, watcher):
    if chat is None:
        # Also poll Electron stdio queue if enabled
        try:
            if _ipc.enabled():
                typed = _ipc.get_nowait()
                if typed is not None:
                    return _process_typed_line(typed, STATE, watcher)
        except Exception:
            pass
        return None, False, False
    typed = None
    try:
        typed = chat.get_nowait()
    except Exception:
        typed = None

    if typed is None:
        # Fall back to Electron stdio queue (if active)
        try:
            if _ipc.enabled():
                typed = _ipc.get_nowait()
        except Exception:
            typed = None

    if typed is None:
        return None, False, False

    return _process_typed_line(typed, STATE, watcher)


def _collect_ptt_input(STATE, watcher, chat):
    # Always drain any queued chat text before falling back to microphone input.
    while True:
        heard, consumed, skip_turn = _poll_chat_queue(chat, STATE, watcher)
        if not consumed:
            break
        if skip_turn:
            return None, True
        if heard:
            return heard, False

    if watcher:
        while not watcher.space_down.is_set() and not watcher.quit_pressed.is_set():
            heard, consumed, skip_turn = _poll_chat_queue(chat, STATE, watcher)
            if consumed:
                if skip_turn:
                    return None, True
                if heard:
                    return heard, False
            time.sleep(0.02)
        if watcher and watcher.quit_pressed.is_set():
            watcher.quit_pressed.clear()
            return None, True
        if not STATE.voice.enabled:
            return None, False
        heard = _ptt_stream_transcribe(watcher, getattr(STATE.voice, 'device_hint', None)) or listen_once(timeout=4, phrase_time_limit=3)
        return heard, False

    try:
        print('\n[PTT] Press Enter to speak (or type /q to skip)…', end='', flush=True)
        user = input()
    except EOFError:
        user = ''
    cmd = (user or '').strip()
    if cmd.lower() == '/q':
        return None, True
    if cmd.lower() in {'/voice off', '/speech off'}:
        STATE.set_speech(False)
        return None, True
    if cmd.lower() in {'/voice on', '/speech on'}:
        STATE.set_speech(True)
        return None, True
    heard = listen_once(timeout=10, phrase_time_limit=8)
    return heard, False


def _collect_open_input(STATE, chat, watcher):
    while True:
        heard, consumed, skip_turn = _poll_chat_queue(chat, STATE, watcher)
        if not consumed:
            break
        if skip_turn:
            return None, True
        if heard:
            return heard, False
    if not STATE.voice.enabled:
        time.sleep(0.05)
        return None, False
    heard = listen_once(timeout=10, phrase_time_limit=8)
    return heard, False


def _collect_user_input(STATE, chat, watcher):
    if STATE.voice.ptt:
        return _collect_ptt_input(STATE, watcher, chat)
    return _collect_open_input(STATE, chat, watcher)


def _predict_with_timeout(chain, prompt: str, timeout_s: float | None):
    if timeout_s is not None:
        try:
            timeout_val = float(timeout_s)
        except Exception:
            timeout_val = 0.0
    else:
        timeout_val = 0.0
    if timeout_val and timeout_val > 0:
        with _futures.ThreadPoolExecutor(max_workers=1) as _exec:
            future = _exec.submit(chain.predict, input=prompt)
            try:
                return future.result(timeout=timeout_val)
            except _futures.TimeoutError as exc:
                future.cancel()
                raise TimeoutError(f'LLM response timed out after {timeout_val:.1f}s') from exc
    return chain.predict(input=prompt)


def _handle_memory_controls(heard: str, mem: LongTermMemory, watcher) -> bool:
    pin_days = re.match(r'^\s*pin\s+(that|it|this)\s+for\s+(?P<n>\d+)\s+day(s)?\s*$', heard, re.I)
    if pin_days:
        try:
            duration = int(pin_days.group('n'))
        except Exception:
            duration = None
        if duration is not None:
            candidate = getattr(mem, 'last_ref', None)
            if candidate and mem.set_ttl_for_text(candidate, duration):
                msg = f"Pinned for {duration} day(s)."
            else:
                msg = "No recent memory to pin for a duration."
            print(msg)
            safe_speak(msg, watcher)
            return True

    forget_match = re.match(r'^\s*forget\s*(?:that\s+)?(?P<q>.*)$', heard, re.I)
    if forget_match:
        query = (forget_match.group('q') or '').strip()
        removed, matched = mem.forget_smart(query, last_hint=None)
        msg = f"Removed 1 memory item: {matched}." if removed else "I couldn't find a matching memory to forget."
        print(msg)
        safe_speak(msg, watcher)
        return True

    unpin_match = re.match(r'^\s*unpin\s*(?P<q>.*)$', heard, re.I)
    if unpin_match:
        query = (unpin_match.group('q') or '').strip()
        removed, matched = mem.unpin_smart(query, last_hint=None)
        msg = f"Unpinned 1 item: {matched}." if removed else "I couldn't find a pinned item to unpin."
        print(msg)
        safe_speak(msg, watcher)
        return True

    pin_match = re.match(r'^\s*pin\s+(that|it|this)\s*$', heard, re.I)
    if pin_match:
        candidate = getattr(mem, 'last_ref', None)
        msg = ('Pinned 1 memory item.' if mem.pin_fact_text(candidate or '') else 'Nothing to pin.') if candidate else 'No recent memory fact to pin.'
        print(msg)
        safe_speak(msg, watcher)
        return True

    return False


# --- Learning trigger (debounced) -------------------------------------------
_LEARN_LAST_TS = 0.0
_LEARN_DEBOUNCE_S = None
def _maybe_schedule_learn_review():
    global _LEARN_LAST_TS, _LEARN_DEBOUNCE_S
    try:
        if _LEARN_DEBOUNCE_S is None:
            try:
                _LEARN_DEBOUNCE_S = float((os.getenv('NERION_LEARN_DEBOUNCE_S') or '300').strip())
            except Exception:
                _LEARN_DEBOUNCE_S = 300.0
        now = time.time()
        if (now - _LEARN_LAST_TS) < float(_LEARN_DEBOUNCE_S or 0):
            return
        _LEARN_LAST_TS = now
        def _bg():
            try:
                from selfcoder.learning.continuous import review_outcomes as _rev
                _rev()
            except Exception:
                pass
        threading.Thread(target=_bg, name='learn_review_debounced', daemon=True).start()
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Helpers moved from runner (small, deterministic)

def _load_last_artifact_from_state(state, max_chars: int = 12000) -> Optional[str]:
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

# _parse_task_slots is imported from .slots to avoid duplication


# ----------------------------------------------------------------------------
# ---- Parent + Notebook wiring (Incubator Phase 1) -----------------------------------
# Parent ON by default; set NERION_USE_PARENT=0/false/no to disable

def _env_false(val: str | None) -> bool:
    return (val or '').strip().lower() in {'0', 'false', 'no'}

_PARENT_ENABLED = not _env_false(os.getenv('NERION_USE_PARENT'))
_LOGGER = _ExperienceLogger()

if _PARENT_ENABLED:
    if DEBUG:
        print("[Parent] Enabled (default). Set NERION_USE_PARENT=0 to disable.")
else:
    if DEBUG:
        print("[Parent] Disabled. Set NERION_USE_PARENT=1 to enable.")

def _load_tools_manifest() -> _ToolsManifest:
    try:
        if os.path.exists(os.path.join('config','tools.yaml')):
            return _load_manifest_from_yaml(os.path.join('config','tools.yaml'))
    except Exception:
        pass
    # Fallback: empty manifest is allowed; Parent still plans in abstract
    return _ToolsManifest(tools=[])

class _NoopParentLLM(ParentLLM):
    """Default Parent adapter if none is provided. Returns a clarify plan.
    Replace with a DeepSeek-backed adapter and set NERION_USE_PARENT=1 to enable.
    """
    def complete(self, messages):
        return (
            '{"intent":"clarify","plan":[{"action":"ask_user","tool":null,'
            '"args":{},"summary":"clarify request"}],"final_response":null,'
            '"confidence":0.0,"requires_network":false,"notes":null}'
        )

_PARENT_DRIVER = None
if _PARENT_ENABLED:
    try:
        # Try to use DeepSeek local model first
        try:
            from app.parent.deepseek_local import DeepSeekLocalLLM
            _PARENT_DRIVER = ParentDriver(llm=DeepSeekLocalLLM(), tools=_load_tools_manifest())
            if DEBUG:
                print('[Parent] Master\'s Voice enabled (DeepSeek local adapter).')
        except ImportError:
            # Fall back to Noop adapter if DeepSeek isn't available
            _PARENT_DRIVER = ParentDriver(llm=_NoopParentLLM(), tools=_load_tools_manifest())
            if DEBUG:
                print('[Parent] Master\'s Voice enabled (Noop adapter - DeepSeek not found).')
    except Exception as e:
        _PARENT_DRIVER = None
        print(f'[Parent] Failed to initialize: {e}; continuing without Parent.')

# (moved to app/chat/net_access.py) prefs + status chip helpers


# ---- Experience Logger helper -----------------------------------
def _log_experience(user_query: str, parent_decision: dict | None, action_taken: dict, success: bool, error: str | None = None, network_used: bool | None = None):
    try:
        _LOGGER.log(
            user_query=user_query,
            parent_decision=parent_decision or {},
            action_taken=action_taken,
            outcome_success=success,
            error=error,
            latency_ms=None,
            network_used=network_used,
        )
    except Exception:
        pass
    # Optional: schedule learn review after logs (debounced)
    try:
        if (os.getenv('NERION_LEARN_ON_EVENT') or '1').strip().lower() in {'1','true','yes','on'}:
            _maybe_schedule_learn_review()
    except Exception:
        pass


def run_main_loop(STATE, voice_cfg) -> None:
    """Run the interactive loop. Expects STATE to be initialized by the runner."""
    _bootstrap_memory_state()
    mem = _MEMORY_STORE or LongTermMemory('memory_db.json')
    try:
        llm_timeout = float(os.getenv('NERION_LLM_TIMEOUT_S', '25'))
        if llm_timeout <= 0:
            llm_timeout = 0.0
    except Exception:
        llm_timeout = 12.0
    # Controls for prompt behavior
    try:
        _USE_MEM_IN_PROMPT = (os.getenv('NERION_MEMORY_PROMPT') or '0').strip().lower() in {'1','true','yes','on'}
    except Exception:
        _USE_MEM_IN_PROMPT = False

    # (Startup initialization removed here to avoid repeated calibration/greeting)

    watcher = None
    chat = None
    chat_active = False
    router: Optional[ElectronCommandRouter] = None
    if _ipc.enabled():
        router = ElectronCommandRouter(STATE, _SESSION_CACHE, mem)
        _ipc.register_handler('ptt', router.handle_ptt)
        _ipc.register_handler('override', router.handle_override)
        _ipc.register_handler('memory', router.handle_memory)
        _ipc.register_handler('health', router.handle_health)
        _ipc.register_handler('settings', router.handle_settings)
        _ipc.register_handler('llm', router.handle_llm)
        _ipc.register_handler('learning', router.handle_learning)
        _ipc.register_handler('upgrade', router.handle_upgrade)
        _ipc.register_handler('artifact', router.handle_artifact)
        _ipc.register_handler('patch', router.handle_patch)
        _ipc.register_handler('selfcode', router.handle_selfcode)
        with suppress(Exception):
            router.refresh_memory()
            router.handle_learning({'action': 'refresh'})
            router.handle_upgrade({'action': 'refresh'})
            router.handle_artifact({'action': 'refresh'})
            router._emit_health_status()
            router.emit_settings_bootstrap()
            _ipc.emit('state', {
                'phase': 'standby',
                'interaction_mode': 'talk' if STATE.voice.ptt else 'chat',
                'speech_enabled': bool(getattr(STATE.voice, 'enabled', True)),
                'muted': bool(getattr(STATE, 'muted', False)),
            })
            router.emit_llm_options()
            router.emit_metrics()

    # Load data-driven intent rules (from config/intents.yaml) once per session
    try:
        if not hasattr(STATE, '_intent_rules') or not STATE._intent_rules:
            STATE._intent_rules = _load_intent_rules()
    except Exception:
        STATE._intent_rules = []

    # Initialize semantic fallback (data-driven examples; strictly offline)
    try:
        if not hasattr(STATE, '_sem_ready') or not STATE._sem_ready:
            examples = {}
            if yaml is not None:
                try:
                    with open(os.path.join('config', 'intent_examples.yaml'), 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                        ex = data.get('examples') or {}
                        # ensure dict[str, list[str]]
                        if isinstance(ex, dict):
                            examples = {str(k): list(v or []) for k, v in ex.items()}
                except Exception as e:
                    _dbg(f"Failed to load intent_examples.yaml: {e!r}")
            _sem_cfg(examples=examples)
            # Optional local model path via env; no downloads performed
            mdir = os.getenv('NERION_SEM_MODEL_DIR')
            if mdir:
                try:
                    _sem_model(local_model_dir=mdir)
                except Exception as e:
                    _dbg(f"Failed to configure semantic model: {e!r}")
            STATE._sem_ready = True
    except Exception:
        STATE._sem_ready = True

    # PTT setup
    if STATE.voice.ptt:
        if _ipc.enabled():
            watcher = _VirtualPTTWatcher()

            def _on_ptt_press():
                try:
                    cancel_speech()
                except Exception:
                    pass
                watcher.space_down.set()

            def _on_ptt_release():
                watcher.space_down.clear()

            if router:
                router.set_watcher(watcher)
                router.set_ptt_callbacks(_on_ptt_press, _on_ptt_release)
        else:
            # Provide explicit feedback when toggling speech via CapsLock/F-keys
            def _toggle_speech_feedback():
                new_state = not STATE.voice.enabled
                STATE.set_speech(new_state)
                if new_state:
                    msg = "🔊 Speech ON — you can speak now."
                    print(msg)
                    try:
                        safe_speak(msg)
                    except Exception:
                        pass
                else:
                    msg = "🔇 Speech OFF — type-only mode."
                    print(msg)
            watcher = KeyWatcher(
                on_first_space=lambda: cancel_speech(),
                toggle_speech=_toggle_speech_feedback,
            )
            if not watcher.start():
                print('[PTT] Press Enter to speak (pynput unavailable).')
            else:
                print("[PTT] Hold SPACE to talk (press 'q' to skip)…")

                def _on_ptt_press():
                    try:
                        cancel_speech()
                    except Exception:
                        pass
                    try:
                        if watcher:
                            watcher.space_down.set()
                    except Exception:
                        pass

                def _on_ptt_release():
                    try:
                        if watcher:
                            watcher.space_down.clear()
                    except Exception:
                        pass

                _ = PttController(on_press=_on_ptt_press, on_release=_on_ptt_release)
                if router:
                    router.set_watcher(watcher)
                    router.set_ptt_callbacks(_on_ptt_press, _on_ptt_release)
            # Start background terminal chat reader (works alongside PTT)
            chat = ChatInput()
            chat_active = chat.start()
            if chat_active:
                print('[CHAT] Type "> your message" and press Enter to chat without mic (e.g., "> what day is it?")')
                print("[CHAT] Controls: /speech on | /speech off | /say <text> | /q")
                print("[CapsLock or F9/F12 (Ctrl+F9 works in VS Code)] toggles Speech ON/OFF • Type '> ...' to chat • '/speech off' disables mic+tts")
    else:
        if router:
            router.set_watcher(None)
    # If running under Electron, start stdio command reader
    try:
        if _ipc.enabled():
            _ipc.start_reader_once()
            print('[HOLO] Electron stdio bridge active')
    except Exception:
        pass

    # Optional acoustic barge-in monitor (only in VAD mode)
    if (not STATE.voice.ptt) and getattr(STATE.voice, 'barge_in', False):
        try:
            from voice.stt.recognizer import start_barge_in_monitor
            vad_cfg = (voice_cfg.get('vad') if isinstance(voice_cfg, dict) else {}) or {}
            start_barge_in_monitor(lambda: cancel_speech(), device=getattr(STATE.voice, 'device_hint', None), **{k: v for k, v in vad_cfg.items() if isinstance(k, str)})
        except Exception as e:
            _dbg(f"Barge-in monitor not started: {e!r}")

    try:
        while True:
            heard, skip_turn = _collect_user_input(STATE, chat, watcher)
            if skip_turn:
                continue
            if not heard:
                continue

            analysis_step: Optional[str] = None
            respond_step: Optional[str] = None
            confidence_score: Optional[float] = None
            confidence_drivers: list[str] = []

            if router:
                router.reset_thoughts()
                router.emit_metrics()
                analysis_step = router.thought_step('Understand request', detail=heard, status='active')

            def _complete_analysis(detail: str, status: str = 'complete') -> None:
                if router and analysis_step:
                    router.update_thought(analysis_step, status=status, detail=detail)

            def _start_response(detail: str, status: str = 'active') -> None:
                nonlocal respond_step
                if not router:
                    return
                if respond_step is None:
                    respond_step = router.thought_step('Compose response', detail=detail, status=status)
                else:
                    router.update_thought(respond_step, detail=detail, status=status)

            def _finish_response(detail: str, status: str = 'complete') -> None:
                if router and respond_step:
                    router.update_thought(respond_step, status=status, detail=detail)

            def _emit_conf(score: Optional[float], drivers: list[str]) -> None:
                if router and score is not None:
                    router.emit_confidence(score, drivers)

            _session_record_user(heard, mem)
            if router:
                router.on_user_turn()

            handled = handle_core_controls(heard, set_mute=lambda v: STATE.set_mute(bool(v)), set_speech_enabled=lambda v: STATE.set_speech(bool(v)))
            if handled:
                detail = f'Handled control: {handled}'
                _complete_analysis(detail)
                _start_response(detail, status='active')
                confidence_score = 0.98
                confidence_drivers = [detail]
                print(f'🗣️ You: {heard}')
                if handled == 'exit':
                    speak('Shutting down, goodbye.')
                    _finish_response('Shutting down', status='complete')
                    _emit_conf(confidence_score, confidence_drivers)
                    if router:
                        router.emit_metrics()
                    break
                if handled == 'sleep':
                    safe_speak("Okay, I'll keep listening.", watcher)
                    _finish_response('Standing by', status='complete')
                    _emit_conf(confidence_score, confidence_drivers)
                    if router:
                        router.emit_metrics()
                    continue
                if handled == 'interrupt':
                    safe_speak('Okay, I’ve paused. Go ahead.', watcher)
                    _finish_response('Paused on request', status='complete')
                    _emit_conf(confidence_score, confidence_drivers)
                    if router:
                        router.emit_metrics()
                    continue
                _finish_response('Control acknowledged', status='complete')
                _emit_conf(confidence_score, confidence_drivers)
                if router:
                    router.emit_metrics()
                continue

            if _handle_memory_controls(heard, mem, watcher):
                _complete_analysis('Memory command executed')
                _start_response('Updating memory state', status='active')
                _finish_response('Memory updated', status='complete')
                confidence_score = 0.9
                confidence_drivers = ['Memory command acknowledged']
                _emit_conf(confidence_score, confidence_drivers)
                if router:
                    router.refresh_memory()
                    router.emit_metrics()
                continue

            # Self-coding trigger
            def is_self_coding_trigger(utter):
                norm = ' ' + _normalize(utter) + ' '
                for p in {"upgrade yourself", "edit yourself", "we need an upgrade", "modify your code"}:
                    if (' ' + _normalize(p) + ' ') in norm:
                        return True
                return False
            if is_self_coding_trigger(heard):
                _complete_analysis('Entering self-coding mode')
                _start_response('Preparing self-coding pipeline', status='active')
                print(f'🗣️ You: {heard}')
                ack = "Entering self-coding mode. What should I change?"
                print('Nerion:', ack)
                safe_speak(ack, watcher)
                instruction = listen_once(timeout=16, phrase_time_limit=14)
                if not instruction:
                    msg = "I didn't catch the self-coding instruction."
                    print('Nerion:', msg)
                    safe_speak(msg, watcher)
                    _finish_response('No instruction captured', status='failed')
                    _emit_conf(0.3, ['Self-coding instruction missing'])
                    if router:
                        router.emit_metrics()
                    continue
                from .self_coding import run_self_coding_pipeline as _core_run_self_coding
                success = _core_run_self_coding(instruction, lambda s: safe_speak(s, watcher), listen_once)
                msg = "I applied the requested change successfully." if success else "I couldn't complete that change."
                print('[SELF-CODING]', msg)
                safe_speak(msg, watcher)
                status = 'complete' if success else 'failed'
                _finish_response('Self-coding pipeline finished', status=status)
                conf = 0.85 if success else 0.45
                driver = 'Self-coding succeeded' if success else 'Self-coding failed'
                _emit_conf(conf, [driver])
                if router:
                    router.emit_metrics()
                continue

            print(f'🗣️ You: {heard}')

            def _deliver_offline_response(text: str, detail: str, *, log_rule: str, drivers: list[str], confidence: float = 0.9, meta: Optional[dict] = None) -> None:
                _complete_analysis(detail)
                _start_response(detail, status='active')
                print('Nerion:', text)
                safe_speak(text, watcher)
                try:
                    STATE.append_turn('assistant', text)
                except Exception:
                    pass
                _finish_response('Response delivered', status='complete')
                _emit_conf(confidence, drivers)
                if router:
                    router.emit_metrics()
                extra = {"routed": "offline_fast", "rule": log_rule}
                if meta:
                    try:
                        extra.update(meta)
                    except Exception:
                        pass
                _log_experience(heard, None, extra, True, None, False)

            # Fast-path offline triage for time/date to avoid any LLM/web paths
            try:
                if re.search(r"\b(what time is it|the current time|time please)\b", heard, flags=re.I):
                    out = _off_time(heard)
                    _deliver_offline_response(out, 'Provided current time', log_rule='local.get_current_time', drivers=['Offline rule: current time'], confidence=0.95)
                    continue
                if re.search(r"\b(what(?:'s| is) today\'s date|what(?:'s| is) the date(?: today)?|what date is it(?: today)?|today(?:'s)? date|what day is it(?: today)?)\b", heard, flags=re.I):
                    out = _off_date(heard)
                    _deliver_offline_response(out, 'Provided current date', log_rule='local.get_current_date', drivers=['Offline rule: current date'], confidence=0.95)
                    continue
                # Gratitude acks (keep terse; avoid LLM)
                if re.search(r"\b(thanks|thank you|thx|ty)\b", heard, flags=re.I):
                    out = "You're welcome!"
                    _deliver_offline_response(out, 'Sent quick acknowledgement', log_rule='local.ack_thanks', drivers=['Offline rule: acknowledgement'], confidence=0.9)
                    continue
                # Help router (offline)
                m_help = re.match(r"^\s*help(?:\s+(?P<topic>[a-z0-9\- _]+))?\s*$", heard, flags=re.I)
                if m_help:
                    topic = (m_help.group('topic') or '').strip()
                    out = _help_text(topic or None)
                    _deliver_offline_response(out, 'Provided help summary', log_rule='local.help', drivers=['Offline rule: help'], confidence=0.85, meta={'topic': topic})
                    continue
                # Concise mode toggle
                m_concise = re.search(r"\b(concise|brief|terse)\s+mode\s+(on|off)\b", heard, flags=re.I)
                if m_concise:
                    on = (m_concise.group(2).lower() == 'on')
                    try:
                        setattr(STATE, '_concise_mode', on)
                    except Exception:
                        pass
                    out = "Concise mode ON (answers will be one sentence)." if on else "Concise mode OFF."
                    _deliver_offline_response(out, 'Concise mode toggled', log_rule='local.concise_mode', drivers=[f'Concise mode {"ON" if on else "OFF"}'], confidence=0.85, meta={'on': on})
                    continue
                # Cancel current operation request
                if re.search(r"\b(cancel|abort|stop operation)\b", heard, flags=re.I):
                    try:
                        setattr(STATE, '_cancel_requested', True)
                    except Exception:
                        pass
                    out = "Cancelling the current operation…"
                    _deliver_offline_response(out, 'Cancellation requested', log_rule='local.cancel', drivers=['Operation cancel signal'], confidence=0.88)
                    continue
                # Action feed (observability)
                m_actions = re.search(r"\b(show|list)\s+(?:last\s+)?(?P<n>\d{1,3})?\s*(actions|activity|events)\b", heard, flags=re.I)
                if m_actions:
                    try:
                        n = int(m_actions.group('n') or 10)
                    except Exception:
                        n = 10
                    n = max(1, min(50, n))
                    # Load last N experience records
                    try:
                        from app.logging.experience import DEFAULT_LOG_PATH as _XP_PATH
                        log_path = getattr(_LOGGER, 'path', _XP_PATH)
                        lines = []
                        with open(log_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[-n:]
                        out_lines = []
                        import json as _json
                        for ln in lines:
                            try:
                                rec = _json.loads(ln)
                                tool = ''
                                try:
                                    steps = (rec.get('action_taken') or {}).get('steps') or []
                                    if steps:
                                        tool = (steps[-1] or {}).get('tool') or ''
                                except Exception:
                                    tool = ''
                                status = 'OK' if rec.get('outcome_success') else 'ERR'
                                dur = rec.get('latency_ms')
                                routed = tool or (rec.get('action_taken') or {}).get('routed','')
                                out_lines.append(f"• {status} {routed} {f'({dur/1000:.1f}s)' if dur else ''} ({int((rec.get('ts') or 0)%86400/3600):02d}:{int((rec.get('ts') or 0)%3600/60):02d})")
                            except Exception:
                                continue
                        out = "Last actions:\n" + ("\n".join(out_lines) or "(no recent actions)")
                        detail = 'Shared recent actions'
                        drivers = ['Observability: action log']
                    except Exception as e:
                        out = f"I couldn't read the action log: {e}"
                        detail = 'Action log unavailable'
                        drivers = ['Observability: action log unavailable']
                    _deliver_offline_response(out, detail, log_rule='local.show_actions', drivers=drivers, confidence=0.8, meta={'n': n})
                    continue
                # Show last errors only
                if re.search(r"\b(show|list)\s+last\s+(errors|failures)\b", heard, flags=re.I):
                    try:
                        from app.logging.experience import DEFAULT_LOG_PATH as _XP_PATH
                        log_path = getattr(_LOGGER, 'path', _XP_PATH)
                        import json as _json
                        with open(log_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[-200:]
                        errs = []
                        for ln in reversed(lines):
                            try:
                                rec = _json.loads(ln)
                                if rec.get('outcome_success') is False:
                                    tool = ''
                                    steps = (rec.get('action_taken') or {}).get('steps') or []
                                    if steps:
                                        tool = (steps[-1] or {}).get('tool') or ''
                                    errs.append(f"• {tool or (rec.get('action_taken') or {}).get('routed','')} — {rec.get('error') or ''}")
                                    if len(errs) >= 5:
                                        break
                            except Exception:
                                continue
                        out = "Last errors:\n" + ("\n".join(errs) or "(no recent errors)")
                    except Exception as e:
                        out = f"I couldn't read the action log: {e}"
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.show_errors"}, True, None, False)
                    continue
                # Model identity queries → standard answer (avoid provider leakage)
                if re.search(r"\b(what|which)\s+(language\s+)?model\s+do\s+you\s+use\b|\bwhich\s+model\b|\bwhat\s+model\b", heard, flags=re.I):
                    out = ("I'm Nerion. I orchestrate the hosted models you've configured—like OpenAI or Anthropic—"
                           "to answer your requests while keeping control on your device.")
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.model_identity"}, True, None, False)
                    continue
                # Dev mode toggles (OFF first to avoid false positives)
                if re.search(r"\b(exit|leave|turn\s+off|disable)\s+(dev(eloper)?\s+)?mode\b", heard, flags=re.I):
                    try:
                        setattr(STATE, '_dev_mode', False)
                    except Exception:
                        pass
                    out = "Developer mode OFF."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_off"}, True, None, False)
                    continue
                if re.search(r"\b(switch|enter|go|enable|turn\s+on|activate)\s+(to\s+)?dev(eloper)?\s*mode\b", heard, flags=re.I):
                    try:
                        setattr(STATE, '_dev_mode', True)
                    except Exception:
                        pass
                    try:
                        from .capabilities import summarize_dev_options as _dev_opts
                        out = "Developer mode ON. " + _dev_opts()
                    except Exception:
                        out = "Developer mode ON. You can ask for details about self-code, self-improve, memory, web-research, voice-text, security, planner, tools-plugins."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_on"}, True, None, False)
                    continue
                if re.search(r"\b(show|list)\s+(dev(eloper)?\s+)?options\b", heard, flags=re.I):
                    try:
                        from .capabilities import summarize_dev_options as _dev_opts
                        out = _dev_opts()
                    except Exception:
                        out = "You can ask for developer details about self-code, self-improve, memory, web-research, voice-text, security, planner, tools-plugins."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_options"}, True, None, False)
                    continue
                # Developer detail for a specific capability
                m_dev = re.search(r"\b(details?\s+(for|about)\s+|show\s+)(self[- ]?code|self[- ]?improve|self[- ]?learn|self\s+learning|learning|memory|web(?:[- ]?research)?|voice(?:[- ]?text)?|security|planner|tools(?:[- ]?plugins)?|plugins)\b", heard, flags=re.I)
                if m_dev:
                    topic = m_dev.group(3)
                    try:
                        from .capabilities import summarize_capability_detail as _cap_detail
                        out = _cap_detail(topic)
                    except Exception:
                        out = f"Developer details unavailable for {topic}."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_detail","topic":topic}, True, None, False)
                    continue
                # Identity / name queries (who are you, your name, how to call you)
                if re.search(r"\b(who|what)\s+are\s+you\b", heard, flags=re.I) or \
                   re.search(r"\bwhat\s+is\s+your\s+name\b", heard, flags=re.I) or \
                   re.search(r"\b(how|what)\s+can\s+i\s+call\s+you\b", heard, flags=re.I) or \
                   re.search(r"\bdo\s+you\s+have\s+a\s+name\b", heard, flags=re.I):
                    out = ("I'm Nerion — your local, privacy‑first assistant running on this device. "
                           "You can call me Nerion.")
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.identity"}, True, None, False)
                    continue
                m_name = re.search(r"\byour\s+name\s+is\s+([a-z][a-z .'_-]{1,40})\b", heard, flags=re.I)
                if m_name:
                    out = "My name is Nerion."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.identity_confirm","heard_name": m_name.group(1).strip()}, True, None, False)
                    continue
                # Capabilities / what can you do
                if re.search(r"\b(what\s+can\s+you\s+do|what\s+are\s+your\s+capabilities|what\s+do\s+you\s+do|what\s+knowledge\s+do\s+you\s+have)\b", heard, flags=re.I):
                    try:
                        from .capabilities import summarize_capabilities as _cap_sum
                        with _Busy("Thinking…", start_delay_s=2.0):
                            out = _cap_sum("brief")
                    except Exception:
                        out = ("I’m Nerion. I coordinate the hosted models you configure to plan tasks, safely self‑code this repo, "
                               "remember preferences, and research the web with your permission.")
                    # Offer dev mode for more specifics
                    if not bool(getattr(STATE, '_dev_mode', False)):
                        out = (out.rstrip('.') +
                               " If you want more specific developer details and options, say 'switch to dev mode'.")
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.capabilities"}, True, None, False)
                    continue
                # Dev mode options and details
                if re.search(r"\b(show|list)\s+(dev(eloper)?\s+)?options\b", heard, flags=re.I):
                    try:
                        from .capabilities import summarize_dev_options as _dev_opts
                        out = _dev_opts()
                    except Exception:
                        out = "You can ask for developer details about self-code, self-improve, memory, web-research, voice-text, security, planner, tools-plugins."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_options"}, True, None, False)
                    continue
                m_dev = re.search(
                    r"\b(?:(?:details?\s+(?:for|about|on)\s+)|(?:show\s+(?:me\s+)?(?:the\s+)?))(?P<topic>self[- ]?code|self[- ]?improve|self[- ]?learn|self\s+learning|learning|memory|web(?:[- ]?research)?|voice(?:[- ]?text)?|security|planner|tools?(?:[- ]?plugins?)?|plugins)(?:\s+details?)?\b",
                    heard,
                    flags=re.I,
                )
                if m_dev:
                    topic = m_dev.group('topic')
                    try:
                        from .capabilities import summarize_capability_detail as _cap_detail
                        out = _cap_detail(topic)
                    except Exception:
                        out = f"Developer details unavailable for {topic}."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_detail","topic":topic}, True, None, False)
                    continue
                if re.search(r"\bshow\b", heard, flags=re.I) and re.search(r"\btool(s)?\b", heard, flags=re.I) and re.search(r"\bplugin(s)?\b", heard, flags=re.I) and re.search(r"\bdetail(s)?\b", heard, flags=re.I):
                    try:
                        from .capabilities import summarize_capability_detail as _cap_detail
                        out = _cap_detail("tools-plugins")
                    except Exception:
                        out = "Tools/Plugins details are unavailable right now."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.dev_detail","topic":"tools-plugins"}, True, None, False)
                    continue
                # Detailed version
                if re.search(r"\b(full|detailed)\s+capabilit(y|ies)|list\s+all\s+capabilities\b", heard, flags=re.I):
                    try:
                        from .capabilities import summarize_capabilities as _cap_sum
                        with _Busy("Gathering details…", start_delay_s=2.0):
                            out = _cap_sum("detailed")
                    except Exception:
                        out = "I’m Nerion. Detailed capabilities overview is unavailable right now."
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.capabilities_detailed"}, True, None, False)
                    continue
                # Memory recall
                if re.search(r"\b(what do you remember about|recall your memory on|summarize your memory of)\b", heard, flags=re.I):
                    out = _off_recall(heard)
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.recall_memory"}, True, None, False)
                    continue
                # Memory add (best-effort): "remember that ..." or "save this to long-term memory ..."
                m_rx = re.search(r"\b(remember that|save this to long[- ]?term memory)\b", heard, flags=re.I)
                if m_rx:
                    facts = mem.consider_storing(heard) or []
                    if facts:
                        msg = "Got it — I’ve saved that in my memory."
                    else:
                        msg = "I didn't find a clear fact to remember."
                    print('Nerion:', msg)
                    safe_speak(msg, watcher)
                    try:
                        STATE.append_turn('assistant', msg)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.pin_memory","facts": facts[:2]}, True, None, False)
                    continue
            except Exception:
                pass

            # --- Parent plan (capture) ---
            # Defer parent planning until after local routing (balanced policy).
            # Allow eager capture only when NERION_PARENT_EAGER is truthy.
            parent_decision_dict = None
            if _PARENT_DRIVER and (os.getenv('NERION_PARENT_EAGER', '').strip().lower() in {'1','true','yes','on'}):
                try:
                    dec = _PARENT_DRIVER.plan_and_route(user_query=heard, context_snippet=None)
                    parent_decision_dict = dec.dict()
                except Exception:
                    parent_decision_dict = {"intent": "error", "plan": []}
            # (normal execution moved to later)

            # --- Intelligent Switchboard: rules → site-query → web/LLM ---
            # 1) Rules: data-driven local intents (no network). If matched, handle and continue.
            try:
                rule = _detect_intent(heard, getattr(STATE, '_intent_rules', []))
            except Exception:
                rule = None
            if rule and isinstance(rule.name, str) and rule.name.startswith('local.'):
                out = None
                try:
                    out = _call_intent_handler(rule, heard)
                except Exception:
                    out = None
                if out:
                    print('Nerion:', out)
                    safe_speak(out, watcher)
                    try:
                        STATE.append_turn('assistant', out)
                    except Exception:
                        pass
                    _log_experience(heard, parent_decision_dict, {"routed":"local_rule","rule":rule.name}, True, None, False)
                    continue

            # 2) If rules indicated a web intent, mark it to prefer the web path below
            _force_web_intent = bool(rule and isinstance(rule.name, str) and rule.name.startswith('web.'))

            # Context routing
            relation = _relation_to_context(heard, STATE)
            if relation != 'off' and _needs_disambiguation(heard):
                relation = 'ambiguous'
            try:
                if not getattr(STATE, 'active', None):
                    STATE.open_new_conversation(topic=_auto_title(heard))
            except Exception:
                pass
            if relation == 'off':
                try:
                    STATE.archive_active()
                except Exception:
                    pass
                try:
                    STATE.open_new_conversation(topic=_auto_title(heard))
                except Exception:
                    pass
            elif relation == 'ambiguous':
                q = _make_clarifying_question(heard)
                try:
                    _ = STATE.append_turn('user', heard)
                except Exception:
                    pass
                print('Nerion:', q)
                safe_speak(q, watcher)
                continue
            try:
                _ = STATE.append_turn('user', heard)
            except Exception:
                pass

            # --- Network permission helper (session-scoped, via shared utility) ---
            def _ensure_network_for(task_type: str, url: Optional[str] = None) -> bool:
                try:
                    from .net_access import ensure_network_for as _ensure_net
                    return _ensure_net(
                        task_type,
                        lambda m: safe_speak(m, watcher),
                        listen_once,
                        url=url,
                        watcher=watcher,
                    )
                except Exception:
                    # Conservative default on error
                    return False

            # === PARENT EXECUTOR BINDINGS BEGIN (ANCHOR) ===
            # Metrics hook for Parent tool execution
            def _metric_hook(tool: str, ok: bool, dur: float, err: str | None):
                try:
                    _log_experience(
                        heard,
                        parent_decision_dict,
                        {"routed": "parent", "tool": tool, "duration_s": round(dur, 3)},
                        ok,
                        err,
                        True if (tool and tool.startswith("web")) else False,
                    )
                except Exception:
                    pass

            # Progress hook for step acks (promote to chat stream)
            def _progress_hook(i: int, total: int, desc: str):
                try:
                    msg = f"Step {i}/{total}: {desc}"
                    print('Nerion:', msg)
                    safe_speak(msg, watcher)
                    try:
                        STATE.append_turn('assistant', msg)
                    except Exception:
                        pass
                except Exception:
                    pass

            _executor = _build_parent_executor(
                ensure_network_for=_ensure_network_for,
                get_heard=lambda: heard,
                parse_task_slots=_parse_task_slots,
                metrics_hook=_metric_hook,
                progress_hook=_progress_hook,
                cancel_check=lambda: bool(getattr(STATE, '_cancel_requested', False)),
            )
            # === PARENT EXECUTOR BINDINGS END (ANCHOR) ===

            # === PARENT PLAN EXECUTION (ANCHOR) ===
            if _PARENT_DRIVER and isinstance(parent_decision_dict, dict):
                try:
                    # Preflight network once if Parent requires it
                    try:
                        if bool(parent_decision_dict.get("requires_network")):
                            _ensure_network_for("parent.plan", watcher, url=None)
                    except Exception:
                        pass
                    if parent_decision_dict.get("plan"):
                        from app.parent.schemas import ParentDecision as _PD
                        _dec_obj = _PD(**parent_decision_dict)
                        outcome = _executor.execute(_dec_obj, heard)
                        # Speak final text if provided
                        if outcome.get("final_text"):
                            txt = str(outcome["final_text"]).strip()
                            if txt:
                                print('Nerion:', txt)
                                safe_speak(txt, watcher)
                                try:
                                    STATE.append_turn('assistant', txt)
                                except Exception:
                                    pass
                        # Log the execution outcome
                        _log_experience(
                            heard,
                            parent_decision_dict,
                            outcome.get("action_taken") or {"routed":"parent_plan"},
                            bool(outcome.get("success", False)),
                            outcome.get("error"),
                            parent_decision_dict.get("requires_network"),
                        )
                        continue
                except Exception as e:
                    # If anything goes wrong, fall back to legacy routing
                    _log_experience(
                        heard,
                        parent_decision_dict,
                        {"routed":"parent_plan","phase":"execute"},
                        False,
                        str(e),
                        parent_decision_dict.get("requires_network") if isinstance(parent_decision_dict, dict) else None,
                    )
                    _dbg(f"Parent plan execution failed: {e!r}")

            # Site-query intent path (extracted)
            if _run_site_query(heard, STATE, watcher, parent_decision_dict):
                continue

            # Semantic fallback (only if no site-query intent handled)
            if not _force_web_intent:
                sem_name = None
                try:
                    sem_name = _sem_match(heard)
                except Exception:
                    sem_name = None
                if sem_name:
                    # If this maps to a known local rule, execute it
                    try:
                        rules = getattr(STATE, '_intent_rules', []) or []
                        rule_match = next((r for r in rules if r.name == sem_name), None)
                    except Exception:
                        rule_match = None
                    if rule_match and isinstance(rule_match.name, str) and rule_match.name.startswith('local.'):
                        out = None
                        try:
                            out = _call_intent_handler(rule_match, heard)
                        except Exception:
                            out = None
                        if out:
                            print('Nerion:', out)
                            safe_speak(out, watcher)
                            try:
                                STATE.append_turn('assistant', out)
                            except Exception:
                                pass
                            continue
                    # If it's a web intent, mark for web path
                    if isinstance(sem_name, str) and sem_name.startswith('web.'):
                        _force_web_intent = True

            # Open-web research path (only if explicitly indicated)
            # Gate by detected/semantic web intent to avoid prompting for unrelated queries
            if _force_web_intent:
                if _run_web_search(heard, STATE, watcher, parent_decision_dict):
                    continue

            # === Parent (after local routing) with short timeout =================
            if _PARENT_DRIVER:
                try:
                    hlow = (heard or '').lower()
                    call_parent = False
                    force_cmd = False
                    # Known safe system command: run health check → bypass planning to avoid stalls
                    if re.search(r"\b(health\s*(check|scan)?|system\s*health)\b", hlow) and \
                       re.search(r"\b(run|scan|check|diagnos(e|is|tic))\b", hlow):
                        ack = "Sure — running the full health scan…"
                        print('Nerion:', ack)
                        safe_speak(ack, watcher)
                        with _Busy("Running full health scan…", start_delay_s=0.2):
                            out = _hc_offline(None)
                        print('Nerion:', out)
                        safe_speak(out, watcher)
                        try:
                            STATE.append_turn('assistant', out)
                        except Exception:
                            pass
                        _log_experience(heard, {}, {"routed":"direct","tool":"run_healthcheck"}, True, None, False)
                        continue
                    # Direct-run diagnostics (offline)
                    if re.search(r"\b(run|show|do)\b", hlow) and re.search(r"\bdiagnostic(s)?\b", hlow):
                        ack = "Sure — running diagnostics…"
                        print('Nerion:', ack)
                        safe_speak(ack, watcher)
                        with _Busy("Running diagnostics…", start_delay_s=0.2):
                            out = _diag_offline(None)
                        print('Nerion:', out)
                        safe_speak(out, watcher)
                        try:
                            STATE.append_turn('assistant', out)
                        except Exception:
                            pass
                        _log_experience(heard, {}, {"routed":"direct","tool":"run_diagnostics"}, True, None, False)
                        continue
                    # Direct-run smoke tests (offline)
                    if re.search(r"\b(run|execute|start)\b", hlow) and re.search(r"\b(smoke\s+tests?|pytest\s+smoke)\b", hlow):
                        try:
                            from ops.security.safe_subprocess import safe_run as _safe
                        except Exception:
                            _safe = None
                        ack = "Sure — running smoke tests…"
                        print('Nerion:', ack)
                        safe_speak(ack, watcher)
                        if _safe is None:
                            out = "Smoke: runner unavailable."
                        else:
                            with _Busy("Running smoke tests…", start_delay_s=0.2):
                                try:
                                    r = _safe(["pytest", "-q", "-k", "smoke", "--maxfail=1"], capture_output=True, timeout=180, check=False)
                                    ok = r.returncode == 0
                                    out_txt = (r.stdout or b"").decode(errors='ignore')
                                    last = next((ln for ln in reversed(out_txt.splitlines()) if ln.strip()), "")
                                    out = f"Smoke: {'OK' if ok else 'FAIL'} — {last[:200]}"
                                except Exception as e:
                                    out = f"Smoke: error: {e}"
                        print('Nerion:', out)
                        safe_speak(out, watcher)
                        try:
                            STATE.append_turn('assistant', out)
                        except Exception:
                            pass
                        _log_experience(heard, {}, {"routed":"direct","tool":"run_pytest_smoke"}, True, None, False)
                        continue
                    if any(k in hlow for k in ['http://', 'https://', ' www.', ' web ', 'internet', 'online', 'search', 'research', 'latest news']):
                        call_parent = True
                    if re.search(r'\b(rename|refactor|extract)\b', hlow):
                        call_parent = True
                    # System/diagnostic and dev verbs → allow planner to pick tools
                    if re.search(r'\b(run|scan|check|diagnose|diagnostic|health\s*(check|scan)?|update|test|benchmark|audit|format|lint|apply)\b', hlow):
                        call_parent = True
                        force_cmd = True
                    if call_parent:
                        timeout_s = float(os.getenv('NERION_PARENT_TIMEOUT_S', '3.0'))
                        if force_cmd:
                            # Allow a bit more time for command verbs
                            timeout_s = float(os.getenv('NERION_PARENT_TIMEOUT_S_CMD', '5.0'))
                        with _futures.ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(_PARENT_DRIVER.plan_and_route, user_query=heard, context_snippet=None)
                            with _Busy("Thinking through a plan…", start_delay_s=2.0):
                                try:
                                    dec = fut.result(timeout=timeout_s)
                                    parent_decision_dict = dec.dict()
                                except Exception as e:
                                    try:
                                        fut.cancel()
                                    except Exception:
                                        pass
                                    parent_decision_dict = {"intent": "timeout", "plan": [], "notes": str(e)}

                        if isinstance(parent_decision_dict, dict) and parent_decision_dict.get('plan'):
                            # UX cue: if plan includes healthcheck, acknowledge before execution
                            try:
                                if any((s or {}).get('tool') == 'run_healthcheck' for s in parent_decision_dict.get('plan', [])):
                                    ack = "Sure — running the full health scan…"
                                    print('Nerion:', ack)
                                    safe_speak(ack, watcher)
                            except Exception:
                                pass
                            # Preflight network for plans marked as requiring network
                            try:
                                if bool(parent_decision_dict.get('requires_network')):
                                    _ensure_network_for('parent.plan', watcher, url=None)
                            except Exception:
                                pass

                            def _metric_hook(tool: str, ok: bool, dur: float, err: str | None):
                                try:
                                    _log_experience(
                                        heard,
                                        parent_decision_dict,
                                        {"routed": "parent", "tool": tool, "duration_s": round(dur, 3)},
                                        ok,
                                        err,
                                        True if (tool and tool.startswith('web')) else False,
                                    )
                                except Exception:
                                    pass

                            _executor = _build_parent_executor(
                                ensure_network_for=_ensure_network_for,
                                get_heard=lambda: heard,
                                parse_task_slots=_parse_task_slots,
                                metrics_hook=_metric_hook,
                            )
                            try:
                                from app.parent.schemas import ParentDecision as _PD
                                _dec_obj = _PD(**parent_decision_dict)
                                outcome = _executor.execute(_dec_obj, heard)
                                if outcome.get('final_text'):
                                    txt = str(outcome['final_text']).strip()
                                    if txt:
                                        print('Nerion:', txt)
                                        safe_speak(txt, watcher)
                                        try:
                                            STATE.append_turn('assistant', txt)
                                        except Exception:
                                            pass
                                _log_experience(
                                    heard,
                                    parent_decision_dict,
                                    outcome.get('action_taken') or {"routed": "parent_plan"},
                                    bool(outcome.get('success', False)),
                                    outcome.get('error'),
                                    parent_decision_dict.get('requires_network') if isinstance(parent_decision_dict, dict) else None,
                                )
                                continue
                            except Exception as e:
                                _log_experience(
                                    heard,
                                    parent_decision_dict,
                                    {"routed": "parent_plan", "phase": "execute"},
                                    False,
                                    str(e),
                                    parent_decision_dict.get('requires_network') if isinstance(parent_decision_dict, dict) else None,
                                )
                                _dbg(f"Parent plan execution failed: {e!r}")
                        else:
                            # No plan produced; for command-like utterances, try direct safe execution for known tasks.
                            if force_cmd:
                                if re.search(r"\bhealth\b", hlow):
                                    ack = "Sure — running the full health scan…"
                                    print('Nerion:', ack)
                                    safe_speak(ack, watcher)
                                    out = _hc_offline(None)
                                    print('Nerion:', out)
                                    safe_speak(out, watcher)
                                    try:
                                        STATE.append_turn('assistant', out)
                                    except Exception:
                                        pass
                                    _log_experience(heard, parent_decision_dict or {}, {"routed":"direct","tool":"run_healthcheck"}, True, None, False)
                                    continue
                                # Otherwise ask a crisp clarify instead of freeform chat
                                msg = "I can run system tools like the health check. Should I proceed with a full health scan?"
                                print('Nerion:', msg)
                                safe_speak(msg, watcher)
                                try:
                                    STATE.append_turn('assistant', msg)
                                except Exception:
                                    pass
                                _log_experience(heard, parent_decision_dict or {}, {"routed":"parent_plan","phase":"no_plan"}, False, None, False)
                                continue
                except Exception:
                    pass

            # General chat fallback
            new_facts = mem.consider_storing(heard)
            if new_facts:
                print('[LEARNED]')
                for f in new_facts:
                    print(' • ' + f)
                try:
                    mem.record_reference(new_facts[-1])
                except Exception:
                    pass
                if _SESSION_CACHE is not None:
                    _SESSION_CACHE.set_last_ref(new_facts[-1])
                if router:
                    router.refresh_memory()
            relevant = mem.find_relevant(heard, k=5)
            mem_block = ''
            if _USE_MEM_IN_PROMPT and relevant:
                try:
                    limit = max(1, int(os.getenv('NERION_MEMORY_PROMPT_K', '2')))
                except Exception:
                    limit = 2
                try:
                    min_conf = float(os.getenv('NERION_MEMORY_PROMPT_MIN_CONF', '0.6'))
                except Exception:
                    min_conf = 0.6
                inject = []
                seen_ids = set()
                for item in relevant:
                    fact_text = item.get('fact') if isinstance(item, dict) else item
                    if not fact_text:
                        continue
                    conf = float(item.get('confidence', 0.7)) if isinstance(item, dict) else 0.7
                    if conf < min_conf:
                        continue
                    ident = item.get('id') if isinstance(item, dict) else None
                    if ident and ident in seen_ids:
                        continue
                    inject.append({'id': ident or '', 'text': sanitize_for_prompt(fact_text)})
                    if ident:
                        seen_ids.add(ident)
                    if len(inject) >= limit:
                        break
                if inject:
                    mem_block = _render_memory_block(inject)
            if is_command(heard, { 'exit','quit','shutdown','goodbye','good bye','bye','terminate','stop listening','sleep','go to sleep','talk later','see you','hold on','stop','pause','mute','be quiet','silence','stop speaking','mute yourself','unmute','speak','sound on','voice on' }):
                print('[CTRL] final-guard (no LLM)')
                continue

            # Small talk and diagnostics shortcuts (avoid LLM)
            try:
                # Upgrade details/status queries (offline answer)
                if re.search(r"\b(what|which|did you|you)\s+(?:upgrade|update|change)\b", heard, flags=re.I):
                    note = "I completed a code upgrade recently."
                    try:
                        import json as _json
                        p = os.path.join('out','policies','upgrade_state.json')
                        if os.path.exists(p):
                            st = _json.loads(open(p,'r',encoding='utf-8').read())
                            last = int(st.get('last_upgrade_ts') or 0)
                            sched = int(st.get('scheduled_ts') or 0)
                            if last:
                                note = "I completed a recent self-upgrade: non-blocking upgrade flow, fewer prompts, concise replies by default, opt-in memory, and quieter TTS logs."
                            elif sched:
                                note = "An upgrade is scheduled for tonight."
                    except Exception:
                        pass
                    _deliver_offline_response(note, 'Shared upgrade status', log_rule='local.upgrade_status', drivers=['Upgrade knowledge'], confidence=0.82)
                    continue
                if re.search(r"\b(details|more details|explain|what changed)\b", heard, flags=re.I) and re.search(r"\b(upgrade|update|change)\b", heard, flags=re.I):
                    details = (
                        "Here’s what changed: 1) upgrade prompt no longer blocks PTT and has a cooldown, 2) concise replies by default, 3) memory in prompts is opt-in, 4) small-talk and upgrade-status handled offline, 5) TTS console noise reduced."
                    )
                    print('Nerion:', details)
                    safe_speak(details, watcher)
                    try:
                        STATE.append_turn('assistant', details)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.upgrade_details"}, True, None, False)
                    continue
                if re.search(r"\bwhat\s+did\s+you\s+(upgrade|update)\b", heard, flags=re.I):
                    # Best-effort summary from upgrade state
                    note = "I applied a recent self-upgrade."
                    try:
                        import json as _json
                        p = os.path.join('out','policies','upgrade_state.json')
                        if os.path.exists(p):
                            st = _json.loads(open(p,'r',encoding='utf-8').read())
                            last = int(st.get('last_upgrade_ts') or 0)
                            sched = int(st.get('scheduled_ts') or 0)
                            if last:
                                note = "I completed a code upgrade recently."
                            elif sched:
                                note = "An upgrade is scheduled for tonight."
                    except Exception:
                        pass
                    print('Nerion:', note)
                    safe_speak(note, watcher)
                    try:
                        STATE.append_turn('assistant', note)
                    except Exception:
                        pass
                    _log_experience(heard, None, {"routed":"offline_fast","rule":"local.upgrade_status"}, True, None, False)
                    continue
            except Exception:
                pass

            # Run provider-backed chat model to produce a response
            is_stub_model = False
            model_label = 'unconfigured'
            response_failed = False
            try:
                temp = STATE.voice.current_temperature(0.7)
                chat_chain = build_chain_with_temp(temp)
                is_stub_model = bool(getattr(chat_chain, 'IS_STUB', False))
                model_label = getattr(chat_chain, '_nerion_model_name', 'unconfigured')
                art_txt = _load_last_artifact_from_state(STATE)
                if relation == 'on' and getattr(STATE, 'active', None):
                    hist = list(getattr(STATE.active, 'chat_history', []) or [])
                    prompt = _build_followup_prompt(hist, heard, art_txt)
                else:
                    artifact_block = ('Relevant artifact (truncated):\n' + art_txt + '\n') if art_txt else ''
                    style = "Be clear and conversational. Offer 2–3 sentences by default and add detail when it genuinely helps."
                    concise_hint = "\nPlease answer in one sentence." if bool(getattr(STATE, '_concise_mode', False)) else ""
                    system = (
                        "You are Nerion, a privacy-first assistant coordinating hosted LLM APIs using the user's credentials. "
                        "Deliver helpful, natural answers while respecting privacy and never inventing user preferences. "
                        "Do NOT include disclaimers like 'as an AI' or training cutoffs; answer directly. "
                        "Never output <think> tags or chain-of-thought; provide only the final answer. "
                        "If essential details are missing and you cannot proceed safely, ask one concise clarifying question; otherwise give your best helpful answer."
                    )
                    prompt = f"{system}\n{mem_block}\n{artifact_block}User said: {heard}\n{style}{concise_hint}"
                detail_label = f'Composing response via {model_label}'
                _complete_analysis('Planning response')
                _start_response(detail_label, status='active')
                with _Busy("Thinking…", start_delay_s=1.0):
                    response = _predict_with_timeout(chat_chain, prompt, llm_timeout)
            except Exception as e:
                _dbg(f"LLM fallback failed: {e!r}")
                response = f'Sorry, I had an issue thinking about that: {e}'
                response_failed = True
                confidence_score = 0.25
                confidence_drivers = [f'Response generation error: {e}']
            response = _strip_think_blocks(response).strip()
            print('Nerion:', response)
            safe_speak(response, watcher)
            try:
                STATE.append_turn('assistant', response)
            except Exception:
                pass
            if response_failed:
                _finish_response('Response generation failed', status='failed')
            else:
                _finish_response('Response delivered', status='complete')
            llm_meta = getattr(chat_chain, 'last_response', None)
            if confidence_score is None:
                if is_stub_model:
                    confidence_score = 0.45
                    confidence_drivers = ['Fallback response (stub model)']
                else:
                    confidence_score = 0.78
                    if llm_meta is not None:
                        latency_ms = int(llm_meta.latency_s * 1000)
                        confidence_drivers = [
                            f"LLM: {llm_meta.provider}:{llm_meta.model}",
                            f"Latency: {latency_ms} ms",
                        ]
                    else:
                        confidence_drivers = [f'LLM: {model_label}']
            _emit_conf(confidence_score, confidence_drivers)
            if router:
                setattr(router.state, 'last_llm_details', {
                    'provider': getattr(llm_meta, 'provider', model_label.split(':')[0] if ':' in model_label else model_label),
                    'model': getattr(llm_meta, 'model', model_label.split(':')[-1]),
                    'latency_ms': int(llm_meta.latency_s * 1000) if llm_meta else None,
                })
                router.emit_metrics()
            _session_record_assistant(response)
            if router:
                router.on_assistant_turn()
            try:
                _log_experience(heard, parent_decision_dict, {"routed":"llm_fallback"}, True, None, False)
            except Exception:
                pass
            # If the user answered the upgrade prompt here (PTT/text), handle it directly.
            try:
                if re.search(r"\b(upgrade now|remind me later|tonight|tonite|do it|proceed)\b", heard, flags=re.I):
                    if _upgrade_handle_choice(heard, watcher):
                        continue
            except Exception:
                pass
            # Mid-session upgrade offer (light-weight check)
            try:
                maybe_offer_upgrade(watcher, ptt_mode=bool(getattr(getattr(STATE, 'voice', None), 'ptt', True)))
            except Exception:
                pass
            # Non-blocking shadow replay for self-upgrade safety evaluation
            try:
                if _shadow_should():
                    _shadow_schedule()
            except Exception:
                pass
    except KeyboardInterrupt:
        print('\n🛑 Nerion interrupted and stopped.')
    finally:
        # Shutdown plumbing returned to the runner via a small hook
        try:
            if chat:
                chat.stop()
        except Exception as e:
            _dbg(f"chat.stop() failed: {e!r}")
        _session_shutdown()
        # Ensure TTS and speech pipelines are stopped cleanly
        try:
            cancel_speech()
        except Exception as e:
            _dbg(f"cancel_speech() failed: {e!r}")
        try:
            from app.chat.tts_router import shutdown as _tts_shutdown
            _tts_shutdown()
        except Exception as e:
            _dbg(f"tts_router.shutdown() failed: {e!r}")
        try:
            if watcher and getattr(STATE.voice, 'barge_in', False):
                from voice.stt.recognizer import stop_barge_in_monitor
                stop_barge_in_monitor()
        except Exception as e:
            _dbg(f"stop_barge_in_monitor() failed: {e!r}")
        try:
            _save_session_state()
        except Exception as e:
            _dbg(f"_save_session_state() failed: {e!r}")
        print('🛑 Nerion has shut down.')
