"""Electron UI command router for Nerion backend.

Bridges JSON commands from the Electron shell to internal Nerion helpers and
emits structured events back to the renderer.
"""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as dt

from selfcoder.selfaudit import generate_improvement_plan as _generate_upgrade_plan

from ops.security.net_gate import NetworkGate

from .memory_session import SessionCache
from .memory_bridge import LongTermMemory
from .state import ChatState
from .voice_io import safe_speak, set_device_index, tts_set_params as _tts_set_params
from . import ipc_electron as _ipc
from .offline_tools import run_healthcheck as _run_healthcheck, run_diagnostics as _run_diagnostics
from ..config import load_config
from selfcoder.learning.continuous import load_prefs as _learning_load_prefs
from app.learning.upgrade_agent import handle_choice as _upgrade_handle_choice, readiness_report


def _parse_timestamp(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    with suppress(Exception):
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        with suppress(Exception):
            if text.isdigit():
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
        with suppress(Exception):
            dt_obj = datetime.fromisoformat(text.replace('Z', '+00:00'))
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj
        with suppress(Exception):
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
    return None


def _humanize_timesince(value) -> str:
    dt_obj = _parse_timestamp(value)
    if not dt_obj:
        return '—'
    now = datetime.now(timezone.utc)
    delta = now - dt_obj
    seconds = int(delta.total_seconds())
    if seconds < 10:
        return 'just now'
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    weeks = days // 7
    if weeks < 10:
        return f"{weeks}w ago"
    return dt_obj.strftime('%Y-%m-%d')


def _default_filters() -> list[str]:
    return ['session', 'long', 'pinned', 'all']


class ElectronCommandRouter:
    """Process Electron UI commands and keep renderer state in sync."""

    def __init__(
        self,
        state: ChatState,
        session_cache: Optional[SessionCache],
        mem: Optional[LongTermMemory],
        *,
        ptt_press_cb=None,
        ptt_release_cb=None,
    ):
        self.state = state
        self.session_cache = session_cache
        self.mem = mem
        self.memory_filter = 'session'
        self.memory_index: Dict[str, Dict[str, Any]] = {}
        self.learning_events: Dict[str, Dict[str, Any]] = {}
        self.learning_selected: Optional[str] = None
        self.watcher = None
        self._ptt_press_cb = ptt_press_cb
        self._ptt_release_cb = ptt_release_cb
        self._upgrade_plan_cache: Dict[str, Dict[str, Any]] = {}
        self._thought_seq = 0
        self._last_metrics_signature: Optional[str] = None

    # --- plumbing -------------------------------------------------------------
    def set_watcher(self, watcher) -> None:
        self.watcher = watcher

    def set_ptt_callbacks(self, press_cb, release_cb) -> None:
        self._ptt_press_cb = press_cb
        self._ptt_release_cb = release_cb

    def _emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not _ipc.enabled():
            return
        try:
            _ipc.emit(event_type, payload or {})
        except Exception:
            pass

    def _state_payload(self) -> Dict[str, Any]:
        voice = getattr(self.state, 'voice', None)
        return {
            'interaction_mode': 'talk' if getattr(voice, 'ptt', False) else 'chat',
            'speech_enabled': bool(getattr(voice, 'enabled', True)),
            'muted': bool(getattr(self.state, 'muted', False)),
        }

    def emit_phase(self, phase: str, *, reset_thoughts: bool = False) -> None:
        payload = self._state_payload()
        payload['phase'] = phase
        if reset_thoughts:
            payload['reset_thoughts'] = True
        payload['ts'] = int(time.time() * 1000)
        self._emit('state', payload)

    def reset_thoughts(self) -> None:
        self._thought_seq = 0
        self.emit_phase('thinking', reset_thoughts=True)

    def thought_step(self, title: str, detail: Optional[str] = None, *, status: str = 'pending') -> Optional[str]:
        if not title:
            return None
        self._thought_seq += 1
        step_id = f'step-{self._thought_seq}'
        payload = {
            'id': step_id,
            'title': title,
            'detail': detail or '',
            'status': status,
            'ts': int(time.time() * 1000),
        }
        self._emit('thought_step', payload)
        return step_id

    def update_thought(self, step_id: Optional[str], *, title: Optional[str] = None, detail: Optional[str] = None, status: Optional[str] = None) -> None:
        if not step_id:
            return
        payload: Dict[str, Any] = {'id': step_id, 'ts': int(time.time() * 1000)}
        if title is not None:
            payload['title'] = title
        if detail is not None:
            payload['detail'] = detail
        if status is not None:
            payload['status'] = status
        self._emit('thought_step', payload)

    def emit_confidence(self, score: float, drivers: Optional[list[str]] = None) -> None:
        if score is None:
            return
        try:
            val = float(score)
        except Exception:
            return
        clamped = max(0.0, min(1.0, val))
        payload = {
            'score': clamped,
            'value': clamped,
        }
        if drivers:
            payload['drivers'] = [str(d) for d in drivers if d]
        self._emit('confidence', payload)

    def emit_metrics(self) -> None:
        metrics: list[dict[str, str]] = []
        voice = getattr(self.state, 'voice', None)
        mode = 'Voice (PTT)' if getattr(voice, 'ptt', False) else 'Text chat'
        metrics.append({'label': 'Interaction', 'value': mode})
        speech_enabled = bool(getattr(voice, 'enabled', True))
        muted = bool(getattr(self.state, 'muted', False))
        if speech_enabled and not muted:
            speech_val = 'Speaking enabled'
        elif muted:
            speech_val = 'Muted'
        else:
            speech_val = 'Speech off'
        metrics.append({'label': 'Speech', 'value': speech_val})

        session_count = 0
        long_count = 0
        if self.memory_index:
            for item in self.memory_index.values():
                src = item.get('source')
                if src == 'session':
                    session_count += 1
                elif src == 'long':
                    long_count += 1
        elif self.session_cache is not None:
            # fallback: count from session cache if index not yet populated
            with suppress(Exception):
                session_count = len(self.session_cache.state.get('short_facts', []))
        if self.mem is not None and long_count == 0:
            with suppress(Exception):
                long_count = len([m for m in self.mem.list_memories() if m.get('scope', 'short') == 'long'])
        metrics.append({'label': 'Session facts', 'value': str(session_count)})
        metrics.append({'label': 'Long-term facts', 'value': str(long_count)})

        turns = 0
        if self.session_cache is not None:
            with suppress(Exception):
                turns = len(self.session_cache.state.get('turns', []))
        subtitle = f"{turns} turns · {session_count} session · {long_count} long-term"

        signature = f"{mode}|{speech_val}|{session_count}|{long_count}|{turns}"
        if signature == self._last_metrics_signature:
            return
        self._last_metrics_signature = signature
        self._emit('metrics', {'items': metrics, 'subtitle': subtitle})

    def _emit_error(self, code: str, message: str) -> None:
        self._emit('error', {'code': code, 'message': message})

    def _settings_backend_options(self) -> List[Dict[str, str]]:
        options: List[Dict[str, str]] = [
            {'value': 'default', 'label': 'Default'},
            {'value': 'pyttsx3', 'label': 'pyttsx3'},
        ]
        if sys.platform == 'darwin':
            options.insert(1, {'value': 'say', 'label': 'macOS Say'})
        options.append({'value': 'piper', 'label': 'Piper'})
        return options

    def _settings_device_options(self) -> List[Dict[str, str]]:
        devices: List[Dict[str, str]] = [{'value': 'system_default', 'label': 'System Default'}]
        hint = getattr(getattr(self.state, 'voice', None), 'device_hint', None)
        if hint:
            hint_str = str(hint)
            if not any(d['value'] == hint_str for d in devices):
                devices.append({'value': hint_str, 'label': hint_str})
        return devices

    def _build_settings_options(self) -> Dict[str, Any]:
        return {
            'voiceBackends': self._settings_backend_options(),
            'devices': self._settings_device_options(),
        }

    def _build_settings_values(self) -> Dict[str, Any]:
        try:
            cfg = load_config()
        except Exception:
            cfg = {}
        voice_cfg: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            maybe_voice = cfg.get('voice') or {}
            if isinstance(maybe_voice, dict):
                voice_cfg = maybe_voice
        backend = str(
            voice_cfg.get('tts_backend')
            or voice_cfg.get('backend')
            or getattr(getattr(self.state, 'voice', None), 'backend', '')
            or ''
        ).strip() or 'default'
        rate = voice_cfg.get('rate')
        if rate is None:
            rate = getattr(getattr(self.state, 'voice', None), 'rate', None)
        if rate is None:
            rate = getattr(getattr(self.state, 'voice', None), 'tts_rate', None)
        if rate is None:
            rate = 190
        try:
            rate_int = int(rate)
        except Exception:
            rate_int = 190
        device = voice_cfg.get('device') or getattr(getattr(self.state, 'voice', None), 'device_hint', None) or 'system_default'
        autospeak_pref = voice_cfg.get('always_speak')
        if autospeak_pref is None:
            autospeak_pref = getattr(getattr(self.state, 'voice', None), 'autospeak', None)
        if autospeak_pref is None:
            autospeak_pref = bool(getattr(getattr(self.state, 'voice', None), 'enabled', True))
        offline = False
        if isinstance(cfg, dict):
            allow_net = cfg.get('allow_network_access')
            if allow_net is not None:
                offline = not bool(allow_net)
        env_net = os.getenv('NERION_ALLOW_NETWORK')
        if env_net is not None:
            offline = env_net.strip().lower() in {'0', 'false', 'no'}
        hotkey_default = 'Cmd+Shift+Space' if sys.platform == 'darwin' else 'Ctrl+Shift+Space'
        hotkey = getattr(getattr(self.state, 'voice', None), 'hotkey', hotkey_default)
        return {
            'voiceBackend': backend,
            'voiceRate': str(rate_int),
            'device': str(device),
            'hotkey': hotkey,
            'offline': bool(offline),
            'autospeak': bool(autospeak_pref),
        }

    def _emit_settings_snapshot(self, *, include_options: bool = True) -> None:
        if not _ipc.enabled():
            return
        if include_options:
            self._emit('settings_options', self._build_settings_options())
        self._emit('settings_values', self._build_settings_values())

    def emit_settings_bootstrap(self) -> None:
        self._emit_settings_snapshot(include_options=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

    def _hash_id(self, prefix: str, value: str) -> str:
        digest = hashlib.blake2s(value.encode('utf-8', 'ignore'), digest_size=8).hexdigest()
        return f"{prefix}-{digest}"

    # --- memory --------------------------------------------------------------
    def refresh_memory(self, *, active_filter: Optional[str] = None, open_drawer: bool = False) -> None:
        if active_filter:
            self.memory_filter = active_filter
        session_items, drawer_items = self._build_memory_payloads()
        self._emit('memory_session', {'items': session_items})
        drawer_payload = {
            'facts': self._filter_drawer(drawer_items),
            'filters': _default_filters(),
            'active_filter': self.memory_filter,
        }
        if open_drawer:
            drawer_payload['open'] = True
        self._emit('memory_drawer', drawer_payload)
        self.emit_metrics()

    def _build_memory_payloads(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        index: Dict[str, Dict[str, Any]] = {}
        session_items: list[dict[str, Any]] = []
        drawer_items: list[dict[str, Any]] = []

        if self.session_cache is not None:
            facts = list(self.session_cache.state.get('short_facts', []))
            for fact in facts:
                text = fact.get('fact') or ''
                if not text:
                    continue
                fid = fact.get('id') or self._hash_id('sess', text)
                fact['id'] = fid
                item = {
                    'id': fid,
                    'fact': text,
                    'scope': 'session',
                    'confidence': min(1.0, float(fact.get('score', 0.7)) / 5.0),
                    'last_used': _humanize_timesince(fact.get('ts')),
                    'pinned': bool(fact.get('pinned', False)),
                    'tags': fact.get('tags') or [],
                }
                session_items.append(item)
                drawer_item = dict(item)
                drawer_item['source'] = 'session'
                drawer_items.append(drawer_item)
                index[fid] = {
                    'source': 'session',
                    'fact': text,
                    'tags': fact.get('tags') or [],
                    'ref': fact,
                }

        if self.mem is not None:
            memories: list[dict[str, Any]] = []
            with suppress(Exception):
                memories = self.mem.list_memories()
            for entry in memories:
                text = entry.get('fact') or ''
                if not text:
                    continue
                fid = entry.get('id') or self._hash_id('mem', text)
                entry['id'] = fid
                scope = entry.get('scope', 'short')
                item = {
                    'id': fid,
                    'fact': text,
                    'scope': scope,
                    'confidence': float(entry.get('confidence', 0.7)),
                    'last_used': _humanize_timesince(entry.get('last_used_ts') or entry.get('timestamp')),
                    'pinned': scope == 'long',
                    'tags': entry.get('tags') or [],
                }
                drawer_item = dict(item)
                drawer_item['source'] = 'long'
                drawer_items.append(drawer_item)
                index[fid] = {
                    'source': 'long',
                    'fact': text,
                    'tags': entry.get('tags') or [],
                    'ref': entry,
                    'scope': scope,
                }

        def _sort_key(it: dict[str, Any]):
            dt_obj = _parse_timestamp(it.get('last_used'))
            return 0 if not dt_obj else -dt_obj.timestamp()

        drawer_items.sort(key=_sort_key)
        session_items.sort(key=_sort_key)
        self.memory_index = index
        return session_items, drawer_items

    def _filter_drawer(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filt = (self.memory_filter or 'session').lower()
        if filt == 'all':
            return items
        if filt == 'pinned':
            return [i for i in items if i.get('pinned')]
        if filt == 'expiring':
            out: list[dict[str, Any]] = []
            now = datetime.now(timezone.utc)
            for item in items:
                ref = self.memory_index.get(item.get('id'))
                if not ref or ref.get('source') != 'long':
                    continue
                ttl = ref['ref'].get('ttl_days')
                if ttl is None:
                    continue
                ts = _parse_timestamp(ref['ref'].get('timestamp'))
                if ts is None:
                    continue
                expires = ts + dt.timedelta(days=int(ttl))
                if expires <= now + dt.timedelta(days=2):
                    out.append(item)
            return out
        return [i for i in items if i.get('scope', '').lower() == filt]

    def handle_memory(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'refresh').lower()
        if action in {'drawer', 'refresh'}:
            filt = payload.get('filter')
            if filt:
                self.memory_filter = str(filt).lower()
            self.refresh_memory(active_filter=self.memory_filter, open_drawer=True)
            return
        fact_id = str(payload.get('fact_id') or '').strip()
        if not fact_id:
            return
        entry = self.memory_index.get(fact_id)
        if not entry:
            self._emit('error', {'code': 'memory', 'message': f'Item {fact_id} not found'})
            return
        if action == 'pin':
            self._memory_pin(entry)
        elif action == 'unpin':
            self._memory_unpin(entry)
        elif action == 'forget':
            self._memory_forget(entry)
        elif action == 'edit':
            new_value = str(payload.get('value') or '').strip()
            if new_value:
                self._memory_edit(entry, new_value)
        self.refresh_memory(active_filter=self.memory_filter)

    def _memory_pin(self, entry: Dict[str, Any]) -> None:
        text = entry.get('fact')
        if not text or self.mem is None:
            return
        tags = entry.get('tags') or []
        with suppress(Exception):
            self.mem.add_fact(text if text.endswith('.') else text + '.', tags=tags, scope='long', provenance='ui_pin', confidence=0.9)
        self._emit('memory_update', {'action': 'pin', 'fact': text})

    def _memory_unpin(self, entry: Dict[str, Any]) -> None:
        text = entry.get('fact')
        if not text or self.mem is None:
            return
        with suppress(Exception):
            self.mem.unpin_matching(text)
        self._emit('memory_update', {'action': 'unpin', 'fact': text})

    def _memory_forget(self, entry: Dict[str, Any]) -> None:
        text = entry.get('fact')
        if not text:
            return
        if entry.get('source') == 'session' and self.session_cache is not None:
            facts = self.session_cache.state.get('short_facts', [])
            self.session_cache.state['short_facts'] = [f for f in facts if f.get('fact') != text]
            with suppress(Exception):
                self.session_cache.save()
        if self.mem is not None:
            with suppress(Exception):
                self.mem.forget_smart(text, last_hint=None)
        self._emit('memory_update', {'action': 'forget', 'fact': text})

    def _memory_edit(self, entry: Dict[str, Any], new_value: str) -> None:
        source = entry.get('source')
        if source == 'session' and self.session_cache is not None:
            ref = entry.get('ref') or {}
            ref['fact'] = new_value
            ref['ts'] = self._now_iso()
            with suppress(Exception):
                self.session_cache.save()
            self._emit('memory_update', {'action': 'edit', 'fact': new_value})
            return
        if self.mem is not None:
            text = entry.get('fact')
            tags = entry.get('tags') or []
            scope = entry.get('scope', 'long')
            with suppress(Exception):
                if text:
                    self.mem.forget_smart(text, last_hint=None)
                self.mem.add_fact(new_value if new_value.endswith('.') else new_value + '.', tags=tags, scope=scope, provenance='ui_edit', confidence=0.85)
            self._emit('memory_update', {'action': 'edit', 'fact': new_value})

    def on_user_turn(self) -> None:
        if self.session_cache is not None:
            with suppress(Exception):
                self.session_cache.save()
        self.refresh_memory(active_filter=self.memory_filter)

    def on_assistant_turn(self) -> None:
        if self.session_cache is not None:
            with suppress(Exception):
                self.session_cache.save()
        self.refresh_memory(active_filter=self.memory_filter)

    # --- realtime controls -------------------------------------------------
    def handle_ptt(self, payload: Dict[str, Any]) -> None:
        state_val = str((payload or {}).get('state') or '').strip().lower()
        target = str((payload or {}).get('target') or '').strip().lower()
        if state_val == 'pressed':
            if callable(self._ptt_press_cb):
                with suppress(Exception):
                    self._ptt_press_cb()
        elif state_val == 'released':
            if callable(self._ptt_release_cb):
                with suppress(Exception):
                    self._ptt_release_cb()
        elif state_val == 'toggle':
            enabled = self.state.toggle_speech()
            self._emit('state', {
                'phase': 'listening' if enabled else 'standby',
                'interaction_mode': 'talk',
                'speech_enabled': enabled,
            })
        elif state_val == 'toggle_mute' or target == 'mute':
            muted = self.state.toggle_mute()
            self._emit('state', {
                'phase': 'muted' if muted else 'listening',
                'interaction_mode': 'talk',
                'muted': muted,
            })

    def handle_override(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or '').strip().lower()
        if action == 'accept_suggestion':
            label = str(payload.get('label') or '').strip()
            if label:
                _ipc.enqueue_chat(label)

    # --- health ------------------------------------------------------------
    def handle_health(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'run').lower()
        if action == 'clear':
            self._emit('health_log', {'message': ''})
            return
        if action == 'status':
            self._emit_health_status()
            return
        message = _run_healthcheck(None)
        self._emit('health_log', {'message': message})
        diag = _run_diagnostics(None)
        if diag:
            for line in diag.splitlines():
                self._emit('health_log', {'message': line})
        self._emit_health_status()

    def _emit_health_status(self) -> None:
        voice_enabled = bool(getattr(self.state.voice, 'enabled', True))
        muted = bool(getattr(self.state, 'muted', False))
        net_state = NetworkGate.state().name if NetworkGate._inited else 'UNKNOWN'
        tiles = {
            'voice': {
                'status': 'OK' if voice_enabled and not muted else ('MUTED' if muted else 'OFF'),
                'value': 'Ready' if voice_enabled else 'Disabled',
                'note': 'Speech enabled' if voice_enabled else 'Speech disabled',
            },
            'network': {
                'status': net_state,
                'value': 'enabled' if NetworkGate.state().name == 'SESSION' else 'offline',
                'note': 'Network gate active' if NetworkGate.state().name == 'SESSION' else 'Requests require approval',
            },
        }
        remaining = NetworkGate.time_remaining()
        if isinstance(remaining, (int, float)) and remaining > 0:
            tiles['network']['note'] = f"Expires in {int(remaining // 60)}m"
        self._emit('health_status', {'tiles': tiles})

    # --- settings ----------------------------------------------------------
    def handle_settings(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'values').lower()
        if action == 'refresh':
            self._emit_settings_snapshot(include_options=True)
            return
        if action == 'capture_hotkey':
            self._emit_error('settings', 'Hotkey capture not supported yet.')
            return
        if action == 'reset':
            hotkey_default = 'Cmd+Shift+Space' if sys.platform == 'darwin' else 'Ctrl+Shift+Space'
            self.state.set_voice(
                enabled=True,
                ptt=True,
                backend='default',
                autospeak=True,
                rate=190,
                device_hint='system_default',
                hotkey=hotkey_default,
            )
            self.state.set_speech(True)
            self.state.set_mute(False)
            os.environ['NERION_ALLOW_NETWORK'] = '1'
            with suppress(Exception):
                NetworkGate.init(load_config())
            self._emit_health_status()
            self._emit_settings_snapshot(include_options=True)
            return
        values = payload.get('values') or payload
        if not isinstance(values, dict):
            return
        rate = values.get('voiceRate')
        if rate is not None:
            with suppress(Exception):
                rate_int = int(rate)
                _tts_set_params(rate=rate_int)
                self.state.set_voice(rate=rate_int)
        backend = values.get('voiceBackend')
        if backend:
            self.state.set_voice(backend=backend)
        device = values.get('device')
        if device:
            with suppress(Exception):
                set_device_index(None if device in {'system', 'system_default'} else device)
                hint = None if device in {'system', 'system_default'} else device
                self.state.set_voice(device_hint=hint or 'system_default')
        autospeak = values.get('autospeak')
        if autospeak is not None:
            self.state.set_voice(autospeak=bool(autospeak))
        hotkey = values.get('hotkey')
        if hotkey:
            self.state.set_voice(hotkey=str(hotkey))
        offline = values.get('alwaysOffline') or values.get('offline')
        if offline is not None:
            os.environ['NERION_ALLOW_NETWORK'] = '0' if offline else '1'
            NetworkGate.init(load_config())
            self._emit_health_status()
        self._emit_settings_snapshot(include_options=False)

    # --- learning ---------------------------------------------------------
    def handle_learning(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'refresh').lower()
        if action in {'refresh', 'drawer'}:
            self._emit_learning_timeline()
            return
        if action == 'clear':
            self.learning_events = {}
            self.learning_selected = None
            self._emit('learning_timeline', {'events': [], 'selected_id': None})
            return
        if action == 'select':
            event_id = payload.get('event_id')
            if event_id:
                self._emit_learning_diff(str(event_id))
            return

    def _emit_learning_timeline(self) -> None:
        events: list[dict[str, Any]] = []
        self.learning_events = {}
        try:
            prefs = _learning_load_prefs(merge_global=True)
        except Exception:
            prefs = {}
        stats = prefs.get('stats') or {}
        if stats:
            total = int(stats.get('successes', 0) or 0) + int(stats.get('failures', 0) or 0)
            successes = int(stats.get('successes', 0) or 0)
            confidence = (successes / total) if total else 0.0
            event_id = 'learn-stats'
            event = {
                'id': event_id,
                'scope': 'workspace',
                'key': 'success_ratio',
                'value': f"{successes}/{max(1,total)}",
                'confidence': min(1.0, confidence),
                'timestamp': _humanize_timesince(stats.get('updated_at')),
                'summary': 'Learning successes vs failures updated',
                'source': 'auto',
                'details': stats,
            }
            events.append(event)
            self.learning_events[event_id] = event
        guard = (prefs.get('guardrails') or {})
        metrics = guard.get('metrics') or {}
        if guard:
            event_id = 'learn-guardrails'
            event = {
                'id': event_id,
                'scope': 'guardrails',
                'key': 'guardrails',
                'value': 'breach' if guard.get('breached') else 'ok',
                'confidence': max(0.0, min(1.0, 1.0 - float(metrics.get('error_rate', 0.0)))),
                'timestamp': _humanize_timesince(prefs.get('last_migrated')),
                'summary': 'Guardrail metrics updated',
                'source': 'auto',
                'details': guard,
            }
            events.append(event)
            self.learning_events[event_id] = event
        live_path = Path('out/learning/live.json')
        if live_path.exists():
            with suppress(Exception):
                live = json.loads(live_path.read_text(encoding='utf-8')) or {}
                if live:
                    event_id = 'learn-live'
                    event = {
                        'id': event_id,
                        'scope': 'live',
                        'key': 'tool_success_rate',
                        'value': ', '.join(f"{k}:{int(v*100)}%" for k, v in (live.get('tool_success_rate') or {}).items()),
                        'confidence': 0.6,
                        'timestamp': 'now',
                        'summary': 'Live tool success rates snapshot',
                        'source': 'auto',
                        'details': live,
                    }
                    events.append(event)
                    self.learning_events[event_id] = event
        ab_path = Path('out/learning/ab_status.json')
        if ab_path.exists():
            with suppress(Exception):
                ab = json.loads(ab_path.read_text(encoding='utf-8')) or {}
                event_id = 'learn-experiments'
                event = {
                    'id': event_id,
                    'scope': 'experiments',
                    'key': 'experiments',
                    'value': 'active' if ab.get('active') else 'idle',
                    'confidence': 0.5,
                    'timestamp': 'recent',
                    'summary': 'A/B experiment status updated',
                    'source': 'auto',
                    'details': ab,
                }
                events.append(event)
                self.learning_events[event_id] = event
        selected = events[0]['id'] if events else None
        self.learning_selected = selected
        self._emit('learning_timeline', {'events': events, 'selected_id': selected})
        if selected:
            self._emit_learning_diff(selected)

    def _emit_learning_diff(self, event_id: str) -> None:
        event = self.learning_events.get(event_id)
        if not event:
            return
        details = event.get('details') or {}
        diff = json.dumps(details, indent=2, ensure_ascii=False) if details else json.dumps(event, indent=2, ensure_ascii=False)
        self._emit('learning_diff', {'event_id': event_id, 'diff': diff})

    def _format_upgrade_plan(self, plan: Dict[str, Any], offer_id: str, *, step_status: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return {
                'id': offer_id,
                'summary': 'No upgrade plan available.',
                'source': 'unknown',
                'files': [],
                'estimate': '—',
                'steps': [{'label': 'Review upgrade readiness', 'status': step_status or 'pending'}],
            }
        meta = plan.get('metadata') if isinstance(plan.get('metadata'), dict) else {}
        summary = meta.get('summary')
        if isinstance(summary, dict):
            summary = ', '.join(f"{k}: {v}" for k, v in summary.items())
        if not isinstance(summary, str) or not summary.strip():
            summary = 'Self-audit improvement plan generated.'
        estimate = meta.get('estimate')
        if not isinstance(estimate, str) or not estimate.strip():
            estimate = '—'
        actions = plan.get('actions') if isinstance(plan.get('actions'), list) else []
        steps: List[Dict[str, Any]] = []
        for idx, action in enumerate(actions or [], start=1):
            if not isinstance(action, dict):
                continue
            kind = str(action.get('kind') or action.get('action') or f'step {idx}')
            label = kind.replace('_', ' ').strip().capitalize()
            payload = action.get('payload') if isinstance(action.get('payload'), dict) else {}
            target = ''
            for key in ('path', 'file', 'target', 'name', 'module'):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    target = val.strip()
                    break
            if target:
                label = f"{label}: {target}"
            steps.append({'label': label, 'status': step_status or 'pending'})
        if not steps:
            steps = [{'label': 'Review upgrade readiness', 'status': step_status or 'pending'}]
        files: set[str] = set()
        files_field = plan.get('files')
        if isinstance(files_field, list):
            for item in files_field:
                if isinstance(item, str) and item.strip():
                    files.add(item.strip())
        target_file = plan.get('target_file')
        if isinstance(target_file, str) and target_file.strip():
            files.add(target_file.strip())
        for action in actions or []:
            if not isinstance(action, dict):
                continue
            payload = action.get('payload') if isinstance(action.get('payload'), dict) else {}
            for key in ('path', 'file', 'target_file'):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    files.add(val.strip())
        formatted = {
            'id': offer_id,
            'summary': summary,
            'source': meta.get('source', 'selfaudit'),
            'files': sorted(files),
            'estimate': estimate,
            'steps': steps,
        }
        score = meta.get('score')
        if isinstance(score, (int, float)):
            formatted['score'] = float(score)
        formatted['meta'] = meta
        return formatted

    def _emit_upgrade_plan(self, plan: Dict[str, Any], offer_id: str, *, step_status: Optional[str] = None) -> None:
        formatted = self._format_upgrade_plan(plan, offer_id, step_status=step_status)
        self._emit('selfcode_plan', {'plan': formatted})

    # --- upgrade -----------------------------------------------------------
    def handle_upgrade(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'refresh').lower()
        offer_id = str((payload or {}).get('offer_id') or 'upgrade-readiness').strip() or 'upgrade-readiness'
        if action in {'refresh', 'status'}:
            self._emit_upgrade_offers()
            cached = self._upgrade_plan_cache.get(offer_id)
            if cached:
                self._emit_upgrade_plan(cached, offer_id)
            return
        if action == 'clear':
            self._upgrade_plan_cache.clear()
            self._emit('upgrade_offer', {'offers': []})
            self._emit('selfcode_plan', {'plan': None})
            return
        if action == 'preview':
            try:
                plan = self._upgrade_plan_cache.get(offer_id)
                if plan is None:
                    plan = _generate_upgrade_plan(Path('.'))
                    self._upgrade_plan_cache[offer_id] = plan
                self._emit_upgrade_plan(plan, offer_id)
            except Exception as exc:
                self._emit_error('upgrade', f'Preview failed: {exc}')
            return
        if action in {'safe_apply', 'force_apply'}:
            ok = False
            try:
                ok = _upgrade_handle_choice('upgrade now', watcher=self.watcher)
            except Exception as exc:
                self._emit_error('upgrade', f'Apply failed: {exc}')
            if ok:
                plan = self._upgrade_plan_cache.pop(offer_id, None)
                if plan:
                    self._emit_upgrade_plan(plan, offer_id, step_status='done')
                else:
                    self._emit('selfcode_plan', {'plan': None})
            self._emit_upgrade_offers()
            return
        if action == 'dismiss':
            handled = False
            try:
                handled = _upgrade_handle_choice('remind me later', watcher=self.watcher)
            except Exception as exc:
                self._emit_error('upgrade', f'Dismiss failed: {exc}')
            if handled:
                self._upgrade_plan_cache.pop(offer_id, None)
                self._emit('selfcode_plan', {'plan': None})
            self._emit_upgrade_offers()
            return
        if action == 'defer':
            handled = False
            try:
                handled = _upgrade_handle_choice('tonight', watcher=self.watcher)
            except Exception as exc:
                self._emit_error('upgrade', f'Defer failed: {exc}')
            if handled:
                self._upgrade_plan_cache.pop(offer_id, None)
                self._emit('selfcode_plan', {'plan': None})
            self._emit_upgrade_offers()
            return
        self._emit_error('upgrade', f'Action "{action}" not supported yet.')

    def _emit_upgrade_offers(self) -> None:
        readiness: Dict[str, Any] = {}
        with suppress(Exception):
            readiness = readiness_report()
        recent = int((readiness or {}).get('recent_knowledge', 0) or 0)
        threshold = int((readiness or {}).get('threshold', 5) or 5)
        score = min(1.0, recent / max(1, threshold)) if threshold else 0.0
        offer = {
            'id': 'upgrade-readiness',
            'title': 'Self-learning readiness',
            'why': f"{recent} new knowledge item(s) since last upgrade.",
            'score': score,
            'active': bool((readiness or {}).get('should_offer', False)),
            'meta': readiness,
        }
        self._emit('upgrade_offer', {'offers': [offer]})

    # --- artifacts ---------------------------------------------------------
    def handle_artifact(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'refresh').lower()
        if action in {'refresh', 'list'}:
            self._emit_artifact_snapshot()
            return
        if action == 'speak':
            summary = self._emit_artifact_snapshot(return_summary=True)
            if summary:
                with suppress(Exception):
                    safe_speak(summary, watcher=self.watcher, force=False)

    def _emit_artifact_snapshot(self, *, return_summary: bool = False):
        active = getattr(self.state, 'active', None)
        path = None
        if isinstance(active, dict):
            path = active.get('last_artifact_path')
        elif hasattr(active, 'get'):
            path = active.get('last_artifact_path')
        if not path:
            self._emit('artifact_list', {'items': []})
            return '' if return_summary else None
        artifact_path = Path(path)
        if not artifact_path.exists():
            self._emit('artifact_list', {'items': []})
            return '' if return_summary else None
        try:
            text = artifact_path.read_text(encoding='utf-8')
        except Exception:
            text = ''
        lines = text.strip().splitlines()
        summary_text = '\n'.join(lines[:12])
        items = [{
            'id': artifact_path.name,
            'title': artifact_path.name,
            'kind': 'doc',
            'summary': summary_text,
            'meta': {'path': str(artifact_path)},
        }]
        self._emit('artifact_list', {'items': items})
        if return_summary:
            return '\n'.join(lines[:4])
        return None

    # --- patch -------------------------------------------------------------
    def handle_patch(self, payload: Dict[str, Any]) -> None:
        action = str((payload or {}).get('action') or 'review').lower()
        if action in {'review', 'refresh'}:
            overview, diff_payload, findings = self._collect_patch_state()
            if overview:
                self._emit('patch_overview', overview)
            if diff_payload:
                self._emit('patch_diff', diff_payload)
            if findings:
                self._emit('patch_findings', findings)
            return
        if action == 'clear':
            self._emit('patch_clear', {})
            return
        if action in {'toggle_hunk', 'set_diff_mode'}:
            return
        self._emit('error', {'code': 'patch', 'message': f'{action} not supported in Electron UI yet.'})

    def _collect_patch_state(self):
        try:
            status = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, check=False)
            lines = [ln.strip() for ln in (status.stdout or '').splitlines() if ln.strip()]
        except Exception:
            lines = []
        entries: List[Dict[str, Any]] = []
        for raw in lines:
            if len(raw) < 3:
                continue
            status_code = raw[:2].strip()
            remainder = raw[3:].strip()
            old_path = remainder
            new_path = remainder
            if '->' in remainder:
                parts = [p.strip() for p in remainder.split('->', 1)]
                if len(parts) == 2:
                    old_path, new_path = parts
            entries.append({'status': status_code or '--', 'path': new_path, 'old_path': old_path})
        files = [entry['path'] for entry in entries if entry.get('path')]
        summary = f"{len(files)} file(s) changed" if files else 'Working tree clean'
        overview = {
            'summary': summary,
            'meta': [f"Files: {len(files)}"] if files else [],
            'findings': ['Review git diff to inspect pending changes.'],
        }
        diff_payload = None
        findings = None
        if entries:
            first_entry = entries[0]
            first = first_entry.get('path')
            try:
                diff = subprocess.run(['git', 'diff', '--', first], capture_output=True, text=True, check=False)
                diff_text = diff.stdout or ''
            except Exception:
                diff_text = ''
            hunks = []
            for idx, line in enumerate(diff_text.splitlines()):
                if line.startswith('@@'):
                    hunks.append({'id': f'h{idx}', 'label': line})
            old_text = ''
            git_ref = first_entry.get('old_path') or first
            if git_ref:
                try:
                    show = subprocess.run(['git', 'show', f'HEAD:{git_ref}'], capture_output=True, text=True, check=False)
                    if show.returncode == 0:
                        old_text = show.stdout or ''
                except Exception:
                    old_text = ''
            new_text = ''
            path_obj = Path(first) if first else None
            if path_obj and path_obj.exists():
                with suppress(Exception):
                    new_text = path_obj.read_text(encoding='utf-8')
            diff_payload = {
                'file': first,
                'left': old_text,
                'right': new_text,
                'diff': diff_text,
                'hunks': hunks,
            }
            findings = {
                'findings': ['Diff generated from git workspace.'],
                'risk': 'UNASSESSED',
            }
        return overview, diff_payload, findings

    # --- selfcode ----------------------------------------------------------
    def handle_selfcode(self, payload: Dict[str, Any]) -> None:
        self._emit('error', {'code': 'selfcode', 'message': 'Self-code actions are not supported in Electron UI yet.'})
