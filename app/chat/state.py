from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque
import threading
import time

@dataclass
class VoiceSettings:
    """Runtime voice controls (seeded from settings.yaml)."""
    enabled: bool = True           # global speech toggle (F1 / /speech on|off)
    ptt: bool = True               # push-to-talk mode
    device_hint: Optional[str] = None
    barge_in: bool = True          # allow key press to interrupt TTS
    temperature_override: Optional[float] = None   # live override set by meta-commands
    temperature_override_set_at: Optional[float] = None
    temperature_override_ttl_s: int = 300         # seconds; auto-expire override

    def set_temp_override(self, value: Optional[float]) -> None:
        """Set or clear the live temperature override.
        When set, it remains active until TTL expires or cleared explicitly.
        """
        self.temperature_override = value
        self.temperature_override_set_at = time.time() if value is not None else None

    def current_temperature(self, default_temp: float) -> float:
        """Return the active temperature for this turn, expiring override after TTL."""
        if (
            self.temperature_override is not None
            and self.temperature_override_set_at is not None
        ):
            if time.time() - self.temperature_override_set_at > self.temperature_override_ttl_s:
                # auto-expire
                self.temperature_override = None
                self.temperature_override_set_at = None
        return self.temperature_override if self.temperature_override is not None else float(default_temp)

@dataclass
class ChatState:
    """Shared chat runtime state with a lock for thread-safety."""
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    muted: bool = False
    last_user_text: str = ""
    last_bot_text: str = ""
    _ts_created: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ---- conversational state manager (CSM) ----
    # One active conversation + small LRU of recent ones
    # Each conversation is a dict with keys:
    #   conversation_id, topic, created_at, last_touched, chat_history, last_artifact_path, slots
    active_conversation: Optional[Dict] = None
    recent_conversations: deque = field(default_factory=lambda: deque(maxlen=5))

    # ---- configuration setters ----
    def set_voice(self, **kwargs) -> None:
        """Update fields on self.voice atomically."""
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self.voice, k):
                    setattr(self.voice, k, v)

    # ---- speech/mute controls ----
    def toggle_speech(self) -> bool:
        with self.lock:
            self.voice.enabled = not self.voice.enabled
            return self.voice.enabled

    def set_speech(self, on: bool) -> None:
        with self.lock:
            self.voice.enabled = bool(on)

    def toggle_mute(self) -> bool:
        with self.lock:
            self.muted = not self.muted
            return self.muted

    def set_mute(self, on: bool) -> None:
        with self.lock:
            self.muted = bool(on)

    # ---- convenience queries ----
    def should_speak(self) -> bool:
        """True when TTS should be audible right now."""
        with self.lock:
            return self.voice.enabled and not self.muted

    # ---- CSM helpers ----
    @staticmethod
    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _norm(text: str) -> List[str]:
        """Very small normalizer for topic/keyword overlap matching."""
        if not text:
            return []
        import re as _re
        t = _re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        toks = [w for w in t.split() if w and w not in {"the","a","an","of","for","to","and","or","in","on"}]
        return toks

    def _overlap(self, a: str, b: str) -> float:
        aa, bb = set(self._norm(a)), set(self._norm(b))
        if not aa or not bb:
            return 0.0
        inter = len(aa & bb)
        union = len(aa | bb)
        return inter / max(1, union)

    # Open a new conversation tab; archive current active if present
    def open_new_conversation(self, *, topic: str, slots: Optional[Dict] = None, seed_turns: Optional[List[Dict]] = None) -> str:
        with self.lock:
            if self.active_conversation:
                # push active to recent before switching
                self.active_conversation["last_touched"] = self._now_iso()
                self.recent_conversations.appendleft(self.active_conversation)
            conv_id = f"conv_{int(time.time())}"
            conv = {
                "conversation_id": conv_id,
                "topic": topic,
                "created_at": self._now_iso(),
                "last_touched": self._now_iso(),
                "chat_history": list(seed_turns or []),
                "last_artifact_path": None,
                "slots": dict(slots or {}),
            }
            self.active_conversation = conv
            return conv_id

    def get_active(self) -> Optional[Dict]:
        with self.lock:
            return self.active_conversation

    @property
    def active(self) -> Optional[Dict]:
        """Back-compat alias: return the currently active conversation dict."""
        return self.get_active()

    def touch_active(self) -> None:
        with self.lock:
            if self.active_conversation:
                self.active_conversation["last_touched"] = self._now_iso()

    def archive_active(self) -> None:
        with self.lock:
            if self.active_conversation:
                self.active_conversation["last_touched"] = self._now_iso()
                self.recent_conversations.appendleft(self.active_conversation)
                self.active_conversation = None

    def switch_to(self, conversation_id: str) -> bool:
        """Switch to a recent conversation by id. Returns True if switched."""
        with self.lock:
            if self.active_conversation and self.active_conversation.get("conversation_id") == conversation_id:
                return True
            idx = None
            for i, c in enumerate(self.recent_conversations):
                if c.get("conversation_id") == conversation_id:
                    idx = i
                    break
            if idx is None:
                return False
            # move current active to recent
            if self.active_conversation:
                self.active_conversation["last_touched"] = self._now_iso()
                self.recent_conversations.appendleft(self.active_conversation)
            # bring selected to active and remove from recent
            conv = self.recent_conversations[idx]
            del self.recent_conversations[idx]
            self.active_conversation = conv
            self.active_conversation["last_touched"] = self._now_iso()
            return True

    def find_recent_by_topic(self, query: str, *, min_overlap: float = 0.25, ttl_s: int = 3600) -> Optional[Dict]:
        """Find a recent conversation whose topic overlaps the query within TTL."""
        with self.lock:
            now = time.time()
            for conv in list(self.recent_conversations):
                # TTL gating
                try:
                    ts = time.mktime(time.strptime(conv.get("last_touched", ""), "%Y-%m-%dT%H:%M:%SZ"))
                except Exception:
                    ts = now
                if now - ts > ttl_s:
                    continue
                if self._overlap(query, conv.get("topic", "")) >= min_overlap:
                    return conv
            return None

    def append_turn(self, role: str = None, content: str = None, **kwargs) -> None:
        """Append a chat turn to the active conversation.
        Accepts either positional (role, content) or keyword args for back-compat.
        """
        if role is None:
            role = kwargs.get("role")
        if content is None:
            content = kwargs.get("content")
        if role is None or content is None:
            return
        with self.lock:
            if not self.active_conversation:
                return
            self.active_conversation.setdefault("chat_history", []).append({"role": role, "content": content})
            self.active_conversation["last_touched"] = self._now_iso()

    def set_last_artifact_path(self, path: Optional[str]) -> None:
        with self.lock:
            if self.active_conversation is not None:
                self.active_conversation["last_artifact_path"] = path
