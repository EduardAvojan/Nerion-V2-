from __future__ import annotations

import base64
import datetime as dt
import difflib
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.security import fs_guard

from .retrieval import DefaultEmbedder, HybridRetriever

# ---- Configurable defaults (can be overridden via init(cfg=...) or env) ----
_DEF = {
    "path": os.getenv("NERION_MEMORY_PATH", "memory_db.json"),
    "short_term_ttl_days": int(os.getenv("NERION_MEMORY_TTL_DAYS", "14")),
    "decay_per_day": float(os.getenv("NERION_MEMORY_DECAY", "0.15")),
    "promotion_threshold": float(os.getenv("NERION_MEMORY_PROMOTION", "2.5")),
    "max_items": int(os.getenv("NERION_MEMORY_MAX_ITEMS", "400")),
    "min_match": float(os.getenv("NERION_MEMORY_MIN_MATCH", "0.55")),
}

# Repo root (project root) used to enforce on-repo storage for memory file
REPO_ROOT = Path(__file__).resolve().parents[2]

try:  # Optional journaling backend
    from core.memory import journal_store as _memory_journal  # type: ignore
except Exception:  # pragma: no cover
    _memory_journal = None  # type: ignore

TRUTHY = {"1", "true", "yes", "on"}
PII_REGEX = re.compile(r"(\b\d{3}-\d{2}-\d{4}\b|\b\d{16}\b|@|api_key|secret|password)", re.I)
POISON_REGEX = re.compile(r"(ignore previous|system:|developer:|override|jailbreak)", re.I)

def _repo_default_memory_path() -> Path:
    return REPO_ROOT / "memory_db.json"

def _ensure_repo_path(p: str | os.PathLike) -> Path:
    """Ensure memory path lives inside repo; fallback to repo default if not."""
    try:
        return fs_guard.ensure_in_repo(REPO_ROOT, str(p))
    except Exception:
        # Fallback to a safe default inside the repo
        return _repo_default_memory_path()

def _cfg_get(cfg: Optional[dict], *path, default=None):
    try:
        cur = cfg if cfg is not None else {}
        for key in path:
            if cur is None:
                return default
            if hasattr(cur, 'get'):
                cur = cur.get(key)
            else:
                return default
        return default if cur is None else cur
    except Exception:
        return default

# Ensure default memory file is jailed to repo
MEMORY_FILE = str(_ensure_repo_path(_DEF["path"]))
_DEFAULT_NS = {"user": "global", "workspace": "default", "project": "default"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _pad_b64(data: str) -> str:
    rem = len(data) % 4
    return data if rem == 0 else data + "=" * (4 - rem)

# Optional initializer: allows app layer to set default memory path and knobs.
# Priority:
#   1) Explicit arg `path`
#   2) cfg['memory']['db_path'] or cfg['chat']['memory_path']
#   3) Env NERION_MEMORY_PATH
#   4) Existing default
# Also loads TTL/decay knobs if provided in cfg['memory'].
def init(cfg=None, path: Optional[str] = None) -> None:
    global MEMORY_FILE, _DEF
    try:
        p = (
            path
            or _cfg_get(cfg, "paths", "memory_db_path")
            or _cfg_get(cfg, "paths", "memory_db")  # legacy alias
            or _cfg_get(cfg, "memory", "db_path")
            or _cfg_get(cfg, "chat", "memory_path")
            or os.getenv("NERION_MEMORY_PATH")
            or MEMORY_FILE
        )
        p = str(_ensure_repo_path(p))
        if p:
            MEMORY_FILE = str(p)

        # Optional knobs
        ttl = _cfg_get(cfg, "memory", "short_term_ttl_days")
        if ttl is not None:
            _DEF["short_term_ttl_days"] = int(ttl)
        decay = _cfg_get(cfg, "memory", "decay_per_day")
        if decay is not None:
            _DEF["decay_per_day"] = float(decay)
        promo = _cfg_get(cfg, "memory", "promotion_threshold")
        if promo is not None:
            _DEF["promotion_threshold"] = float(promo)
        max_items = _cfg_get(cfg, "memory", "max_items")
        if max_items is not None:
            _DEF["max_items"] = int(max_items)
        min_match = _cfg_get(cfg, "memory", "min_match")
        if min_match is not None:
            _DEF["min_match"] = float(min_match)
    except Exception:
        # never crash app on init
        pass


class Verdict:
    def __init__(self, flagged: bool, reasons: List[str], source: str, confidence: float) -> None:
        self.flagged = flagged
        self.reasons = reasons
        self.source = source
        self.confidence = confidence


def _summarize(texts: List[str]) -> str:
    seen = set()
    ordered: List[str] = []
    for text in texts:
        norm = (text or '').strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        ordered.append(norm)
        seen.add(key)
    return '; '.join(ordered)[:512]


def _merge_tags(items: List[Dict[str, Any]]) -> List[str]:
    acc: set[str] = set()
    for item in items:
        acc.update(item.get('tags', []) or [])
    return sorted(acc)


class LongTermMemory:
    SCHEMA_VERSION = 2
    ENTRY_VERSION = 1

    def __init__(
        self,
        path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ns: Optional[Dict[str, str]] = None,
    ) -> None:
        if path is None and cfg:
            path = (
                cfg.get("path")
                or cfg.get("memory_db_path")
                or cfg.get("memory_path")
            )
        if path is None:
            path = MEMORY_FILE
        self.path = str(_ensure_repo_path(path))
        # knobs (copied from _DEF at construction time)
        self.ttl_days = int(_DEF["short_term_ttl_days"])
        self.decay_per_day = float(_DEF["decay_per_day"])
        self.promotion_threshold = float(_DEF["promotion_threshold"])
        self.max_items = int(_DEF["max_items"])
        self.min_match = float(_DEF["min_match"])
        self._last_scope: str = "short"
        env_ns = {
            'user': os.getenv('USER', _DEFAULT_NS['user']),
            'workspace': os.getenv('NERION_SCOPE_WS', _DEFAULT_NS['workspace']),
            'project': os.getenv('NERION_SCOPE_PROJECT', _DEFAULT_NS['project']),
        }
        self.ns: Dict[str, str] = {
            **_DEFAULT_NS,
            **{k: str(v) for k, v in env_ns.items() if v is not None},
            **{k: str(v) for k, v in (ns or {}).items()},
        }
        self._data: Dict[str, Any] = {"schema": self.SCHEMA_VERSION, "version": 1, "facts": []}
        self.memories: List[Dict[str, Any]] = self._data["facts"]
        self._crypto = self._init_crypto()
        self._rotation_marker: int = 0
        self._embedder = DefaultEmbedder()
        self._retriever: Optional[HybridRetriever] = None
        self.last_ref: Optional[str] = None
        self.last_learned: Optional[str] = None
        self._load()

    # --- persistence helpers -------------------------------------------------
    def _init_crypto(self):
        key = (os.getenv("NERION_MEMORY_KEY") or "").strip()
        if not key:
            return None
        material = self._derive_crypto_key(key)
        try:
            from cryptography.fernet import Fernet  # type: ignore
        except Exception:
            return None
        try:
            return Fernet(material)
        except Exception:
            return None

    def _derive_crypto_key(self, key: str) -> bytes:
        with suppress(ValueError):
            raw = bytes.fromhex(key)
            return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
        with suppress(Exception):
            decoded = base64.urlsafe_b64decode(_pad_b64(key))
            if len(decoded) == 32:
                return base64.urlsafe_b64encode(decoded)
        raw = key.encode("utf-8")
        return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())

    def _encrypt_if_needed(self, payload: bytes) -> bytes:
        if not payload or self._crypto is None:
            return payload
        try:
            return self._crypto.encrypt(payload)
        except Exception:
            return payload

    def _decrypt_if_needed(self, payload: bytes) -> bytes:
        if not payload:
            return payload
        if self._crypto is None:
            return payload
        try:
            return self._crypto.decrypt(payload)
        except Exception:
            return payload

    def _atomic_write(self, payload: bytes) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".mem.", dir=directory)
        try:
            with io.open(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)

    def _maybe_rotate(self) -> None:
        try:
            threshold = int(os.getenv("NERION_MEMORY_ROTATE_BYTES", "134217728"))
        except Exception:
            threshold = 134_217_728
        if threshold <= 0:
            return
        with suppress(Exception):
            size = os.path.getsize(self.path)
            if size <= threshold:
                self._rotation_marker = 0
                return
            if self._rotation_marker == size:
                return
            ts = time.strftime("%Y%m%d-%H%M%S")
            gz_path = f"{self.path}.{ts}.gz"
            with open(self.path, "rb") as src, gzip.open(gz_path, "wb") as dst:
                dst.write(src.read())
            self._rotation_marker = size

    def _source_hash(self, text: str) -> str:
        return hashlib.blake2b((text or "").encode("utf-8"), digest_size=16).hexdigest()

    def _mk_id(self, fact: str, ts: str) -> str:
        return hashlib.sha256(f"{fact}\x1f{ts}".encode("utf-8")).hexdigest()[:16]

    def _ensure_entry_defaults(self, item: Dict[str, Any]) -> None:
        fact = item.get("fact", "")
        timestamp = item.get("timestamp") or _now_iso()
        item.setdefault("id", self._mk_id(fact, timestamp))
        item.setdefault("tags", [])
        item.setdefault("score", 1.0)
        item.setdefault("timestamp", timestamp)
        item.setdefault("last_used_ts", item.get("last_used_ts") or timestamp)
        item.setdefault("scope", item.get("scope") or "short")
        item.setdefault("deleted", bool(item.get("deleted", False)))
        item.setdefault("superseded_by", item.get("superseded_by"))
        item.setdefault("ttl_days", item.get("ttl_days"))
        item.setdefault("provenance", item.get("provenance", "import"))
        item.setdefault("confidence", float(item.get("confidence", 0.7)))
        item.setdefault("source_hash", item.get("source_hash") or self._source_hash(fact))
        item.setdefault("version", item.get("version", self.ENTRY_VERSION))
        ns_val = item.get("ns")
        if not isinstance(ns_val, dict):
            item['ns'] = dict(self.ns)
        else:
            item['ns'] = {k: str(v) for k, v in ns_val.items()}
        try:
            item['uses_count'] = int(item.get('uses_count', 0) or 0)
        except Exception:
            item['uses_count'] = 0

    def _parse_ts(self, value: Optional[str], fallback: Optional[dt.datetime] = None) -> dt.datetime:
        fallback = fallback or dt.datetime.now(dt.timezone.utc)
        if not value:
            return fallback
        with suppress(Exception):
            return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
        return fallback

    def _load(self) -> None:
        path = Path(self.path)
        if not path.exists():
            self._data = {"schema": self.SCHEMA_VERSION, "version": 1, "facts": []}
            self.memories = self._data["facts"]
            return
        raw_bytes = b""
        with suppress(Exception):
            raw_bytes = path.read_bytes()
        if not raw_bytes:
            self._data = {"schema": self.SCHEMA_VERSION, "version": 1, "facts": []}
            self.memories = self._data["facts"]
            return
        decoded = raw_bytes
        if self._crypto is not None:
            decrypted = self._decrypt_if_needed(raw_bytes)
            if decrypted != raw_bytes or not decoded:
                decoded = decrypted
        try:
            payload = json.loads(decoded.decode("utf-8"))
        except Exception:
            payload = []
        if isinstance(payload, list):
            self._data = {"schema": 1, "version": 1, "facts": payload}
        elif isinstance(payload, dict):
            payload.setdefault("facts", [])
            payload.setdefault("schema", payload.get("schema", self.SCHEMA_VERSION))
            payload.setdefault("version", payload.get("version", 1))
            self._data = payload
        else:
            self._data = {"schema": self.SCHEMA_VERSION, "version": 1, "facts": []}
        self.memories = self._data["facts"]
        for it in self.memories:
            self._ensure_entry_defaults(it)
        self._data["schema"] = self.SCHEMA_VERSION
        self._retriever = None

    def _save(self) -> None:
        self._retriever = None
        self._data["schema"] = self.SCHEMA_VERSION
        self._data.setdefault("version", 1)
        self._data["facts"] = self.memories
        payload = json.dumps(self._data, ensure_ascii=False, indent=2).encode("utf-8")
        blob = self._encrypt_if_needed(payload)
        self._atomic_write(blob)
        self._maybe_rotate()

    # --- journaling ---------------------------------------------------------
    def _journal_mutation(
        self,
        op: str,
        *,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
    ) -> None:
        if _memory_journal is None:
            return
        try:
            _memory_journal.log_event(
                "memory_mutation",
                op,
                path=self.path,
                ns=dict(self.ns),
                before=deepcopy(before) if before is not None else None,
                after=deepcopy(after) if after is not None else None,
            )
        except Exception:
            pass

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.memories, embedder=self._embedder)
        return self._retriever

    def _namespace_chain(self) -> List[Dict[str, str]]:
        user = str(self.ns.get('user', 'global'))
        workspace = str(self.ns.get('workspace', 'default'))
        project = str(self.ns.get('project', 'default'))
        return [
            {'user': user, 'workspace': workspace, 'project': project},
            {'user': user, 'workspace': workspace, 'project': '*'},
            {'user': user, 'workspace': '*', 'project': '*'},
            {'user': '*', 'workspace': '*', 'project': '*'},
        ]

    def _match_namespace(self, entry_ns: Any, target: Dict[str, str]) -> bool:
        if not isinstance(entry_ns, dict):
            return False
        for key, expected in target.items():
            if expected == '*':
                continue
            if str(entry_ns.get(key)) != expected:
                return False
        return True

    def _gate(self, text: str, provenance: str, confidence: float) -> Verdict:
        reasons: List[str] = []
        strict = (os.getenv('NERION_MEMORY_STRICT_PIIGATE', '1') or '').strip().lower() in TRUTHY
        if strict and PII_REGEX.search(text or ''):
            reasons.append('pii')
        if POISON_REGEX.search(text or ''):
            reasons.append('prompt_poison')
        flagged = bool(reasons)
        effective_conf = float(confidence) * (0.8 if flagged else 1.0)
        return Verdict(flagged, reasons, provenance, effective_conf)

    def _quarantine_enabled(self) -> bool:
        return (os.getenv('NERION_MEMORY_QUARANTINE', '1') or '').strip().lower() in TRUTHY

    def _quarantine(self, item: Dict[str, Any], verdict: Verdict) -> None:
        if not self._quarantine_enabled():
            return
        try:
            q_path = fs_guard.ensure_in_repo(REPO_ROOT, os.path.join('out', 'memory', 'quarantine.jsonl'))
            Path(q_path).parent.mkdir(parents=True, exist_ok=True)
            record = {
                'ts': _now_iso(),
                'item': item,
                'reasons': verdict.reasons,
                'source': verdict.source,
                'confidence': verdict.confidence,
            }
            with open(q_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write('\n')
            if _memory_journal is not None:
                _memory_journal.log_event(
                    'memory_quarantine',
                    ','.join(verdict.reasons),
                    path=self.path,
                    ns=dict(self.ns),
                    record=record,
                )
        except Exception:
            pass

    def _normalize_text(self, s: str) -> str:
        return ' '.join(re.sub(r"[^a-z0-9\s]", ' ', (s or '').lower()).split())

    def _token_set(self, s: str) -> set:
        return set(self._normalize_text(s).split())

    def _similarity(self, a: str, b: str) -> float:
        ta, tb = self._token_set(a), self._token_set(b)
        if not ta and not tb:
            tok_sim = 0.0
        else:
            inter = len(ta & tb)
            union = len(ta | tb) or 1
            tok_sim = inter / union
        seq_sim = difflib.SequenceMatcher(None, self._normalize_text(a), self._normalize_text(b)).ratio()
        return (0.6 * tok_sim) + (0.4 * seq_sim)

    def find_best_match(self, query: str, min_score: Optional[float] = None):
        q = (query or '').strip()
        if not q:
            return None, 0.0
        min_score = self.min_match if (min_score is None) else float(min_score)
        best = None
        best_score = 0.0
        for m in self._live_items():
            fact = m.get('fact', '')
            score = self._similarity(q, fact)
            if m.get('scope', 'short') == 'long':
                score += 0.05
            if score > best_score:
                best, best_score = m, score
        if best and best_score >= min_score:
            return best, best_score
        return None, best_score

    def record_reference(self, text: Optional[str]) -> None:
        if text:
            self.last_ref = text.strip()

    def forget_smart(self, query: Optional[str], last_hint: Optional[str] = None) -> tuple:
        target = (query or '').strip()
        if not target or target.lower() in {'that', 'it', 'this'}:
            target = last_hint or self.last_ref or ''
        item, score = self.find_best_match(target) if target else (None, 0.0)
        if not item:
            return 0, None
        before = deepcopy(item)
        now = _now_iso()
        item['deleted'] = True
        item['timestamp'] = now
        item['last_used_ts'] = now
        self._journal_mutation('forget', before=before, after=item)
        self._save()
        self.last_ref = item.get('fact')
        self.last_learned = None
        return 1, item.get('fact')

    def unpin_smart(self, query: Optional[str], last_hint: Optional[str] = None) -> tuple:
        target = (query or '').strip()
        if not target or target.lower() in {'that', 'it', 'this'}:
            target = last_hint or self.last_ref or ''
        item, score = self.find_best_match(target) if target else (None, 0.0)
        if not item or item.get('scope', 'short') != 'long':
            return 0, None
        before = deepcopy(item)
        now = _now_iso()
        item['scope'] = 'short'
        item['timestamp'] = now
        item['last_used_ts'] = now
        self._journal_mutation('unpin', before=before, after=item)
        self._save()
        self.last_ref = item.get('fact')
        self.last_learned = None
        return 1, item.get('fact')

    def add_fact(
        self,
        fact: str,
        tags: Optional[List[str]] = None,
        score: float = 1.0,
        scope: str = "short",
        *,
        provenance: str = "manual",
        confidence: float = 0.7,
        ttl_days: Optional[int] = None,
    ) -> bool:
        if not fact:
            return False
        normalized = fact.strip()
        now = _now_iso()
        fact_lc = normalized.lower()
        verdict = self._gate(normalized, provenance, float(confidence))
        effective_conf = float(verdict.confidence)
        if verdict.flagged:
            candidate = {
                'fact': normalized,
                'tags': sorted(tags or []),
                'scope': scope,
                'provenance': provenance,
                'confidence': effective_conf,
                'timestamp': now,
                'ns': dict(self.ns),
            }
            self._quarantine(candidate, verdict)
            return False
        for item in self.memories:
            if item.get('fact', '').strip().lower() == fact_lc:
                before = deepcopy(item)
                item['score'] = min(float(item.get('score', 1.0)) + 0.3 * float(score or 1.0), 5.0)
                item['timestamp'] = now
                item['last_used_ts'] = now
                item['confidence'] = max(float(item.get('confidence', 0.7)), effective_conf)
                prev_scope = item.get('scope', 'short')
                item['scope'] = 'long' if (prev_scope == 'long' or scope == 'long') else scope
                item['deleted'] = bool(item.get('deleted', False))
                item['provenance'] = provenance or item.get('provenance', 'manual')
                if ttl_days is not None:
                    with suppress(Exception):
                        item['ttl_days'] = int(ttl_days)
                if tags:
                    try:
                        cur = set(item.get('tags', []) or [])
                        cur.update(tags)
                        item['tags'] = sorted(cur)
                    except Exception:
                        item['tags'] = list(tags)
                item['source_hash'] = self._source_hash(item.get('fact', ''))
                item['version'] = self.ENTRY_VERSION
                self.last_learned = item.get('fact')
                self.last_ref = self.last_learned
                self._journal_mutation('update', before=before, after=item)
                self._save()
                return True
        entry: Dict[str, Any] = {
            'id': self._mk_id(normalized, now),
            'fact': normalized,
            'tags': sorted(tags or []),
            'score': float(score),
            'timestamp': now,
            'last_used_ts': now,
            'scope': scope,
            'deleted': False,
            'superseded_by': None,
            'ttl_days': int(ttl_days) if ttl_days is not None else None,
            'provenance': provenance,
            'confidence': effective_conf,
            'source_hash': self._source_hash(normalized),
            'version': self.ENTRY_VERSION,
            'ns': dict(self.ns),
            'uses_count': 0,
        }
        self._ensure_entry_defaults(entry)
        self.memories.append(entry)
        self.last_learned = entry['fact']
        self.last_ref = entry['fact']
        self._journal_mutation('insert', after=entry)
        self._save()
        return True

    def forget_matching(self, query: str) -> int:
        count = 0
        now = _now_iso()
        q = query.strip().lower()
        for m in self.memories:
            if not m.get('deleted', False) and q in m.get('fact', '').lower():
                before = deepcopy(m)
                m['deleted'] = True
                m['timestamp'] = now
                m['last_used_ts'] = now
                count += 1
                self._journal_mutation('forget', before=before, after=m)
        if count:
            self._save()
        return count

    def unpin_matching(self, query: str) -> int:
        count = 0
        now = _now_iso()
        q = query.strip().lower()
        for m in self.memories:
            if not m.get('deleted', False) and q in m.get('fact', '').lower() and m.get('scope', 'short') == 'long':
                before = deepcopy(m)
                m['scope'] = 'short'
                m['timestamp'] = now
                m['last_used_ts'] = now
                count += 1
                self._journal_mutation('unpin', before=before, after=m)
        if count:
            self._save()
        return count

    def pin_fact_text(self, text: str) -> int:
        if not text:
            return 0
        best = None
        best_score = 0.0
        text_lc = text.strip().lower()
        for m in self._live_items():
            ratio = difflib.SequenceMatcher(None, m.get('fact', '').lower(), text_lc).ratio()
            if ratio > best_score:
                best_score = ratio
                best = m
        if best and best_score > 0.7:
            if best.get('scope', 'short') != 'long':
                before = deepcopy(best)
                best['scope'] = 'long'
                now = _now_iso()
                best['timestamp'] = now
                best['last_used_ts'] = now
                self._journal_mutation('pin', before=before, after=best)
                self._save()
                return 1
        return 0

    def _utility(self, item: Dict[str, Any], now: dt.datetime) -> float:
        last_used = self._parse_ts(item.get('last_used_ts'), fallback=now)
        recency_days = max(0.0, (now - last_used).total_seconds() / 86400.0)
        score = float(item.get('score', 0.0))
        freq = math.log1p(float(item.get('uses_count', 0) or 0))
        trust = float(item.get('confidence', 0.7))
        tier_boost = 0.3 if item.get('scope', 'short') == 'long' else 0.0
        return 0.6 * score - 0.04 * recency_days + 0.25 * freq + 0.11 * trust + tier_boost

    def prune(self) -> dict:
        """Decay scores, enforce TTL, and tier memories by utility."""
        now_dt = dt.datetime.now(dt.timezone.utc)
        now_iso = _now_iso()
        removed = 0
        decayed = 0

        for item in self.memories:
            if item.get('deleted'):
                continue
            ttl = item.get('ttl_days')
            if ttl is not None:
                ts = self._parse_ts(item.get('timestamp'), fallback=now_dt)
                if (now_dt - ts).days >= int(ttl):
                    before = deepcopy(item)
                    item['deleted'] = True
                    item['timestamp'] = now_iso
                    item['last_used_ts'] = now_iso
                    removed += 1
                    self._journal_mutation('ttl_expire', before=before, after=item)

        for item in self.memories:
            if item.get('deleted') or item.get('superseded_by'):
                continue
            if item.get('scope', 'short') != 'long':
                ts = self._parse_ts(item.get('timestamp'), fallback=now_dt)
                days = max(0, (now_dt - ts).days)
                if days > 0:
                    new_score = max(0.0, float(item.get('score', 1.0)) - self.decay_per_day * days)
                    if new_score != item.get('score', 1.0):
                        before = deepcopy(item)
                        item['score'] = new_score
                        item['timestamp'] = now_iso
                        decayed += 1
                        self._journal_mutation('decay', before=before, after=item)

        carryover = [m for m in self.memories if m.get('superseded_by')]
        live = [m for m in self.memories if not m.get('deleted') and not m.get('superseded_by')]
        pinned = [m for m in live if m.get('scope', 'short') == 'long']
        short_term = [m for m in live if m.get('scope', 'short') != 'long']

        short_term.sort(key=lambda it: self._utility(it, now_dt), reverse=True)
        try:
            hot_cap = max(0, int(os.getenv('NERION_MEMORY_TIER_HOT', '200')))
        except Exception:
            hot_cap = 200
        try:
            warm_cap = max(0, int(os.getenv('NERION_MEMORY_TIER_WARM', '400')))
        except Exception:
            warm_cap = 400
        try:
            cold_cap = max(0, int(os.getenv('NERION_MEMORY_TIER_COLD', '800')))
        except Exception:
            cold_cap = 800

        keep_limit = hot_cap + warm_cap + cold_cap
        keep_short = short_term[:keep_limit]
        evict_short = short_term[keep_limit:]

        for item in evict_short:
            before = deepcopy(item)
            item['deleted'] = True
            item['timestamp'] = now_iso
            item['last_used_ts'] = now_iso
            removed += 1
            self._journal_mutation('capacity_evict', before=before, after=item)

        self.memories = carryover + pinned + keep_short
        self._save()
        return {"decayed": decayed, "removed": removed, "total": len(self.memories)}

    def consolidate(self) -> dict:
        if (os.getenv('NERION_MEMORY_CONSOLIDATE', '0') or '').strip().lower() not in TRUTHY:
            return {"created": 0, "superseded": 0}
        try:
            min_cluster = max(2, int(os.getenv('NERION_MEMORY_CLUSTER_MIN', '4')))
        except Exception:
            min_cluster = 4
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.memories:
            if item.get('deleted') or item.get('superseded_by'):
                continue
            key = (item.get('source_hash') or '')[:6]
            if not key:
                continue
            buckets.setdefault(key, []).append(item)
        created = 0
        superseded = 0
        for items in buckets.values():
            if len(items) < min_cluster:
                continue
            summary = _summarize([it.get('fact', '') for it in items])
            if not summary:
                continue
            existing = next((m for m in self.memories if not m.get('deleted') and m.get('fact', '').strip().lower() == summary.strip().lower()), None)
            if existing:
                canon_id = existing.get('id')
            else:
                ts = _now_iso()
                entry: Dict[str, Any] = {
                    'id': self._mk_id(summary, ts),
                    'fact': summary,
                    'tags': _merge_tags(items),
                    'score': 3.0,
                    'timestamp': ts,
                    'last_used_ts': ts,
                    'scope': 'long',
                    'deleted': False,
                    'superseded_by': None,
                    'ttl_days': None,
                    'provenance': 'consolidate',
                    'confidence': 0.7,
                    'source_hash': self._source_hash(summary),
                    'version': self.ENTRY_VERSION,
                    'ns': dict(self.ns),
                    'uses_count': 0,
                }
                self.memories.append(entry)
                self._journal_mutation('insert', after=entry)
                canon_id = entry['id']
                created += 1
            if not canon_id:
                continue
            for item in items:
                if item.get('id') == canon_id:
                    continue
                before = deepcopy(item)
                item['superseded_by'] = canon_id
                item['timestamp'] = _now_iso()
                superseded += 1
                self._journal_mutation('consolidate_supersede', before=before, after=item)
        self._save()
        return {"created": created, "superseded": superseded}

    def set_ttl_for_text(self, text: str, days: int) -> bool:
        """Set a TTL in days for the best-matching memory fact containing `text`."""
        if not text:
            return False
        item, _ = self.find_best_match(text)
        if not item:
            # substring fallback
            tl = text.strip().lower()
            for m in self._live_items():
                if tl in (m.get('fact','').lower()):
                    item = m
                    break
        if not item:
            return False
        try:
            before = deepcopy(item)
            item['ttl_days'] = int(days)
            now = _now_iso()
            item['timestamp'] = now
            item['last_used_ts'] = now
            self._save()
            self._journal_mutation('set_ttl', before=before, after=item)
            return True
        except Exception:
            return False

    def set_ttl_for_last_learned(self, days: int) -> bool:
        try:
            if not self.last_learned:
                return False
            return self.set_ttl_for_text(self.last_learned, int(days))
        except Exception:
            return False

    def _live_items(self):
        for m in self.memories:
            if not m.get('deleted', False) and not m.get('superseded_by'):
                yield m

    def _dedup(self, items):
        seen = {}
        for m in items:
            key = m.get('fact', '').strip().lower()
            if not key:
                continue
            if key not in seen:
                seen[key] = m
            else:
                prev = seen[key]
                if m.get('score', 1.0) > prev.get('score', 1.0):
                    seen[key] = m
                elif m.get('score', 1.0) == prev.get('score', 1.0):
                    ts1 = prev.get('timestamp', '')
                    ts2 = m.get('timestamp', '')
                    if ts2 > ts1:
                        seen[key] = m
        return list(seen.values())

    def summarize_top(self, k: int=8) -> str:
        items = list(self._live_items())
        if not items:
            return 'No long-term memories stored yet.'
        deduped = self._dedup(items)
        def _key(m):
            ts = m.get('timestamp') or '1970-01-01T00:00:00'
            return (m.get('score', 1.0), ts)
        top = sorted(deduped, key=_key, reverse=True)[:k]
        lines = [f"- {m.get('fact')}" for m in top]
        return '\n'.join(lines)

    def find_relevant(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not (text or '').strip():
            return []
        retriever = self._get_retriever()
        pool = retriever.topk(text, k=max(20, k * 4))
        if not pool:
            return []
        for scope in self._namespace_chain():
            scoped = [item for item in pool if self._match_namespace(item.get('ns'), scope)]
            if scoped:
                return [deepcopy(item) for item in scoped[:k]]
        return [deepcopy(item) for item in pool[:k]]

    def erase_all(self) -> int:
        """Hard clear all memories (returns count)."""
        n = len(self.memories)
        if n:
            snapshot = [deepcopy(m) for m in self.memories]
            self._journal_mutation('erase_all', before={'items': snapshot}, after=None)
        self.memories = []
        self._save()
        self.last_ref = None
        self.last_learned = None
        return n

    def list_memories(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a shallow copy of live memories, optionally filtered by scope."""
        items = [m for m in self._live_items()]
        if scope in {"short", "long"}:
            items = [m for m in items if m.get("scope", "short") == scope]
        return [dict(m) for m in items]

    _patterns = [
        (re.compile('\\b(my favorite (?:drink|food|song|color|movie|game|sport) is)\\s+(?P<val>.+)', re.I), lambda m: f"{m.group(1).strip().capitalize()} {m.group('val').strip()}."),
        (re.compile('\\b(i (?:really )?(?:like|love|enjoy))\\s+(?P<val>.+)', re.I), lambda m: f"User likes {m.group('val').strip()}."),
        (re.compile("\\b(call me|my name is)\\s+(?P<val>[A-Za-z][A-Za-z .'-]+)", re.I), lambda m: f"User prefers to be called {m.group('val').strip()}."),
        (re.compile('\\b(i (?:live|am based) in)\\s+(?P<val>.+)', re.I), lambda m: f"User lives in {m.group('val').strip()}."),
        (re.compile("\\b(i (?:work at|work for|am employed at|am a|i'm a))\\s+(?P<val>.+)", re.I), lambda m: f"User role/work: {m.group('val').strip()}."),
        (re.compile('\\b(my birthday is|i was born on)\\s+(?P<val>.+)', re.I), lambda m: f"User birthday: {m.group('val').strip()}."),
        (re.compile('\\b(i prefer)\\s+(?P<val>.+)', re.I), lambda m: f"User preference: {m.group('val').strip()}."),
    ]

    def extract_from_utterance(self, utterance: str) -> List[str]:
        facts = []
        u = utterance.strip()
        if not u:
            return facts
        # capture TTL hint like "for N days"
        self._last_ttl_days = None
        mttl = re.search(r"\bfor\s+(\d+)\s+day(s)?\b", u, flags=re.I)
        if mttl:
            try:
                self._last_ttl_days = int(mttl.group(1))
            except Exception:
                self._last_ttl_days = None
        m = re.search('\\bremember that\\b\\s*(?P<val>.+)', u, flags=re.I)
        if m:
            fact = m.group('val').strip().rstrip('.')
            if fact:
                facts.append(fact if fact.endswith('.') else fact + '.')
                self._last_scope = 'long'
                return facts
        m2 = re.search('\\bsave this to long[- ]?term memory\\b', u, flags=re.I)
        if m2:
            after = u[m2.end():].strip()
            if after:
                facts.append(after if after.endswith('.') else after + '.')
                self._last_scope = 'long'
                return facts
        scope = 'short'
        for (rx, builder) in self._patterns:
            mm = rx.search(u)
            if mm:
                try:
                    built = builder(mm)
                    if built:
                        facts.append(built if built.endswith('.') else built + '.')
                except Exception:
                    pass
        mm2 = re.search("\\bi (?:like|love) ([a-z0-9 ,.'-]+)", u, re.I)
        if mm2:
            val = mm2.group(1).strip()
            if val:
                facts.append(f'User likes {val}.')
        if facts:
            self._last_scope = scope
        return facts

    def consider_storing(self, utterance: str) -> List[str]:
        facts = self.extract_from_utterance(utterance)
        scope = getattr(self, '_last_scope', 'short')
        # naive tag inference
        def _infer_tags(s: str) -> List[str]:
            low = s.lower()
            tags: List[str] = []
            if any(w in low for w in ["pizza", "food", "restaurant", "cook", "eat", "coffee", "tea", "lunch", "dinner"]):
                tags.append("food")
            if any(w in low for w in ["python", "code", "git", "refactor", "test", "build", "deploy", "tool"]):
                tags.append("tools")
            if any(w in low for w in ["work", "job", "office", "meeting", "project"]):
                tags.append("work")
            if any(w in low for w in ["like", "love", "enjoy"]):
                tags.append("positive")
            if any(w in low for w in ["dislike", "hate", "annoyed", "angry"]):
                tags.append("negative")
            return tags
        ttl_hint = getattr(self, '_last_ttl_days', None)
        stored: List[str] = []
        for idx, f in enumerate(facts):
            ttl = ttl_hint if (ttl_hint is not None and idx == len(facts) - 1) else None
            if self.add_fact(
                f,
                tags=_infer_tags(f),
                scope=scope,
                provenance='utterance',
                confidence=0.75,
                ttl_days=ttl,
            ):
                stored.append(f)
        if stored:
            self.last_learned = stored[-1]
            self.last_ref = self.last_learned
        return stored
