from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# -----------------------------
# Generic, domain-agnostic adapter profile store
# Profiles are simple JSON files on disk (no extra deps required).
# -----------------------------

_DEF_STORE_ENV = "NERION_ADAPTER_DIR"
_DEF_STORE_HOME = Path.home() / ".nerion" / "adapters"
_DEF_STORE_ALT = Path("out/knowledge/adapters")


def _store_dir() -> Path:
    # Priority: env var -> ~/.nerion/adapters -> repo-local out/knowledge/adapters
    p = os.environ.get(_DEF_STORE_ENV)
    if p:
        d = Path(p)
    else:
        d = _DEF_STORE_HOME if _DEF_STORE_HOME.parent.exists() else _DEF_STORE_ALT
    d.mkdir(parents=True, exist_ok=True)
    return d


_slug_re = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    s = (s or "").lower()
    s = _slug_re.sub("-", s).strip("-")
    return s or f"profile-{int(time.time())}"


def _now_iso() -> str:
    # UTC ISO without microseconds
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _extract_seed_signals(seed: Dict[str, Any]) -> List[str]:
    signals: List[str] = []
    query = (seed.get("query") or "").lower()
    url = (seed.get("url") or "").lower()
    # query keywords
    for w in re.findall(r"[a-zA-Z]{3,}", query):
        if w not in signals:
            signals.append(w)
    # host tokens from url
    if url:
        host = urlparse(url).netloc.split(":")[0]
        for part in host.split("."):
            if part and part not in signals:
                signals.append(part)
    return signals[:32]


def _profile_path(profile_id: str) -> Path:
    return _store_dir() / f"{profile_id}.json"


def load(profile_id: str) -> Optional[Dict[str, Any]]:
    path = _profile_path(profile_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save(profile: Dict[str, Any]) -> Path:
    profile = dict(profile)
    if "profile_id" not in profile:
        raise ValueError("profile missing 'profile_id'")
    path = _profile_path(profile["profile_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_or_create(profile_key: str, *, seed_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a profile by a stable key; create a proto-profile on first use using seed context.
    - profile_key: any stable string (e.g., f"{domain}:{host}:{normalized_query}")
    - seed_context: {"query": str, "url": str, ...} used to initialize signals/policies
    """
    profile_id = _slugify(profile_key)
    existing = load(profile_id)
    if existing:
        return existing

    seed = seed_context or {}
    signals = _extract_seed_signals(seed)

    now = _now_iso()
    profile = {
        "profile_id": profile_id,
        "name": seed.get("name") or profile_id,
        "created_at": now,
        "last_used": now,
        "seed_signals": signals,              # keywords/host tokens discovered from prompt + url
        "source_policies": {
            "same_domain_only": bool(seed.get("same_domain_only", True)),
            "depth": int(seed.get("depth", 1)),
            "max_pages": int(seed.get("max_pages", 6)),
            "render": bool(seed.get("render", False)),
            "timeout": int(seed.get("timeout", 10)),
            "render_timeout": int(seed.get("render_timeout", 5)),
        },
        "scoring": {
            "recency": 0.30,
            "authority": 0.25,
            "agreement": 0.20,
            "onsite_signal": 0.15,
            "coverage": 0.10,
        },
        "metrics": {
            "runs": 0,
            "avg_confidence": None,
            "last_confidence": None,
        },
    }
    save(profile)
    return profile


def touch(profile: Dict[str, Any], *, last_confidence: Optional[float] = None) -> Dict[str, Any]:
    p = dict(profile)
    p["last_used"] = _now_iso()
    m = p.setdefault("metrics", {})
    m["runs"] = int(m.get("runs", 0)) + 1
    if last_confidence is not None:
        # update running average safely
        prev_avg = m.get("avg_confidence")
        runs = m["runs"]
        if prev_avg is None:
            m["avg_confidence"] = float(last_confidence)
        else:
            # incremental average
            m["avg_confidence"] = float(prev_avg) + (float(last_confidence) - float(prev_avg)) / max(runs, 1)
        m["last_confidence"] = float(last_confidence)
    save(p)
    return p


__all__ = [
    "load_or_create",
    "save",
    "load",
    "touch",
]
