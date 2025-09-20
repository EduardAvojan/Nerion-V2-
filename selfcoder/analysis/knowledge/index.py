import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Path to the index file
INDEX_PATH = Path("out/knowledge/index.json")
CHUNKS_DIR = Path("out/knowledge/chunks")

ROLLUP_DIR = Path("out/knowledge/rollups")

def _slugify(s: str) -> str:
    import re as _re
    s = (s or "").strip().lower()
    s = _re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "topic"

# Default TTLs (in days) for common topics/domains
DEFAULT_TTLS_DAYS = {
    "site_overview": 180,
    "monthly_best": 90,
    "product_page": 120,
}

def _rollup_threshold_default() -> int:
    try:
        v = int(os.environ.get("NERION_ROLLUP_THRESHOLD", "5"))
        return max(1, v)
    except Exception:
        return 5

def _to_seconds_days(days: int) -> int:
    return int(days) * 86400

def _infer_ttl_for_topic(topic: Optional[str]) -> Optional[int]:
    """Infer a TTL (in seconds) from a topic name when no explicit policy is given.
    Rules:
      - topics containing 'monthly' -> 90d
      - topics starting with 'site_overview' -> 180d
      - fallback -> 120d
    Returns None if topic is empty.
    """
    if not topic:
        return None
    t = str(topic).strip().lower()
    if "monthly" in t:
        return _to_seconds_days(DEFAULT_TTLS_DAYS.get("monthly_best", 90))
    if t.startswith("site_overview"):
        return _to_seconds_days(DEFAULT_TTLS_DAYS.get("site_overview", 180))
    # fallback
    return _to_seconds_days(DEFAULT_TTLS_DAYS.get("product_page", 120))

def _parse_ttl_spec(spec: Any) -> Optional[int]:
    """Accept int(seconds) or strings like '90d', '6m', '1y' and return seconds."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return int(spec)
    if isinstance(spec, str):
        s = spec.strip().lower()
        import re
        m = re.match(r"^(\d+)([dmy])$", s)
        if not m:
            # If plain integer string, treat as seconds
            try:
                return int(s)
            except Exception:
                return None
        n, unit = int(m.group(1)), m.group(2)
        if unit == 'd':
            return n * 86400
        if unit == 'm':
            return n * 30 * 86400
        if unit == 'y':
            return n * 365 * 86400
    return None

def load_index() -> List[Dict[str, Any]]:
    """Load the artifact index from the JSON file. Returns a list of entries."""
    if not INDEX_PATH.exists():
        return []
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def save_index(data: List[Dict[str, Any]]) -> None:
    """Save the artifact index to the JSON file."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def append_entry(entry: Dict[str, Any]) -> None:
    """Append a new artifact entry to the index."""
    data = load_index()
    # Optionally, enforce required fields
    # e.g., topic, domain, query, date, artifact_path, confidence
    entry = entry.copy()
    if "date" not in entry:
        entry["date"] = int(time.time())
    data.append(entry)
    save_index(data)


def append_chunk(entry: Dict[str, Any]) -> Path:
    """Persist a small chunk record for later RAG indexing.

    Expected keys (free-form tolerated): topic, domain, url, extract, date.
    Returns the path written.
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    topic = _slugify(entry.get("topic") or "")
    fn = f"{ts}_{topic or 'chunk'}.json"
    out = CHUNKS_DIR / fn
    blob = dict(entry)
    if "date" not in blob:
        blob["date"] = ts
    with out.open("w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    return out


def search_chunks(topic: Optional[str] = None, domain: Optional[str] = None, contains: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Search chunk files by topic/domain substring and free-text contains in extract/url.
    Returns a list of chunk dicts (path, topic, domain, url, extract, date).
    """
    results: list[dict] = []
    if not CHUNKS_DIR.exists():
        return results
    t_sub = (topic or '').strip().lower()
    d_sub = (domain or '').strip().lower()
    q_sub = (contains or '').strip().lower()
    for fp in sorted(CHUNKS_DIR.glob('*.json'), reverse=True):
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        t = str(data.get('topic') or '').lower()
        d = str(data.get('domain') or '').lower()
        u = str(data.get('url') or '')
        ex = str(data.get('extract') or '')
        if t_sub and t_sub not in t:
            continue
        if d_sub and d_sub not in d:
            continue
        if q_sub and (q_sub not in ex.lower()) and (q_sub not in u.lower()):
            continue
        results.append({
            'path': fp.as_posix(),
            'topic': data.get('topic'),
            'domain': data.get('domain'),
            'url': data.get('url'),
            'extract': data.get('extract'),
            'date': data.get('date'),
        })
        if len(results) >= max(1, int(limit)):
            break
    return results


def _tokenize(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[a-zA-Z0-9]+", (text or '').lower()) if len(t) > 1]


def semantic_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Simple TF‑IDF‑ish semantic search over chunk extracts (offline, no deps)."""
    docs: List[Dict[str, Any]] = []
    if not CHUNKS_DIR.exists():
        return []
    for fp in CHUNKS_DIR.glob('*.json'):
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        extract = str(data.get('extract') or '')
        tokens = _tokenize(extract)
        if not tokens:
            continue
        docs.append({'path': fp.as_posix(), 'data': data, 'tokens': tokens})
    if not docs:
        return []
    import math
    # IDF
    df = {}
    for d in docs:
        for t in set(d['tokens']):
            df[t] = df.get(t, 0) + 1
    N = len(docs)
    idf = {t: math.log((N+1)/(c+1)) + 1.0 for t, c in df.items()}
    q_tokens = _tokenize(query)
    q_weights = {}
    for t in q_tokens:
        q_weights[t] = q_weights.get(t, 0.0) + idf.get(t, 0.0)
    scores: List[tuple[float, Dict[str, Any]]] = []
    for d in docs:
        w = {}
        for t in d['tokens']:
            w[t] = w.get(t, 0.0) + idf.get(t, 0.0)
        # Cosine similarity
        dot = sum(w.get(t, 0.0) * q_weights.get(t, 0.0) for t in set(list(w.keys()) + list(q_weights.keys())))
        norm_d = math.sqrt(sum((v*v) for v in w.values())) or 1.0
        norm_q = math.sqrt(sum((v*v) for v in q_weights.values())) or 1.0
        sim = dot / (norm_d * norm_q)
        scores.append((sim, d))
    scores.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for sim, d in scores[:max(1, limit)]:
        data = d['data']
        out.append({
            'path': d['path'],
            'topic': data.get('topic'),
            'domain': data.get('domain'),
            'url': data.get('url'),
            'extract': data.get('extract'),
            'score': float(sim),
            'date': data.get('date'),
        })
    return out


def prune_topic(topic: str, ttl_spec: Any) -> None:
    ttl_seconds = _parse_ttl_spec(ttl_spec)
    if ttl_seconds is None:
        return
    now = int(time.time())
    data = load_index()
    kept = []
    for entry in data:
        if entry.get("topic") == topic:
            if now - int(entry.get("date", 0)) > ttl_seconds:
                continue
        kept.append(entry)
    save_index(kept)


def prune_domain(domain: str, ttl_spec: Any) -> None:
    ttl_seconds = _parse_ttl_spec(ttl_spec)
    if ttl_seconds is None:
        return
    now = int(time.time())
    data = load_index()
    kept = []
    for entry in data:
        if entry.get("domain") == domain:
            if now - int(entry.get("date", 0)) > ttl_seconds:
                continue
        kept.append(entry)
    save_index(kept)


def prune(ttl_policies: Dict[str, Any]) -> None:
    """
    Prune old entries based on TTL policies.
    ttl_policies: dict mapping topic or domain to TTL (in seconds or string spec).
    If no policy matches, an inferred TTL is applied from the topic name
    (e.g., '*monthly' -> 90d, 'site_overview*' -> 180d, fallback -> 120d).
    """
    now = int(time.time())
    data = load_index()
    pruned: List[Dict[str, Any]] = []
    removed_by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        entry_time = int(entry.get("date", 0))
        ttl_seconds: Optional[int] = None
        topic = entry.get("topic")
        domain = entry.get("domain")
        if topic and topic in ttl_policies:
            ttl_seconds = _parse_ttl_spec(ttl_policies[topic])
        elif domain and domain in ttl_policies:
            ttl_seconds = _parse_ttl_spec(ttl_policies[domain])
        if ttl_seconds is None:
            ttl_seconds = _infer_ttl_for_topic(topic)
        # Drop if expired
        if ttl_seconds is not None and now - entry_time > ttl_seconds:
            if topic:
                removed_by_topic.setdefault(topic, []).append(entry)
            continue
        pruned.append(entry)
    save_index(pruned)
    # Write rollups for topics with many deletions
    ROLLUP_THRESHOLD = _rollup_threshold_default()
    for t, removed in removed_by_topic.items():
        if len(removed) >= ROLLUP_THRESHOLD:
            try:
                write_rollup(t, removed)
            except Exception:
                pass

def search(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Search/filter entries by topic, domain, or query.
    filters: dict with optional keys 'topic', 'domain', 'query'.
    """
    data = load_index()
    if not filters:
        return data
    result = []
    for entry in data:
        match = True
        for key in ("topic", "domain", "query"):
            if key in filters and filters[key] is not None:
                if entry.get(key) != filters[key]:
                    match = False
                    break
        if match:
            result.append(entry)
    return result


def write_rollup(topic: str, removed_entries: List[Dict[str, Any]]) -> Optional[Path]:
    """Write a small rollup.json capturing a winner/headline timeline for a topic.
    Best-effort: loads each artifact JSON and extracts result.consensus.winner.name or result.headline.
    Returns the path if written, else None.
    """
    if not removed_entries:
        return None
    outdir = ROLLUP_DIR / _slugify(topic)
    outdir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for e in sorted(removed_entries, key=lambda x: int(x.get("date", 0))):
        ap = e.get("artifact_path")
        winner = None
        headline = None
        conf = None
        if ap and Path(ap).exists():
            try:
                with open(ap, "r", encoding="utf-8") as f:
                    art = json.load(f)
                res = art.get("result", {})
                cons = res.get("consensus", {})
                w = cons.get("winner", {})
                winner = w.get("name") or None
                headline = res.get("headline") or res.get("conclusion") or None
                conf = res.get("confidence")
            except Exception:
                pass
        items.append({
            "date": int(e.get("date", 0)),
            "artifact_path": ap,
            "winner": winner,
            "headline": headline,
            "confidence": conf,
        })
    rollup_path = outdir / "rollup.json"
    with rollup_path.open("w", encoding="utf-8") as f:
        json.dump({"topic": topic, "items": items}, f, ensure_ascii=False, indent=2)
    return rollup_path


# Optional convenience
def prune_with_defaults() -> None:
    """Apply default TTLs for known domains/topics where applicable."""
    policies = {k: _to_seconds_days(v) for k, v in DEFAULT_TTLS_DAYS.items()}
    prune(policies)
