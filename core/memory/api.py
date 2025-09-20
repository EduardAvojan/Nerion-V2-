import json
import hashlib
from threading import RLock
from pathlib import Path
from ops.security.fs_guard import ensure_in_repo_auto
from ops.security.io_safe import write_text

# Resolve DB path inside the repository root and jail it
# (defense-in-depth: prevents writes outside the repo)
def _db_path():
    return ensure_in_repo_auto(Path("memory_db.json"))

_DB_LOCK = RLock()

def _load_db():
    with _DB_LOCK:
        p = _db_path()
        try:
            with open(p, "r", encoding="utf-8") as f:
                db = json.load(f)
        except FileNotFoundError:
            # Do not create-on-read; return empty structure
            return {"facts": []}
        except Exception:
            return {"facts": []}
        # Handle foreign formats:
        if isinstance(db, list):
            return {"facts": [], "_foreign_format": "list"}
        elif not isinstance(db, dict):
            return {"facts": []}
        if "facts" not in db:
            db["facts"] = []
        return db

def _save_db(db):
    with _DB_LOCK:
        p = _db_path()
        # If db is a dict with a _foreign_format key, skip writing (no-op)
        if isinstance(db, dict) and db.get("_foreign_format"):
            return
        write_text(p, json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

def _fact_hash(entity, attribute, value):
    h = hashlib.sha256()
    h.update(str(entity).encode("utf-8"))
    h.update(str(attribute).encode("utf-8"))
    h.update(str(value).encode("utf-8"))
    return h.hexdigest()

def remember_fact(entity, attribute, value, **meta):
    """
    Store a structured fact, deduplicated by hash of (entity, attribute, value).
    """
    db = _load_db()
    fact_hash = _fact_hash(entity, attribute, value)
    for fact in db["facts"]:
        if fact.get("hash") == fact_hash:
            # Already exists, optionally update meta
            fact.update(meta)
            _save_db(db)
            return fact
    fact = {
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "hash": fact_hash,
    }
    if meta:
        fact.update(meta)
    db["facts"].append(fact)
    _save_db(db)
    return fact

def recall(entity=None, attribute=None, value=None, limit=10):
    """
    Retrieve matching facts.
    """
    db = _load_db()
    facts = db["facts"]
    def match(f):
        if entity is not None and f.get("entity") != entity:
            return False
        if attribute is not None and f.get("attribute") != attribute:
            return False
        if value is not None and f.get("value") != value:
            return False
        return True
    results = [f for f in facts if match(f)]
    return results[:limit]

def forget(predicate):
    """
    Remove facts matching a predicate function(fact)->bool.
    """
    db = _load_db()
    before = len(db["facts"])
    db["facts"] = [f for f in db["facts"] if not predicate(f)]
    after = len(db["facts"])
    _save_db(db)
    return before - after