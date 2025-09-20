from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional


def load_catalog() -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    root = Path(__file__).resolve().parents[2]
    cat = root / "config" / "model_catalog.yaml"
    if not cat.exists():
        return {}
    try:
        return yaml.safe_load(cat.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def resolve(backend: str, model: str) -> Optional[Dict[str, Any]]:
    data = load_catalog()
    sect = (data.get("catalog") or {}).get(backend) or {}
    if model in sect:
        return sect.get(model)
    ml = model.lower()
    for k, v in sect.items():
        if k in ml or ml in k:
            return v
    return None

