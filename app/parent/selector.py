"""Auto-select a local coder backend + model based on availability.

Respects offline-first: only queries local endpoints; no downloads unless
NERION_ALLOW_NETWORK=1 and explicit pull is requested by env.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os


PREFERRED_ORDER = [
    "deepseek-coder",  # covers deepseek-coder-v2
    "qwen2.5-coder",
    "starcoder",
    "codellama",
    "codegemma",
]


def _env_bool(name: str) -> bool:
    v = os.getenv(name) or ""
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _ollama_base() -> str:
    return (os.getenv("NERION_CODER_BASE_URL") or "http://localhost:11434").rstrip("/")


def probe_ollama() -> List[str]:
    try:
        import requests  # type: ignore
        r = requests.get(_ollama_base() + "/api/tags", timeout=2)
        if not r.ok:
            return []
        data = r.json() or {}
        models = [m.get("name") for m in (data.get("models") or []) if isinstance(m, dict)]
        return [m for m in models if isinstance(m, str)]
    except Exception:
        return []


def probe_vllm() -> List[str]:
    base = os.getenv("NERION_CODER_BASE_URL") or ""
    if not base:
        return []
    try:
        import requests  # type: ignore
        r = requests.get(base.rstrip("/") + "/v1/models", timeout=2)
        if not r.ok:
            return []
        data = r.json() or {}
        return [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict) and isinstance(m.get("id"), str)]
    except Exception:
        return []


def probe_llama_cpp() -> List[str]:
    p = os.getenv("LLAMA_CPP_MODEL_PATH") or ""
    return [Path(p).name] if p and Path(p).exists() else []


def probe_exllamav2() -> List[str]:
    d = os.getenv("EXLLAMA_MODEL_DIR") or ""
    return [Path(d).name] if d and Path(d).exists() else []


def _score(model_name: str) -> int:
    m = model_name.lower()
    for i, pref in enumerate(PREFERRED_ORDER):
        if pref in m:
            return len(PREFERRED_ORDER) - i
    return 0


def auto_select_model() -> Optional[Tuple[str, str, Optional[str]]]:
    """Return (backend, model, base_url?) if any backend is ready; else None.

    Selection order prefers models with names matching PREFERRED_ORDER across
    all available backends. User can override priority via NERION_PREFERRED_MODELS
    (comma-separated, highest first).
    """
    preferred = [s.strip().lower() for s in (os.getenv("NERION_PREFERRED_MODELS") or "").split(",") if s.strip()]
    if preferred:
        order = preferred + [p for p in PREFERRED_ORDER if p not in preferred]
    else:
        order = PREFERRED_ORDER

    avail: Dict[str, List[str]] = {}
    oll = probe_ollama()
    if oll:
        avail["ollama"] = oll
    vllm = probe_vllm()
    if vllm:
        avail["vllm"] = vllm
    llcpp = probe_llama_cpp()
    if llcpp:
        avail["llama_cpp"] = llcpp
    exv2 = probe_exllamav2()
    if exv2:
        avail["exllamav2"] = exv2

    if not avail:
        return None

    # Rank all candidates by preference
    best: Optional[Tuple[str, str]] = None
    best_score = -1
    for be, models in avail.items():
        for m in models:
            sc = 0
            ml = m.lower()
            for i, pref in enumerate(order):
                if pref in ml:
                    sc = len(order) - i
                    break
            if sc > best_score:
                best = (be, m)
                best_score = sc
    if not best:
        # Choose any deterministic item
        be = sorted(avail.keys())[0]
        m = sorted(avail[be])[0]
        best = (be, m)
    backend, model = best
    base = os.getenv("NERION_CODER_BASE_URL") if backend in {"ollama", "vllm"} else None
    if backend == "ollama" and not base:
        base = _ollama_base()
    return (backend, model, base)

