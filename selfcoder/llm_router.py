"""Task-aware, local-first LLM router for Nerion.

Responsibility:
- Decide coder/chat model family based on task and target language.
- Prefer locally available backends/models; fall back deterministically.
- Set environment vars consumed by downstream components.

Environment knobs:
- NERION_ROUTER_VERBOSE=1 → print routing decision
- NERION_AUTOPULL=1 → attempt to provision missing models (best-effort)
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import os
import re


TS_EXT = {".ts", ".tsx"}
JS_EXT = {".js", ".jsx", ".mjs", ".cjs"}


def _ext_of(path: Optional[str]) -> str:
    if not path:
        return ""
    p = str(path).lower().strip()
    m = re.search(r"(\.[a-z0-9]+)$", p)
    return m.group(1) if m else ""


def _env_true(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _available_models() -> List[Tuple[str, str, Optional[str]]]:
    """Return a list of (backend, model, base_url?) candidates detected locally."""
    out: List[Tuple[str, str, Optional[str]]] = []
    try:
        from app.parent.selector import probe_ollama as _oll, probe_vllm as _vllm
        base_oll = (os.getenv("NERION_CODER_BASE_URL") or "http://localhost:11434").rstrip("/")
        for m in (_oll() or []):
            out.append(("ollama", str(m), base_oll))
        for m in (_vllm() or []):
            out.append(("vllm", str(m), os.getenv("NERION_CODER_BASE_URL")))
    except Exception:
        pass
    # llama_cpp / exllamav2 are path-based; do not enumerate models here.
    return out


def _pick_first_present(candidates: List[str], avail: List[Tuple[str, str, Optional[str]]]) -> Tuple[str, str, Optional[str]]:
    """Pick the first candidate family that exists in avail; otherwise fallback to first candidate with default backend."""
    # Normalize to family names for matching
    fams = [c.lower() for c in candidates]
    for fam in fams:
        for be, model, base in avail:
            ml = model.lower()
            if fam in ml:
                return be, model, base
    # Fallback: prefer ollama for known families
    primary = candidates[0]
    be = os.getenv("NERION_CODER_BACKEND") or "ollama"
    base = os.getenv("NERION_CODER_BASE_URL") or ("http://localhost:11434" if be == "ollama" else None)
    return be, primary, base

def _lang_from_ext(ext: str) -> str:
    if ext in TS_EXT:
        return 'ts'
    if ext in JS_EXT:
        return 'js'
    return 'py' if ext == '.py' else (ext.lstrip('.') or '')


def _maybe_autopull(backend: str, model: str) -> None:
    if not _env_true("NERION_AUTOPULL"):
        return
    # Only attempt when network is explicitly allowed
    if not _env_true("NERION_ALLOW_NETWORK"):
        return
    try:
        from app.parent.provision import ensure_available
        ok, _msg = ensure_available(backend, model)
        if _env_true("NERION_ROUTER_VERBOSE"):
            print(f"[router] autopull {backend}:{model} → {'OK' if ok else 'skip'}")
    except Exception:
        pass


def _log_decision(kind: str, payload: dict) -> None:
    try:
        if not _env_true("NERION_ROUTER_LOG"):
            return
        import json
        import time
        import pathlib
        root = pathlib.Path('.nerion')
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'router_log.jsonl'
        rec = {"ts": int(time.time()), "kind": kind, **payload}
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def apply_router_env(*, instruction: Optional[str], file: Optional[str], task: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    """Decide and set env for coder/chat models; return (backend, model, base).

    - task: 'chat' routes to NERION_LLM_MODEL; others route to coder env.
    - file extension determines coder family for JS/TS.
    - prefers locally detected models; optional autopull when enabled.
    """
    t = (task or "").strip().lower()
    ext = _ext_of(file)
    ins = (instruction or "").strip().lower()

    if t == "chat":
        model = os.getenv("NERION_LLM_MODEL") or "deepseek-r1:14b"
        os.environ.setdefault("NERION_LLM_MODEL", model)
        if _env_true("NERION_ROUTER_VERBOSE"):
            print(f"[router] task=chat model={model}")
        _log_decision("chat", {"model": model})
        # For chat we don't set coder backend; return sentinel
        return ("ollama", model, os.getenv("NERION_CODER_BASE_URL"))

    # Code tasks: choose coder family based on ext/text
    if ext in TS_EXT or ("typescript" in ins and ext in JS_EXT):
        prefs = ["qwen2.5-coder", "deepseek-coder-v2", "starcoder2", "codellama"]
    elif ext in JS_EXT:
        prefs = ["qwen2.5-coder", "deepseek-coder-v2", "starcoder2", "codellama"]
    else:
        prefs = ["deepseek-coder-v2", "qwen2.5-coder", "starcoder2", "codellama"]

    avail = _available_models()
    backend, model, base = _pick_first_present(prefs, avail)

    # Apply env for coder
    os.environ.setdefault("NERION_CODER_BACKEND", backend)
    os.environ.setdefault("NERION_CODER_MODEL", model)
    if base:
        os.environ.setdefault("NERION_CODER_BASE_URL", base)
    # Allow localhost model access by default for coder actions
    os.environ.setdefault("NERION_ALLOW_NETWORK", "1")

    if _env_true("NERION_ROUTER_VERBOSE"):
        print(f"[router] task=code lang={_lang_from_ext(ext)} model={model}")
    _log_decision("code", {"backend": backend, "model": model, "base": base, "ext": ext})

    _maybe_autopull(backend, model)
    return backend, model, base
