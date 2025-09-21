"""
Parent executor bindings for the chat engine.
"""
from __future__ import annotations
from typing import Callable, Optional, Dict, Any
from urllib.parse import urlparse as _urlparse

from selfcoder.analysis.adapters import engine as _sq_engine
from selfcoder.analysis.search_api import search_enriched as _search_enriched
from app.parent.executor import make_default_executor, ParentExecutor
from app.parent.tools_manifest import load_tools_manifest_from_yaml
from app.chat.offline_tools import run_healthcheck as _run_healthcheck, run_diagnostics as _diag_offline
from ops.security import fs_guard as _fs_guard
import json
import os
try:
    from ops.security.safe_subprocess import safe_run as _safe_run
except Exception:
    _safe_run = None

EnsureNetCB = Callable[[str, Optional[str]], bool]
GetHeardCB = Callable[[], str]
ParseSlotsCB = Callable[[str], Optional[Dict[str, Any]]]

def build_executor(*, ensure_network_for: EnsureNetCB, get_heard: GetHeardCB, parse_task_slots: ParseSlotsCB, metrics_hook: Optional[Callable[[str, bool, float, Optional[str]], None]] = None, progress_hook: Optional[Callable[[int, int, str], None]] = None, cancel_check: Optional[Callable[[], bool]] = None) -> ParentExecutor:
    def _run_site_query_from_parent(url: str, **kwargs):
        if not ensure_network_for('site_query', url):
            raise PermissionError("network not allowed for site_query")
        host = _urlparse(url).netloc
        profile_key = f"{host}:{url.lower()}"
        pol = {"render": True, "timeout": 15, "render_timeout": 7, "depth": 2, "max_pages": 6,
               "augment": False, "external_allow": [], "fresh_within": "60d", "max_external": 4}
        prof = _sq_engine.store_mod.load_or_create(profile_key, seed_context={"query": url, "url": url, **pol})
        prof.setdefault("source_policies", {}).update(pol)
        _sq_engine.store_mod.save(prof)
        out = _sq_engine.run(profile_key, query=url, url=url) or {}
        return {"url": url, "artifact_path": out.get("artifact_path")}

    def _run_web_search_from_parent(query: Optional[str] = None, **kwargs):
        q = (query or get_heard() or "").strip()
        if not ensure_network_for('web_search', None):
            raise PermissionError("network not allowed for web_search")
        ts_hint = (parse_task_slots(q) or {})
        tf = (ts_hint.get('timeframe') or '').lower()
        freshness = None
        if 'today' in tf or 'tomorrow' in q.lower():
            freshness = 'day'
        elif 'week' in tf:
            freshness = 'week'
        elif 'month' in tf:
            freshness = 'month'
        enriched = _search_enriched(q, n=5, freshness=freshness, allow=ts_hint.get('sources')) or {}
        urls = enriched.get('urls', [])
        return {"query": q, "urls": urls}

    def _run_rename_symbol_from_parent(old: str, new: str, simulate: bool = True, **kwargs):
        return {"rename": {"old": old, "new": new, "simulate": bool(simulate)}}

    def _run_healthcheck_from_parent(**kwargs):
        # Offline system healthcheck (no network). Return a concise string.
        try:
            return _run_healthcheck(None)
        except Exception as e:
            return f"Healthcheck failed: {e}"

    def _run_diagnostics_from_parent(**kwargs):
        return _diag_offline(None)

    def _list_plugins_from_parent(**kwargs):
        try:
            p = os.path.join("plugins", "allowlist.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {"plugins": data}
            if os.path.isdir("plugins"):
                names = [n for n in os.listdir("plugins") if n.endswith('.py') and n != '__init__.py']
                return {"plugins": names}
        except Exception as e:
            return {"error": str(e)}
        return {"plugins": []}

    def _run_pytest_smoke_from_parent(**kwargs):
        if _safe_run is None:
            return {"error": "safe_run unavailable"}
        try:
            r = _safe_run(["pytest", "-q", "-k", "smoke", "--maxfail=1"], capture_output=True, timeout=120, check=False)
            ok = r.returncode == 0
            out = (r.stdout or b"").decode(errors='ignore')
            err = (r.stderr or b"").decode(errors='ignore')
            last = next((ln for ln in reversed(out.splitlines()) if ln.strip()), "")
            return {"ok": ok, "summary": last[:200], "errors": err[:400]}
        except Exception as e:
            return {"error": str(e)}

    def _read_file_from_parent(path: str, **kwargs):
        try:
            safe_p = _fs_guard.ensure_in_repo_auto(path)
            data = safe_p.read_text(encoding='utf-8', errors='ignore')
            if len(data) > 100000:
                data = data[:100000] + "\n… (truncated)"
            return {"path": safe_p.as_posix(), "content": data}
        except Exception as e:
            return {"error": str(e)}

    def _summarize_file_from_parent(path: str, **kwargs):
        try:
            safe_p = _fs_guard.ensure_in_repo_auto(path)
            text = safe_p.read_text(encoding='utf-8', errors='ignore')
            snippet = text[:8000]
            # Try local LLM summary; fall back to a heuristic
            try:
                if str(os.getenv('NERION_SUMMARY_LLM', '')).strip().lower() in {'0','false','no'}:
                    raise RuntimeError('summary LLM disabled by env')
                registry = get_registry()
                prompt = "Summarize the following file in 5 bullets with concrete facts."
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": snippet},
                ]
                resp = registry.generate(role='chat', messages=messages, temperature=0.2)
                out = resp.text or ''
            except (ProviderNotConfigured, ProviderError, RuntimeError):
                lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
                out = "\n".join(["- " + ln for ln in lines[:5]])
            return {"path": safe_p.as_posix(), "summary": out}
        except Exception as e:
            return {"error": str(e)}

    # Load allowed tools from config/tools.yaml
    # Note: keep local, offline-safe utilities always allowed to avoid breaking
    # tests and basic UX when the manifest omits them.
    try:
        manifest = load_tools_manifest_from_yaml("config/tools.yaml")
        allowed_set = {t.name for t in manifest.tools}
        # Always-allow a baseline of safe, offline tools
        allowed_set.update({
            "read_file",
            "summarize_file",
            "run_healthcheck",
            "run_diagnostics",
            "list_plugins",
            "run_pytest_smoke",
        })
        allowed = sorted(allowed_set)
    except Exception:
        allowed = None

    return make_default_executor(
        read_url=_run_site_query_from_parent,
        web_search=_run_web_search_from_parent,
        rename_symbol=_run_rename_symbol_from_parent,
        run_healthcheck=_run_healthcheck_from_parent,
        run_diagnostics=_run_diagnostics_from_parent,
        list_plugins=_list_plugins_from_parent,
        run_pytest_smoke=_run_pytest_smoke_from_parent,
        read_file=_read_file_from_parent,
        summarize_file=_summarize_file_from_parent,
        ensure_network=lambda required: ensure_network_for('web_search', None) if required else None,
        allowed_tools=allowed,
        metrics_hook=metrics_hook,
        progress_hook=progress_hook,
        cancel_check=cancel_check,
    )
