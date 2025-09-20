from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from selfcoder.analysis.adapters import engine as adapt_engine, store as adapt_store

from selfcoder.analysis import docs as docs_mod
from selfcoder.analysis.knowledge import index as kb_index
from core.ui.progress import progress
from core.ui.progress import cancelled
from selfcoder.config import allow_network
try:
    from selfcoder.policy.profile_resolver import decide as _decide_profile, apply_env_scoped as _apply_env_scoped
except Exception:
    _decide_profile = None
    _apply_env_scoped = None


def cmd_read(args: argparse.Namespace) -> int:
    if getattr(args, "url", None) and not allow_network():
        print(json.dumps({
            "error": "network disabled",
            "hint": "Set NERION_ALLOW_NETWORK=1 (or true/yes/on) to enable URL fetching.",
            "where": "docs read --url"
        }))
        return 1
    # Auto-apply profile (scoped)
    scope = None
    try:
        if _decide_profile and _apply_env_scoped:
            dec = _decide_profile('docs_read', signals={'network_intent': bool(getattr(args, 'url', None))})
            scope = _apply_env_scoped(dec)
    except Exception:
        scope = None
    try:
        with progress("Docs: read"):
            out = docs_mod.read_doc(
                args.path,
                url=getattr(args, "url", None),
                query=getattr(args, "query", None),
                timeout=getattr(args, "timeout", 10),
                render=bool(getattr(args, "render", False)),
                render_timeout=getattr(args, "render_timeout", 5),
                selector=getattr(args, "selector", None),
            )
    finally:
        if scope and hasattr(scope, '__exit__'):
            try:
                scope.__exit__(None, None, None)
            except Exception:
                pass
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    if getattr(args, "url", None) and not allow_network():
        print(json.dumps({
            "error": "network disabled",
            "hint": "Set NERION_ALLOW_NETWORK=1 (or true/yes/on) to enable URL fetching.",
            "where": "docs summarize --url"
        }))
        return 1
    scope = None
    try:
        if _decide_profile and _apply_env_scoped:
            dec = _decide_profile('docs_summarize', signals={'network_intent': bool(getattr(args, 'url', None))})
            scope = _apply_env_scoped(dec)
    except Exception:
        scope = None
    try:
        with progress("Docs: summarize.read"):
            doc = docs_mod.read_doc(
                args.path,
                url=getattr(args, "url", None),
                query=getattr(args, "query", None),
                timeout=getattr(args, "timeout", 10),
                render=bool(getattr(args, "render", False)),
                render_timeout=getattr(args, "render_timeout", 5),
                selector=getattr(args, "selector", None),
            )
    finally:
        if scope and hasattr(scope, '__exit__'):
            try:
                scope.__exit__(None, None, None)
            except Exception:
                pass
    with progress("Docs: summarize.build"):
        out = docs_mod.summarize_text(doc.get("raw_text") or doc["text"])
    if cancelled():
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_assimilate(args: argparse.Namespace) -> int:
    if getattr(args, "url", None) and not allow_network():
        print(json.dumps({
            "error": "network disabled",
            "hint": "Set NERION_ALLOW_NETWORK=1 (or true/yes/on) to enable URL fetching.",
            "where": "docs assimilate --url"
        }))
        return 1
    scope = None
    try:
        if _decide_profile and _apply_env_scoped:
            dec = _decide_profile('docs_assimilate', signals={'network_intent': bool(getattr(args, 'url', None))})
            scope = _apply_env_scoped(dec)
    except Exception:
        scope = None
    try:
        if bool(getattr(args, "render", False)):
            with progress("Docs: assimilate.render"):
                doc = docs_mod.read_doc(
                    args.path,
                    url=getattr(args, "url", None),
                    query=getattr(args, "query", None),
                    timeout=getattr(args, "timeout", 10),
                    render=True,
                    render_timeout=getattr(args, "render_timeout", 5),
                    selector=getattr(args, "selector", None),
                )
            with progress("Docs: assimilate.summarize"):
                summary = docs_mod.summarize_text(doc.get("raw_text") or doc["text"])
            if cancelled():
                return 1
            with progress("Docs: assimilate.persist"):
                bundle = {**doc, **summary}
                artifact = docs_mod.persist_assimilation(bundle)
                bundle["artifact_path"] = str(artifact)
            print(json.dumps(bundle, ensure_ascii=False, indent=2))
            return 0
        else:
            with progress("Docs: assimilate"):
                out = docs_mod.assimilate(args.path, url=getattr(args, "url", None), query=getattr(args, "query", None), timeout=getattr(args, "timeout", 10))
            if cancelled():
                return 1
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0
    finally:
        if scope and hasattr(scope, '__exit__'):
            try:
                scope.__exit__(None, None, None)
            except Exception:
                pass


def _persist_site_query(bundle: dict) -> str:
    outdir = Path("out/knowledge/site_queries")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    path = outdir / f"site_query_{ts}.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def cmd_site_query(args: argparse.Namespace) -> int:
    # Profile hint for network tasks
    try:
        from selfcoder.policy.profile_resolver import decide as _dec
        dec = _dec('docs_query', signals={'network_intent': True, 'domains_scoped': True})
        if dec and dec.name:
            print(f"[profile] hint: {dec.name} ({dec.why})")
    except Exception:
        pass
    url = args.url
    query = args.query
    if not url or not query:
        print(json.dumps({"error": "--url and --query are required"}))
        return 1

    # Master network gate
    if not allow_network():
        print(json.dumps({
            "error": "network disabled",
            "hint": "Set NERION_ALLOW_NETWORK=1 (or true/yes/on) to enable network (URL fetch / augmentation).",
            "where": "docs site-query"
        }))
        return 1

    # Helpers for CSV flags
    def _csv_opt(s: str | None) -> list[str]:
        if not s:
            return []
        return [p.strip() for p in s.split(",") if p.strip()]

    augment = bool(getattr(args, "augment", False))
    allow = _csv_opt(getattr(args, "allow", None))
    block = _csv_opt(getattr(args, "block", None))
    max_external = int(getattr(args, "max_external", 6))
    fresh_within = getattr(args, "fresh_within", "60d")
    persist_topic = getattr(args, "persist_topic", None)
    no_prune = bool(getattr(args, "no_prune", False))

    # Build a stable profile key from host + query
    host = (url or "").split("//", 1)[-1].split("/", 1)[0]
    profile_key = f"{host}:{query.strip().lower()}"

    # Seed/update profile policies from flags before running engine
    pol = {
        "render": bool(getattr(args, "render", False)),
        "timeout": int(getattr(args, "timeout", 10)),
        "render_timeout": int(getattr(args, "render_timeout", 5)),
        # Depth/max_pages are placeholders for future gather expansion
        "depth": int(getattr(args, "depth", 1)),
        "max_pages": int(getattr(args, "max_pages", 6)),
        "same_domain_only": True,
        "augment": augment,
        "external_allow": allow,
        "external_block": block,
        "max_external": max_external,
        "fresh_within": fresh_within,
        "persist_topic": persist_topic,
    }
    prof = adapt_store.load_or_create(
        profile_key,
        seed_context={
            "query": query,
            "url": url,
            **pol,
        },
    )
    # Ensure policies reflect current flags (overwrite defaults if needed)
    prof.setdefault("source_policies", {}).update(pol)
    adapt_store.save(prof)

    # Run the generic adapter engine
    # Auto-apply a network-aware profile (scoped)
    scope = None
    try:
        if _decide_profile and _apply_env_scoped:
            dec = _decide_profile('docs_query', signals={'network_intent': True, 'domains_scoped': True})
            scope = _apply_env_scoped(dec)
            print(f"[profile] hint: {dec.name} ({dec.why})")
    except Exception:
        scope = None
    try:
        with progress("Docs: site-query"):
            out = adapt_engine.run(profile_key, query=query, url=url)
        if cancelled():
            return 1
    finally:
        if scope and hasattr(scope, '__exit__'):
            try:
                scope.__exit__(None, None, None)
            except Exception:
                pass

    # Prepare a compact printable bundle & persist (prefer top-level citations; avoid duplication)
    result_clean = dict(out.get("result", {}))
    result_clean.pop("citations", None)
    top_citations = out.get("citations", []) or out.get("result", {}).get("citations", [])

    bundle = {
        "profile_id": out.get("profile_id"),
        "query": query,
        "url": url,
        "result": result_clean,
        "citations": top_citations,
        "augment": augment,
        "external_policies": {"allow": allow, "block": block, "max_external": max_external, "fresh_within": fresh_within},
        "persist_topic": persist_topic,
        "index": {"appended": True, "pruned": (not no_prune)},
    }
    bundle["artifact_path"] = _persist_site_query(bundle)

    # Append knowledge index entry (23e)
    try:
        result_conf = float(result_clean.get("confidence") or 0.0)
    except Exception:
        result_conf = 0.0
    kb_entry = {
        "topic": persist_topic or profile_key,
        "domain": "site_query",
        "query": query,
        "url": url,
        "artifact_path": bundle["artifact_path"],
        "confidence": result_conf,
        "date": int(time.time()),
    }
    try:
        kb_index.append_entry(kb_entry)
    except Exception:
        pass

    # Optional pruning with default TTLs (can be customized later)
    if not no_prune:
        TTL_90D = 90 * 86400
        try:
            kb_index.prune({persist_topic or profile_key: TTL_90D})
        except Exception:
            pass

    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    p = subparsers.add_parser("docs", help="Document assimilation commands")
    sp = p.add_subparsers(dest="docs_cmd", required=True)

    p_read = sp.add_parser("read", help="Read and normalize a document")
    grp_r = p_read.add_mutually_exclusive_group(required=True)
    grp_r.add_argument("--path", help="Path to document file")
    grp_r.add_argument("--url", help="HTTP/HTTPS URL to fetch (opt-in)")
    p_read.add_argument("--timeout", type=int, default=10, help="Network timeout in seconds (with --url)")
    p_read.add_argument("--render", action="store_true", help="Enable JS rendering for heavy sites (requires docs-web extras)")
    p_read.add_argument("--render-timeout", type=int, default=5, help="JS settle timeout in seconds (with --render)")
    p_read.add_argument("--selector", help="Optional CSS selector to wait for before snapshot")
    p_read.add_argument("--query", help="Optional intent text to improve domain classification (e.g., 'best laptop this month')")
    p_read.set_defaults(func=cmd_read)

    p_sum = sp.add_parser("summarize", help="Summarize a document")
    grp_s = p_sum.add_mutually_exclusive_group(required=True)
    grp_s.add_argument("--path", help="Path to document file")
    grp_s.add_argument("--url", help="HTTP/HTTPS URL to fetch (opt-in)")
    p_sum.add_argument("--timeout", type=int, default=10, help="Network timeout in seconds (with --url)")
    p_sum.add_argument("--render", action="store_true", help="Enable JS rendering for heavy sites (requires docs-web extras)")
    p_sum.add_argument("--render-timeout", type=int, default=5, help="JS settle timeout in seconds (with --render)")
    p_sum.add_argument("--selector", help="Optional CSS selector to wait for before snapshot")
    p_sum.add_argument("--query", help="Optional intent text to improve domain classification (e.g., 'recent research on ...')")
    p_sum.set_defaults(func=cmd_summarize)

    p_assim = sp.add_parser("assimilate", help="Assimilate and persist a document")
    grp_a = p_assim.add_mutually_exclusive_group(required=True)
    grp_a.add_argument("--path", help="Path to document file")
    grp_a.add_argument("--url", help="HTTP/HTTPS URL to fetch (opt-in)")
    p_assim.add_argument("--timeout", type=int, default=10, help="Network timeout in seconds (with --url)")
    p_assim.add_argument("--render", action="store_true", help="Enable JS rendering for heavy sites (requires docs-web extras)")
    p_assim.add_argument("--render-timeout", type=int, default=5, help="JS settle timeout in seconds (with --render)")
    p_assim.add_argument("--selector", help="Optional CSS selector to wait for before snapshot")
    p_assim.add_argument("--query", help="Optional intent text to improve domain classification (e.g., 'best ... this month')")
    p_assim.set_defaults(func=cmd_assimilate)

    p_sq = sp.add_parser("site-query", help="Run an adapter-driven site query (generic, no hard-coded domains)")
    p_sq.add_argument("--url", required=True, help="Root site URL to query")
    p_sq.add_argument("--query", required=True, help="Intent text (e.g., 'best laptop this month')")
    p_sq.add_argument("--timeout", type=int, default=10, help="Network timeout in seconds (with --url)")
    p_sq.add_argument("--render", action="store_true", help="Enable JS rendering for heavy sites (requires docs-web extras)")
    p_sq.add_argument("--render-timeout", type=int, default=5, help="JS settle timeout in seconds (with --render)")
    p_sq.add_argument("--selector", help="Optional CSS selector to wait for before snapshot (reserved for future use)")
    p_sq.add_argument("--depth", type=int, default=1, help="Crawl depth (future use; current gather is root only)")
    p_sq.add_argument("--max-pages", type=int, default=6, help="Max pages to gather (future use)")
    p_sq.add_argument("--augment", action="store_true", help="Opt-in: pull a small set of external pages for corroboration (generic, no domains in core)")
    p_sq.add_argument("--allow", help="Comma-separated domains/URLs to allow for external evidence (e.g., 'example.com,another.com')")
    p_sq.add_argument("--block", help="Comma-separated substrings to exclude from external URLs")
    p_sq.add_argument("--max-external", type=int, default=6, help="Max number of external pages to fetch when --augment is used")
    p_sq.add_argument("--fresh-within", default="60d", help="Only accept external pages with date hints within this window (e.g., '60d', '6m', '1y')")
    p_sq.add_argument("--persist-topic", help="Optional topic label to group related runs (e.g., 'best_printer_monthly')")
    p_sq.add_argument("--no-prune", action="store_true", help="Do not run retention pruning after this run")
    p_sq.set_defaults(func=cmd_site_query)

    # RAG search over local knowledge (chunks)
    p_search = sp.add_parser("search", help="Search local knowledge chunks/index (offline)")
    p_search.add_argument("--topic", help="topic substring filter")
    p_search.add_argument("--domain", help="domain substring filter")
    p_search.add_argument("--contains", help="free-text substring in extract/url")
    p_search.add_argument("--limit", type=int, default=20, help="max items to return")
    def _cmd_search(args: argparse.Namespace) -> int:
        items = kb_index.search_chunks(getattr(args, 'topic', None), getattr(args, 'domain', None), getattr(args, 'contains', None), int(getattr(args, 'limit', 20)))
        print(json.dumps({"items": items}, indent=2, ensure_ascii=False))
        return 0
    p_search.set_defaults(func=_cmd_search)

    # Semantic search
    p_sem = sp.add_parser("search-semantic", help="Semantic search over local chunks (offline TF‑IDF)")
    p_sem.add_argument("--q", required=True, help="query text")
    p_sem.add_argument("--limit", type=int, default=10)
    def _cmd_sem(args: argparse.Namespace) -> int:
        items = kb_index.semantic_search(getattr(args, 'q'), int(getattr(args, 'limit', 10)))
        print(json.dumps({"items": items}, indent=2, ensure_ascii=False))
        return 0
    p_sem.set_defaults(func=_cmd_sem)
