from __future__ import annotations

import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from html.parser import HTMLParser
import hashlib
from selfcoder.analysis.domain import classify_query
from selfcoder.config import allow_network

# Simple HTML text extractor
class _HTMLStripper(HTMLParser):
    """Lightweight HTML → visible text extractor.
    - Ignores <script>, <style>, <noscript>, <svg> contents
    - Inserts spaces/newlines for block-ish tags to preserve sentence boundaries
    """
    _SKIP_TAGS = {"script", "style", "noscript", "svg"}
    _BLOCK_TAGS = {"p", "div", "section", "article", "li", "ul", "ol", "header", "footer", "nav",
                   "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        t = (tag or "").lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        if t in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        t = (tag or "").lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if t in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, d: str):
        if self._skip_depth > 0:
            return
        if d and d.strip():
            self._parts.append(d)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


# Simple, topic-agnostic boilerplate cleanup to reduce JSON/JS noise
_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
_JSONISH_LINE_RX = re.compile(r"[{\[][^\n]+[}\]]|\b(function|document\.|window\.|cookie|var\s+\w+)\b")
_URL_RX = re.compile(r"https?://\S+|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
_DURATION_RX = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_SYMBOL_HEAVY_RX = re.compile(r"[^A-Za-z0-9\s,.;:%°/-]")


def _postprocess_text(text: str) -> str:
    if not text:
        return ""
    t = _URL_RX.sub("", text)
    # Drop lines that look like JSON/JS or duration-only rows
    kept: list[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _JSONISH_LINE_RX.search(s):
            continue
        if _DURATION_RX.search(s) and sum(c.isalpha() for c in s) < 8:
            continue
        kept.append(s)
    t = " ".join(kept)
    # Normalize whitespace and drop symbol-noisy fragments
    t = re.sub(r"\s+", " ", t).strip()
    # Keep only reasonably sized sentences and join the top few
    parts = _SENT_SPLIT_RX.split(t)
    clean_sentences: list[str] = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if len(_SYMBOL_HEAVY_RX.findall(s)) > 5:
            continue
        w = s.split()
        if not (5 <= len(w) <= 60):
            continue
        clean_sentences.append(s)
        if len(clean_sentences) >= 40:  # cap to avoid huge contexts
            break
    return " ".join(clean_sentences) if clean_sentences else t


# --- PDF and URL helpers ---
def _read_pdf(fp: Path) -> str:
    try:
        import pypdf
    except Exception as e:
        raise RuntimeError(
            "PDF support is not enabled. Install extras: `pip install -e '.[docs-pdf]'` (or `pip install nerion-selfcoder[docs-pdf]`)."
        ) from e
    reader = pypdf.PdfReader(str(fp))
    texts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            texts.append(txt)
    return " \n".join(texts)


_URL_CACHE_DIR = Path("out/url_cache")
_URL_CACHE_TTL = int(os.environ.get("NERION_URL_TTL", "86400"))  # 1 day default


def _cache_name(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return _URL_CACHE_DIR / f"{h}.json"


def _read_cache_meta(url: str) -> Dict[str, Any]:
    try:
        fp = _cache_name(url)
        if not fp.exists():
            return {"hit": False}
        data = json.loads(fp.read_text(encoding="utf-8"))
        ts = int(data.get("ts", 0))
        age = int(datetime.now(timezone.utc).timestamp() - ts)
        return {
            "hit": True,
            "age_s": age,
            "etag": data.get("etag"),
            "last_modified": data.get("last_modified"),
            "stale": age > _URL_CACHE_TTL,
        }
    except Exception:
        return {"hit": False}


def _read_cache(url: str) -> Optional[str]:
    try:
        fp = _cache_name(url)
        if not fp.exists():
            return None
        data = json.loads(fp.read_text(encoding="utf-8"))
        ts = int(data.get("ts", 0))
        if (datetime.now(timezone.utc).timestamp() - ts) > _URL_CACHE_TTL:
            return None
        return str(data.get("text") or "")
    except Exception:
        return None


def _write_cache(url: str, text: str, etag: Optional[str] = None, last_mod: Optional[str] = None) -> None:
    try:
        _URL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fp = _cache_name(url)
        blob = {"url": url, "ts": int(datetime.now(timezone.utc).timestamp()), "text": text}
        if etag:
            blob["etag"] = etag
        if last_mod:
            blob["last_modified"] = last_mod
        fp.write_text(json.dumps(blob), encoding="utf-8")
    except Exception:
        pass


def _fetch_url(url: str, *, timeout: int = 30) -> str:
    # Support data: URLs for fast, deterministic tests and small inline docs
    if url.startswith("data:"):
        try:
            header, payload = url.split(",", 1)
        except ValueError:
            raise ValueError("Malformed data: URL (missing comma)")
        is_base64 = ";base64" in header.lower()
        if is_base64:
            import base64
            try:
                raw = base64.b64decode(payload)
            except Exception:
                raw = b""
            return raw.decode("utf-8", errors="ignore")
        else:
            from urllib.parse import unquote_to_bytes
            raw = unquote_to_bytes(payload)
            return raw.decode("utf-8", errors="ignore")

    try:
        import requests
    except Exception as e:
        raise RuntimeError(
            "URL fetch requires the web extras. Install: `pip install -e '.[docs-web]'` (or `pip install nerion-selfcoder[docs-web]`)."
        ) from e
    # Try cache first
    cached = _read_cache(url)
    if cached:
        return cached
    headers = {"User-Agent": "nerion-docs/1.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    text = r.text
    try:
        _write_cache(url, text, etag=r.headers.get("ETag"), last_mod=r.headers.get("Last-Modified"))
    except Exception:
        pass
    return text


def read_doc(
    path: Optional[Union[str, Path]] = None,
    *,
    url: Optional[str] = None,
    query: Optional[str] = None,
    timeout: int = 30,
    render: bool = False,
    render_timeout: int = 12,
    selector: Optional[str] = None,
) -> Dict[str, Any]:
    if (path is None) == (url is None):
        raise ValueError("Provide exactly one of `path` or `url`.")

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        ext = p.suffix.lower()
        if ext == ".pdf":
            raw = _read_pdf(p)
            text = raw
        else:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            if ext in {".md", ".txt", ".rst"}:
                text = raw
            elif ext in {".html", ".htm"}:
                stripper = _HTMLStripper()
                stripper.feed(raw)
                text = _postprocess_text(stripper.get_text())
            else:
                # fallback: just return raw
                text = raw
        dom, dconf = classify_query(query or "", url=None)
        return {
            "path": str(p),
            "name": p.name,
            "ext": ext,
            "size": p.stat().st_size,
            "text": _normalize_text(text),
            "raw_text": text,
            "source": "file",
            "query": query,
            "domain": dom,
            "domain_confidence": dconf,
        }
    elif url is not None:
        # Enforce global network gate for library callers as well
        if not allow_network():
            raise RuntimeError("network disabled by configuration (NERION_ALLOW_NETWORK)")
        if render:
            try:
                from selfcoder.analysis import web_render
            except Exception as e:
                raise RuntimeError(
                    "JS rendering requires the web extras. Install: `pip install -e '.[docs-web]'` and run `playwright install chromium`."
                ) from e
            try:
                html = web_render.render_url(url, timeout=timeout, render_timeout=render_timeout, selector=selector)
                text = _postprocess_text(web_render.extract_main_text(html))
                raw = html
                render_fallback = False
            except Exception:
                # Playwright or render failed — fall back to plain HTTP fetch
                raw = _fetch_url(url, timeout=timeout)
                stripper = _HTMLStripper()
                stripper.feed(raw)
                text = _postprocess_text(stripper.get_text())
                render_fallback = True
        else:
            raw = _fetch_url(url, timeout=timeout)
            stripper = _HTMLStripper()
            stripper.feed(raw)
            text = _postprocess_text(stripper.get_text())
            render_fallback = False
        dom, dconf = classify_query(query or "", url=url)
        out = {
            "url": url,
            "timeout": timeout,
            "render": bool(render),
            "render_timeout": render_timeout,
            "selector": selector,
            "text": _normalize_text(text),
            "raw_text": text,
            "source": "url",
            "query": query,
            "domain": dom,
            "domain_confidence": dconf,
            "render_fallback": render_fallback,
        }
        # Attach cache meta
        try:
            meta = _read_cache_meta(url)
            out["cache_hit"] = bool(meta.get("hit"))
            out["cache_age_s"] = int(meta.get("age_s") or 0)
            out["etag"] = meta.get("etag")
            out["last_modified"] = meta.get("last_modified")
        except Exception:
            pass
        return out


def summarize_text(text: str, max_len: int = 200) -> Dict[str, Any]:
    words = text.split()
    summary = " ".join(words[:max_len])
    bullets: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith(("-", "*")):
            bullets.append(_normalize_text(line))
    return {"summary": summary, "bullets": bullets}


def persist_assimilation(bundle: Dict[str, Any]) -> Path:
    outdir = Path("out/docs_assimilation")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fpath = outdir / f"doc_{ts}.json"
    fpath.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
    return fpath


def assimilate(path: Optional[Union[str, Path]] = None, *, url: Optional[str] = None, timeout: int = 30, query: Optional[str] = None) -> Dict[str, Any]:
    doc = read_doc(path, url=url, query=query, timeout=timeout)
    summary = summarize_text(doc.get("raw_text") or doc["text"])
    bundle = {**doc, **summary}
    artifact = persist_assimilation(bundle)
    bundle["artifact_path"] = str(artifact)
    return bundle
