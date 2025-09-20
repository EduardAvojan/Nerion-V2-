from __future__ import annotations

from typing import Optional

# Network gate for enforcing offline-by-default policy
from ops.security.net_gate import NetworkGate


# We deliberately avoid importing heavy deps at module import time.
# Guarded imports happen inside functions with clear, actionable errors.

# Optional HTTP client exposed at module level for tests to monkeypatch
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - absence is acceptable
    httpx = None  # type: ignore

# Expose urllib request helpers at module level so tests can monkeypatch them
try:
    from urllib.request import Request as Request, urlopen as urlopen  # type: ignore
except Exception:  # pragma: no cover
    Request = None  # type: ignore
    urlopen = None  # type: ignore

DEFAULT_UA = "nerion-docs/1.0"


def _friendly_web_extra_error(extra: str) -> RuntimeError:
    return RuntimeError(
        "Web rendering support is not enabled. Install extras: "
        f"`pip install -e '.[{extra}]'` (or `pip install nerion-selfcoder[{extra}]`) and, for Playwright, run "
        "`playwright install chromium`."
    )


def render_url(
    url: str,
    *,
    timeout: int = 30,
    render_timeout: int = 12,
    http_timeout: int = 10,
    selector: Optional[str] = None,
    block_third_party: bool = True,
    user_agent: str = DEFAULT_UA,
) -> str:
    """Render a URL with JavaScript (headless Chromium via Playwright) and return final HTML.

    Parameters
    ----------
    url: str
        The HTTP/HTTPS URL to render.
    timeout: int
        Network timeout in seconds for navigation.
    render_timeout: int
        Additional wait time for JS to settle; we wait up to this long after network idle.
    selector: Optional[str]
        If provided, wait for this CSS selector to appear before snapshotting HTML.
    block_third_party: bool
        If True, block third‑party requests (ads/analytics/images) for speed & privacy.
    user_agent: str
        Custom user‑agent string.

    Returns
    -------
    str
        The final page HTML after rendering.
    """
    # Lightweight HTTP fallback (no JS) used when Playwright times out
    def _http_fetch(raw_url: str, *, http_timeout: int, ua: str) -> str:
        # Safety: enforce network gate again in case this code path is reached after Playwright fallback
        NetworkGate.assert_allowed(task_type="web_render", url=raw_url)
        try:
            if httpx is not None:
                r = httpx.get(raw_url, headers={"User-Agent": ua}, timeout=http_timeout, follow_redirects=True)
                r.raise_for_status()
                return r.text
        except Exception:
            pass
        # Last-gasp urllib fallback
        try:
            if Request is None or urlopen is None:
                raise RuntimeError("urllib.request not available")
            req = Request(raw_url, headers={"User-Agent": ua})
            with urlopen(req, timeout=http_timeout) as resp:  # type: ignore
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise e

    # Enforce offline-by-default policy; allow only if a session grant exists
    NetworkGate.assert_allowed(task_type="web_render", url=url)

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
    except Exception:
        # If Playwright isn't available, fall back to plain HTTP fetch
        return _http_fetch(url, http_timeout=http_timeout, ua=user_agent)

    def _should_block(route_url: str, page_origin: str) -> bool:
        if not block_third_party:
            return False
        # Block obvious heavy/non‑essential types by substring; keep it conservative.
        # We do not parse full origins here to avoid extra deps.
        lower = route_url.lower()
        for bad in (".doubleclick.", "/ads/", "/analytics", "/gtag/", "/collect", ".mp4", ".webm", ".gif", ".jpg", ".png", ".svg"):
            if bad in lower:
                return True
        # Simple third‑party heuristic: different scheme+host prefix
        try:
            from urllib.parse import urlparse
            u, p = urlparse(route_url), urlparse(page_origin)
            if u.netloc and p.netloc and u.netloc != p.netloc:
                return True
        except Exception:
            pass
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        page = context.new_page()

        if block_third_party:
            page.route("**/*", lambda route: route.abort() if _should_block(route.request.url, url) else route.continue_())

        page.set_default_navigation_timeout(timeout * 1000)
        page.set_default_timeout(timeout * 1000)

        # Primary navigation attempt
        try:
            page.goto(url, wait_until="domcontentloaded")
        except PWTimeoutError:
            # Retry once with more permissive wait condition and longer nav timeout
            try:
                page.set_default_navigation_timeout((timeout + 10) * 1000)
                page.goto(url, wait_until="load")
            except PWTimeoutError:
                # Fall back to plain HTTP fetch (no JS)
                context.close()
                browser.close()
                return _http_fetch(url, http_timeout=timeout, ua=user_agent)

        # Wait for either network to be idle briefly or a selector to appear
        try:
            if selector:
                page.wait_for_selector(selector, timeout=render_timeout * 1000)
            else:
                page.wait_for_load_state("networkidle", timeout=render_timeout * 1000)
        except Exception:
            # Non‑fatal; proceed with what we have
            pass

        html = page.content()
        context.close()
        browser.close()
        return html


def extract_main_text(html: str) -> str:
    """Extract main readable text from HTML.

    Prefers `trafilatura` if installed; falls back to a minimal text extractor.
    """
    try:
        import trafilatura  # type: ignore
        extracted = trafilatura.extract(html)  # returns None on failure
        if extracted:
            return " ".join(extracted.split())
    except Exception:
        pass

    # Minimal fallback: strip tags using html.parser
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
        def handle_data(self, d: str):
            self._parts.append(d)
        def get(self) -> str:
            return " ".join(self._parts)

    s = _Stripper()
    s.feed(html)
    return " ".join(s.get().split())