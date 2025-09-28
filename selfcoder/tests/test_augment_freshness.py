


from __future__ import annotations

from typing import Optional

from selfcoder.analysis.augment.external import (
    _parse_window,
    _latest_epoch,
    _html_to_text,
    MIN_TEXT_CHARS,
)


def _recent_iso(days_ago: int = 30) -> str:
    ts = time.time() - days_ago * 86400
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _should_keep(text: str, fresh_within: Optional[str]) -> bool:
    """Mimic gather_external's keep logic (length + freshness)."""
    # length gate first
    if len(text) < MIN_TEXT_CHARS:
        return False
    cutoff = _parse_window(fresh_within)
    if cutoff is None:
        return True
    latest = _latest_epoch(text)
    return bool(latest is not None and latest >= cutoff)


def test_keep_recent_and_long():
    # Build a long HTML with a recent date
    date = _recent_iso(30)  # within 60d
    html = f"<html><body><p>Updated on {date}</p><div>{'X' * (MIN_TEXT_CHARS + 200)}</div></body></html>"
    text = _html_to_text(html)
    assert _should_keep(text, "60d") is True


def test_drop_old_even_if_long():
    # Date older than 60d should be dropped, even with long content
    date = _recent_iso(120)  # older than 60d
    html = f"<html><body><p>Last updated {date}</p><div>{'Y' * (MIN_TEXT_CHARS + 500)}</div></body></html>"
    text = _html_to_text(html)
    assert _should_keep(text, "60d") is False


def test_drop_no_date_when_fresh_within():
    # No detectable date -> drop when fresh window is enforced
    html = f"<html><body><p>No date here</p><div>{'Z' * (MIN_TEXT_CHARS + 100)}</div></body></html>"
    text = _html_to_text(html)
    assert _should_keep(text, "60d") is False


def test_keep_long_when_no_fresh_spec():
    # Without a fresh_within spec, long content passes
    html = f"<html><body><div>{'W' * (MIN_TEXT_CHARS + 50)}</div></body></html>"
    text = _html_to_text(html)
    assert _should_keep(text, None) is True