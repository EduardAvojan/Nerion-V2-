

from __future__ import annotations

from typing import Any, Dict

import types

import selfcoder.analysis.adapters.engine as engine
import selfcoder.analysis.augment.external as ext


def test_engine_uses_non_render_for_external(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_gather_external(query: str, *, root_host: str, allow, block, max_pages: int, timeout: int, render: bool, fresh_within=None):
        captured["render"] = render
        # Return empty list; we just want to inspect the render flag
        return []

    # Patch external gatherer
    monkeypatch.setattr(ext, "gather_external", fake_gather_external)

    # Patch store to avoid disk
    def fake_load_or_create(profile_key: str, *, seed_context=None):
        return {
            "profile_id": profile_key,
            "source_policies": {
                "augment": True,
                "render": True,  # even if True, externals must be forced to False
                "timeout": 3,
                "render_timeout": 2,
                "depth": 1,
                "max_pages": 1,
            },
            "scoring": {"recency": 0.3, "authority": 0.25, "agreement": 0.2, "onsite_signal": 0.15, "coverage": 0.1},
        }

    monkeypatch.setattr(engine.store_mod, "load_or_create", fake_load_or_create)

    # Patch docs read to avoid any network and return minimal text
    def fake_read_doc(path=None, *, url=None, query=None, timeout=10, render=False, render_timeout=5, selector=None):
        return {
            "url": url or "",
            "source": "url",
            "text": "Sample page text for testing.",
            "domain": "site_overview",
            "domain_confidence": 0.9,
        }

    monkeypatch.setattr(engine.docs_mod, "read_doc", fake_read_doc)

    # Avoid HTML fetching within gather
    monkeypatch.setattr(engine, "_fetch_html", lambda *a, **k: "")

    # Run the engine
    out = engine.run("example-key", query="best item", url="https://example.com")

    # Ensure external gather was invoked with render=False
    assert captured.get("render") is False