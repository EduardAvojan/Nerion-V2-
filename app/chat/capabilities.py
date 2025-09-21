"""
Dynamic capability summary for Nerion.

This module inspects the current configuration and runtime to produce a
human‑readable summary that stays up‑to‑date as you add tools, intents, or
plugins. It intentionally avoids crashing if optional files are missing.
"""

from __future__ import annotations

from typing import Tuple, List
import os
import json
import time

_CACHE = {"sig": None, "brief": None, "detailed": None, "ts": 0.0}


def _count_tools_yaml(path: str) -> Tuple[int, int, int]:
    """Return (total, offline, requires_net) from a simple tools.yaml schema.

    Schema tolerated:
      tools:
        - name: str
          description: str
          requires_network: bool (optional)
          args: {...} (ignored)
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return (0, 0, 0)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = list(data.get("tools") or [])
        total = len(items)
        net = 0
        off = 0
        for t in items:
            req = bool((t or {}).get("requires_network", False))
            if req:
                net += 1
            else:
                off += 1
        return (total, off, net)
    except Exception:
        return (0, 0, 0)


def _count_plugins() -> int:
    try:
        # Prefer explicit allowlist if present
        p = os.path.join("plugins", "allowlist.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if isinstance(arr, list):
                return len(arr)
        # Fallback: count python files in plugins directory (best-effort)
        if os.path.isdir("plugins"):
            return len([n for n in os.listdir("plugins") if n.endswith(".py") and n != "__init__.py"])
    except Exception:
        pass
    return 0


def _count_intents_yaml(path: str) -> Tuple[int, int, int]:
    """Return (total, local_count, web_count) for intents.yaml.

    Counts by name prefix: local.* vs web.*; other intents included in total.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return (0, 0, 0)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = list(data.get("intents") or [])
        total = len(items)
        local = 0
        web = 0
        for it in items:
            nm = str((it or {}).get("name") or "")
            if nm.startswith("local."):
                local += 1
            elif nm.startswith("web."):
                web += 1
        return (total, local, web)
    except Exception:
        return (0, 0, 0)


def _signature() -> str:
    """Compute a cheap signature from mtimes of key config files and plugin dir."""
    paths = [
        os.path.join("config", "tools.yaml"),
        os.path.join("config", "intents.yaml"),
        os.path.join("plugins", "allowlist.json"),
    ]
    sig_parts = []
    for p in paths:
        try:
            st = os.stat(p)
            sig_parts.append(f"{p}:{int(st.st_mtime)}:{st.st_size}")
        except Exception:
            sig_parts.append(f"{p}:na")
    # Include plugin filenames
    try:
        if os.path.isdir("plugins"):
            names = sorted([n for n in os.listdir("plugins") if n.endswith(".py")])
            sig_parts.append("plugins:" + ",".join(names))
    except Exception:
        sig_parts.append("plugins:na")
    return "|".join(sig_parts)


def _build_facts() -> List[str]:
    total, off, net = (0, 0, 0)
    intents_total, intents_local, intents_web = (0, 0, 0)
    try:
        cfg_tools = os.path.join("config", "tools.yaml")
        if os.path.exists(cfg_tools):
            total, off, net = _count_tools_yaml(cfg_tools)
    except Exception:
        pass
    try:
        intents_path = os.path.join("config", "intents.yaml")
        if os.path.exists(intents_path):
            intents_total, intents_local, intents_web = _count_intents_yaml(intents_path)
    except Exception:
        pass

    has_selfcode = os.path.exists(os.path.join("selfcoder", "orchestrator.py"))
    has_self_improve = os.path.exists(os.path.join("selfcoder", "self_improve.py"))
    has_upgrade_agent = os.path.exists(os.path.join("app", "learning", "upgrade_agent.py"))
    has_parent = os.path.exists(os.path.join("app", "parent", "executor.py"))
    has_memory = os.path.exists(os.path.join("app", "chat", "memory_bridge.py"))
    has_voice = True
    try:
        import speech_recognition  # noqa: F401
    except Exception:
        has_voice = False
    plugins = _count_plugins()

    facts: List[str] = []
    facts.append("Local, privacy‑first assistant running on this device")
    facts.append("General chat and Q&A are offline by default")
    if has_memory:
        facts.append("Can remember and recall user preferences and facts")
    if has_selfcode:
        facts.append("Can self‑code this repo safely using AST with snapshot & rollback")
    if has_self_improve or has_upgrade_agent:
        facts.append("Proactive self‑improvement and learning with scan→plan→apply and review gates")
    if net > 0:
        facts.append("Can research the web when you allow it")
    if has_voice:
        facts.append("Operates by voice or text with push‑to‑talk and TTS")
    if total > 0:
        facts.append(f"Has {total} tools configured ({off} offline, {net} online)")
    if intents_total > 0:
        facts.append(f"Understands {intents_total} intents ({intents_local} local, {intents_web} web)")
    if plugins:
        facts.append(f"Loads {plugins} plugin(s) from an allowlist")
    if os.path.exists(os.path.join("ops", "security", "fs_guard.py")) or os.path.exists(os.path.join("ops", "security", "safe_subprocess.py")):
        facts.append("Enforces repo‑jail and safe subprocesses")
    if os.path.exists(os.path.join("ops", "security", "net_gate.py")):
        facts.append("Asks before using the internet (network gate)")
    if has_parent:
        facts.append("Uses a planner to route allowed tools when needed")
    return facts


def dev_options() -> List[str]:
    """Return a list of capability option keys available for developer detail."""
    options: List[str] = []
    if os.path.exists(os.path.join("selfcoder", "orchestrator.py")):
        options.append("self-code")
    if os.path.exists(os.path.join("selfcoder", "self_improve.py")) or os.path.exists(os.path.join("app", "learning", "upgrade_agent.py")):
        options.append("self-improve")
    if os.path.exists(os.path.join("app", "chat", "memory_bridge.py")):
        options.append("memory")
    if os.path.exists(os.path.join("ops", "security", "net_gate.py")):
        options.append("web-research")
    options.append("voice-text")
    if os.path.exists(os.path.join("ops", "security", "fs_guard.py")) or os.path.exists(os.path.join("ops", "security", "safe_subprocess.py")):
        options.append("security")
    if os.path.exists(os.path.join("app", "parent", "executor.py")):
        options.append("planner")
    options.append("tools-plugins")
    return options


def summarize_dev_options() -> str:
    opts = dev_options()
    human = ", ".join(opts)
    return (
        "Developer mode is available. You can ask for details about: "
        + human
        + ". For example: 'show self-code details' or 'details for security'."
    )


def summarize_capability_detail(topic: str) -> str:
    """Return a developer‑friendly, dynamic detail for a specific capability.

    Supported topics: self-code, self-improve, memory, web-research, voice-text,
    security, planner, tools-plugins.
    """
    t = (topic or "").strip().lower()
    if t in {"self-code", "self code", "coding", "autocoder"}:
        return (
            "Self‑code: Uses AST transforms via a sandboxed orchestrator to edit code with "
            "syntax guarantees. Snapshots are taken before writes; healthcheck/coverage can gate apply; "
            "automatic rollback on failure. Supports cross‑file refactors, docstrings, wrappers, and batch plans."
        )
    if t in {"self-improve", "self improve", "learning", "self-learn", "self learn"}:
        return (
            "Self‑improve: Periodic scan→plan→apply pipeline with review gates. Generates actionable plans, "
            "runs in simulation first, applies with safety checks, and offers upgrade prompts when data suggests."
        )
    if t in {"memory", "mem"}:
        return (
            "Memory: Long‑ and short‑term store with decay/promotion. Extracts facts from speech (e.g., 'remember that …'), "
            "supports pin/unpin/forget, and surfaces relevant facts into chat prompts."
        )
    if t in {"web", "web-research", "research"}:
        return (
            "Web research: Disabled by default; asks permission per session or task. Performs search and deep reads, "
            "saves artifacts, and cites domains. Network gate enforces allow/deny with time‑boxed grants."
        )
    if t in {"voice", "voice-text", "audio", "speech"}:
        return (
            "Voice/Text: Push‑to‑talk with hold‑to‑record; TTS with barge‑in; reliable toggles (CapsLock/F9). "
            "Works fully offline; recognizes common name aliases for 'Nerion'."
        )
    if t in {"security", "safety"}:
        return (
            "Security: Repo‑jail file I/O, safe subprocess wrapper, plugin allowlist, secret‑redacting logs, and a network gate. "
            "All tool plans are validated; destructive actions prefer simulation first."
        )
    if t in {"planner", "parent", "routing"}:
        return (
            "Planner: A local planner routes allowed tools from a manifest, with argument validation and metrics. "
            "Runs after local intents; short timeout prevents UI stalls."
        )
    if t in {"tools", "plugins", "tools-plugins"}:
        total, off, net = (0, 0, 0)
        try:
            cfg_tools = os.path.join("config", "tools.yaml")
            if os.path.exists(cfg_tools):
                total, off, net = _count_tools_yaml(cfg_tools)
        except Exception:
            pass
        plugins = _count_plugins()
        return (
            f"Tools/Plugins: {total} tool(s) configured ({off} offline, {net} online); "
            f"{plugins} plugin(s) loadable from the allowlist."
        )
    return "I can share developer details for: " + ", ".join(dev_options()) + "."


def summarize_capabilities(style: str = "brief") -> str:
    """Return a concise, auto-updating summary of Nerion's capabilities."""
    sig = _signature()
    cached = _CACHE.get(style)
    if _CACHE['sig'] == sig and cached:
        return cached  # type: ignore

    facts = _build_facts()
    if style == 'detailed':
        body = '\n'.join(f"- {fact}" for fact in facts)
        out_text = "I’m Nerion. Here’s a quick overview of my major capabilities:\n" + body
    else:
        lead = "I’m Nerion, your privacy-first assistant orchestrating the hosted models you configure."
        headline = ', '.join(facts[:6]).rstrip('.') if facts else 'help with local development tasks'
        out_text = lead + ' I can ' + headline + '.'

    _CACHE['sig'] = sig
    _CACHE[style] = out_text
    _CACHE['ts'] = time.time()
    return out_text

