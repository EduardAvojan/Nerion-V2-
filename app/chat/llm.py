

"""LLM utilities for Nerion (consolidated).

This module merges:
- Prompt helpers: follow‑up prompt builder, style hints, clarifier, output scrubber
- LLM chain builders: plain and temperature‑aware chains (with a tiny fallback)

Keeping these together reduces import churn in the runner/engine.
"""
from __future__ import annotations
from typing import List, Optional, Any
import os
import re
import socket
from urllib.parse import urlparse

__all__ = [
    # prompt helpers
    "_build_followup_prompt",
    "_answer_style_hint",
    "_make_clarifying_question",
    "_strip_think_blocks",
    # chain builders
    "build_chain",
    "build_chain_with_temp",
]

# --------------------------- Prompt helpers ---------------------------------

def _build_followup_prompt(history: List[dict], user_text: str, artifact_text: Optional[str]) -> str:
    """Construct a compact follow‑up prompt from recent history and optional artifact."""
    turns = (history or [])[-8:]
    lines: List[str] = []
    lines.append("You are Nerion. Continue the SAME conversation.")
    # Persona + disclosure guardrails
    lines.append("Identify only as 'Nerion' (a local, privacy‑first assistant running on the user's device).")
    lines.append("Never claim to be ChatGPT, GPT‑3.5/4, DeepSeek, LLaMA, or any company‑branded model.")
    lines.append("If asked what model you use, reply: 'I run locally as Nerion on your device.' Do not name a provider or model ID.")
    lines.append("Keep replies concise by default (1–2 sentences) unless the user asks for detail.")
    lines.append("Answer about the previously recommended product/page, not about yourself.")
    lines.append("If asked for a 'model number' or similar identifier, extract it from the artifact/context if present; if missing, say so and ask one precise clarifying question.")
    lines.append("Do NOT explain your chain-of-thought. Output the final answer only.")
    lines.append("\nConversation so far:")
    for t in turns:
        role = (t.get('role') or '').upper()
        content = (t.get('content') or '').strip()
        lines.append(f"{role}: {content}")
    if artifact_text:
        lines.append("\nStructured context (artifact, truncated):")
        lines.append(artifact_text)
    lines.append("\nUSER (follow-up): " + (user_text or ''))
    lines.append("\nASSISTANT:")
    return "\n".join(lines)


def _answer_style_hint(user_q: str) -> str:
    """Return a topic‑agnostic style hint. No keyword branching."""
    return (
        "Prefer concrete facts (numbers, dates, names). Quote values exactly as found. "
        "If sources disagree, prefer the most recent or majority view. "
        "Answer directly in 1–2 sentences."
    )


def _make_clarifying_question(user_text: str) -> str:
    """Return a single, crisp clarifying question (light keyword cues)."""
    low = (user_text or '').lower()
    if any(k in low for k in ['news', 'market', 'stocks', 'stock market']):
        return "Are you asking for today's market summary or major moves this week?"
    if any(k in low for k in ['code', 'python', 'nerion']):
        return "Is this about Nerion's codebase or a general programming question?"
    # Generic fallback
    return 'Got it — can you clarify in one line what you want so I answer precisely?'


def _strip_think_blocks(resp: str) -> str:
    """Remove hidden chain‑of‑thought and prompt‑echo artifacts from a model reply."""
    if not resp:
        return ""
    try:
        # Remove explicit chain-of-thought tags (full and stray)
        s = re.sub(r"<think>.*?</think>", "", resp, flags=re.S|re.I)
        s = re.sub(r"</?think>", "", s, flags=re.I)
        # Drop common prompt-echo lines
        s = "\n".join(
            ln for ln in s.splitlines()
            if not re.match(r"^\s*(Human:|Answer:|CONTEXT:|TASK:)\b", ln)
        ).strip()
        # Remove inline 'Human: ... AI:' chatter sometimes embedded in one line
        s = re.sub(r"Human:.*?AI:\s*", "", s, flags=re.S)
        # Remove stray role tags and prompt echoes
        s = re.sub(r"You are Nerion\..*?Answer:\s*", "", s, flags=re.S)
        s = re.sub(r"\bUser said:\s*", "", s, flags=re.I)
        # Remove generic AI disclaimers that add no value
        s = re.sub(r"\bAs an AI\b.*", "", s, flags=re.I)
        s = re.sub(r"\bI(?: am|'m) an AI\b.*", "", s, flags=re.I)
        s = re.sub(r"\bI (?:do not|don't) have personal (?:experiences|feelings)\b.*", "", s, flags=re.I)
        s = re.sub(r"\bI (?:was|am) trained on data\b.*", "", s, flags=re.I)
        prev = None
        out_lines = []
        for ln in [line.strip() for line in s.splitlines() if line.strip()]:
            if ln != prev:
                out_lines.append(ln)
            prev = ln
        return " ".join(out_lines).strip()
    except Exception:
        return resp.strip()


# --------------------------- Chain builders ---------------------------------

def _ollama_endpoint() -> tuple[str, int]:
    """Return host/port tuple for the configured Ollama endpoint."""
    host = os.getenv('OLLAMA_HOST') or os.getenv('OLLAMA_BASE_URL') or 'http://127.0.0.1:11434'
    if '://' not in host:
        host = f'http://{host}'
    parsed = urlparse(host)
    hostname = parsed.hostname or '127.0.0.1'
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == 'https' else 80
    return hostname, port


def _ollama_available(timeout: float = 0.35) -> bool:
    host, port = _ollama_endpoint()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_chain() -> Any:
    """Construct a conversational chain.

    Tries to build a LangChain + ChatOllama pipeline. If LangChain or the
    provider isn't available, fall back to a tiny local stub that echoes
    with light formatting. This keeps CLI/tests resilient on minimal envs.
    """
    try:
        from langchain_ollama import ChatOllama  # type: ignore
        from langchain.chains import ConversationChain  # type: ignore
        from langchain.memory import ConversationBufferMemory  # type: ignore
        if not _ollama_available():
            raise RuntimeError('Ollama endpoint unavailable')
        model = os.getenv('NERION_LLM_MODEL', 'deepseek-r1:14b')
        llm = ChatOllama(model=model)
        memory = ConversationBufferMemory()
        chain = ConversationChain(llm=llm, memory=memory, verbose=False)
        setattr(chain, '_nerion_model_name', model)
        setattr(chain, 'IS_STUB', False)
        return chain
    except Exception:
        class _StubChain:
            IS_STUB = True
            _nerion_model_name = 'stub'
            def predict(self, input: str) -> str:  # pragma: no cover - simple fallback
                txt = (input or '').strip()
                if '\nUser said:' in txt:
                    txt = txt.split('\nUser said:', 1)[1].strip()
                return f"(fallback) You said: {txt}"
        return _StubChain()


_CHAIN_CACHE: dict[str, Any] = {}

def build_chain_with_temp(temp: float) -> Any:
    key = f"{float(temp):.2f}"
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    try:
        from langchain_ollama import ChatOllama  # type: ignore
        from langchain.chains import ConversationChain  # type: ignore
        from langchain.memory import ConversationBufferMemory  # type: ignore
        model = os.getenv('NERION_LLM_MODEL', 'deepseek-r1:14b')
        llm = ChatOllama(model=model, temperature=float(temp))
        chain = ConversationChain(llm=llm, memory=ConversationBufferMemory(), verbose=False)
        _CHAIN_CACHE[key] = chain
        setattr(chain, '_nerion_model_name', model)
        setattr(chain, 'IS_STUB', False)
        return chain
    except Exception:
        return build_chain()
