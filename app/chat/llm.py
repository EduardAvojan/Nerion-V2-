

"""LLM utilities for Nerion (consolidated).

This module merges:
- Prompt helpers: follow‑up prompt builder, style hints, clarifier, output scrubber
- LLM chain builders: plain and temperature‑aware chains (with a tiny fallback)

Keeping these together reduces import churn in the runner/engine.
"""
from __future__ import annotations
from typing import List, Optional, Any
import re

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
    """Construct a compact follow-up prompt from recent history and optional artifact."""
    turns = (history or [])[-8:]
    lines: List[str] = []
    lines.append("You are Nerion. Continue the SAME conversation.")
    # Persona + disclosure guardrails
    lines.append("Identify only as 'Nerion', the user's private developer assistant orchestrating hosted models on their behalf.")
    lines.append("Never claim to be ChatGPT, GPT-3.5/4, DeepSeek, LLaMA, or any company-branded model.")
    lines.append("If asked about the model, explain that Nerion routes to the user's configured providers (e.g., OpenAI, Google) and summarize which one is active if known.")
    lines.append("Keep replies conversational by default (2–3 sentences) unless the user explicitly asks for something shorter or longer.")
    lines.append("Answer about the previously recommended product/page, not about yourself.")
    lines.append("If asked for a 'model number' or similar identifier, extract it from the artifact/context if present; if missing, say so and ask one precise clarifying question.")
    lines.append("If the user is making small talk, respond naturally without deflecting unless a tool run is required.")
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
        "Answer directly in a conversational tone, usually 2–3 sentences unless the user requests otherwise."
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


# --------------------------- Provider-backed chain --------------------------




class _ProviderBackedChain:
    """Simple adapter exposing LangChain-like interface backed by ProviderRegistry."""

    IS_STUB = False

    def __init__(self, *, role: str, temperature: float):
        from app.chat.providers import get_registry  # local import

        self._role = role
        self._temperature = float(temperature)
        self._registry = get_registry()
        self._nerion_model_name = self._describe_model()
        self.last_response = None

    def _describe_model(self) -> str:
        from app.chat.providers import ProviderNotConfigured  # local import
        try:
            adapter, model_name, _spec = self._registry.resolve(self._role)
            return f"{adapter.name}:{model_name}"
        except ProviderNotConfigured:
            return "unconfigured"

    def predict(self, prompt: Optional[str] = None, **kwargs: Any) -> str:  # pragma: no cover - wrapper
        from app.chat.providers import ProviderError, ProviderNotConfigured  # local import
        if prompt is None:
            if 'input' in kwargs:
                prompt = kwargs.pop('input')
            elif 'prompt' in kwargs:
                prompt = kwargs.pop('prompt')
            else:
                prompt = ''
        if not isinstance(prompt, str):
            prompt = str(prompt or '')
        messages = kwargs.pop('messages', None)
        response_format = kwargs.pop('response_format', None)
        max_tokens = kwargs.pop('max_tokens', None)
        if max_tokens is not None:
            try:
                max_tokens = int(max_tokens)
            except Exception:
                max_tokens = None
        try:
            result = self._registry.generate(
                role=self._role,
                prompt=prompt,
                temperature=self._temperature,
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except ProviderNotConfigured:
            self.IS_STUB = True
            self._nerion_model_name = "unconfigured"
            return (
                "I need API credentials before I can answer. "
                "Set the required keys from app/settings.yaml (e.g., NERION_V2_OPENAI_KEY) and try again."
            )
        except ProviderError as exc:
            self.IS_STUB = True
            self._nerion_model_name = self._describe_model()
            return f"Sorry, I hit a provider error: {exc}"
        self.IS_STUB = False
        self.last_response = result
        self._nerion_model_name = f"{result.provider}:{result.model}"
        return result.text or ""

    def stop(self) -> None:
        """Maintain API parity with legacy LangChain objects."""
        return None


_CHAIN_CACHE: dict[str, _ProviderBackedChain] = {}


def build_chain() -> Any:
    return build_chain_with_temp(0.7)


def build_chain_with_temp(temp: float) -> Any:
    key = f"{float(temp):.2f}"
    chain = _CHAIN_CACHE.get(key)
    if chain is None:
        chain = _ProviderBackedChain(role="chat", temperature=temp)
        _CHAIN_CACHE[key] = chain
    return chain
