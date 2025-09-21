from __future__ import annotations
from typing import Optional, Dict, Any

from .tools_manifest import ToolsManifest

SYSTEM_GUARDRAILS = """
You are the Parent LLM (router + planner). You DO NOT execute tools yourself; you plan.
You must return a single VALID JSON object that matches the ParentDecision schema below.
Policies:
- Offline-by-default; set requires_network=true only if absolutely necessary.
- Prefer preview/dry-run and reversible steps.
- Only ask for clarification when essential details are missing; otherwise make the best reasonable assumption and proceed.
- Keep plans minimal and safe; avoid irreversible actions without confirmation.
- Never include prose around JSON; return JSON only.
"""

PARENT_DECISION_FORMAT = r"""
Return ONLY JSON (no prose), matching exactly:
{
  "intent": "string",
  "plan": [
    {"action": "tool_call" | "ask_user" | "respond",
     "tool": "string|null",
     "args": { },
     "summary": "string|null"}
  ],
  "final_response": "string|null",
  "confidence": 0.0,
  "requires_network": false,
  "notes": "string|null"
}
"""


def build_master_prompt(
    user_query: str,
    tools: ToolsManifest,
    *,
    context_snippet: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    extra_policies: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    epsilon: float = 0.0,
) -> Dict[str, Any]:
    """Construct the universal messages payload for the Parent LLM.

    Parameters
    ----------
    user_query : str
        The raw user request.
    tools : ToolsManifest
        The manifest of tools available to the Child for execution.
    context_snippet : Optional[str]
        Optional short context (recent history, file names, etc.). Keep concise.
    context : Optional[Dict[str, Any]]
        Reserved for future contextual scoring; ignored for now (no behavior change).
    extra_policies : Optional[str]
        Optional additional policy lines to append to the system prompt.
    """
    system = SYSTEM_GUARDRAILS
    if extra_policies:
        system = system + "\n" + extra_policies
    system = system + "\n" + PARENT_DECISION_FORMAT

    blocks = []
    if context_snippet:
        blocks.append(f"CONTEXT:\n{context_snippet}")
    blocks.append(f"USER:\n{user_query}")
    blocks.append("AVAILABLE_TOOLS:\n" + tools.as_prompt_block(learned_weights, epsilon=epsilon))

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": "\n\n".join(blocks).strip()},
    ]
    return {"messages": messages}
