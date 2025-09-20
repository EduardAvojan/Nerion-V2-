"""Heuristic task-aware model picker.

Given a task/instruction and optional repo hints, rank model families:
  deepseek-coder, qwen2.5-coder, starcoder2, codellama

Signals considered (cheap, offline):
- Language mentions (python, js, java, c/cpp)
- Size/complexity hints ("refactor", "rewrite", "architecture", many files)
- Test framework hints (pytest, unittest) – neutral
"""

from __future__ import annotations

from typing import Dict, List, Optional
import re


_FAMILIES = [
    "deepseek-coder",
    "qwen2.5-coder",
    "starcoder2",
    "codellama",
]


def rank_families(instruction: str, context: Optional[Dict] = None) -> List[str]:
    text = (instruction or "").lower()
    ctx = context or {}
    score = {f: 0 for f in _FAMILIES}

    # Complexity hints → prefer stronger coders first
    if re.search(r"\b(refactor|rewrite|redesign|optimi[sz]e|architecture)\b", text):
        score["deepseek-coder"] += 3
        score["qwen2.5-coder"] += 2

    # Python focus: all are fine; slight boost to deepseek/qwen
    if re.search(r"\b(py|python|pytest|unittest)\b", text):
        score["deepseek-coder"] += 2
        score["qwen2.5-coder"] += 1

    # JS/TS: prefer qwen2.5-coder, then starcoder2
    if re.search(r"\b(js|javascript|typescript|node)\b", text):
        score["qwen2.5-coder"] += 2
        score["starcoder2"] += 1

    # C/C++: prefer codellama/starcoder2
    if re.search(r"\b(c\+\+|cpp|\.cc\b|\.cxx\b|\.c\b)\b", text):
        score["codellama"] += 2
        score["starcoder2"] += 1

    # If repo looks large (hint), favor stronger models
    if int(str(ctx.get("files_count") or 0)) > 500:
        score["deepseek-coder"] += 2
        score["qwen2.5-coder"] += 1

    ranked = sorted(_FAMILIES, key=lambda f: score[f], reverse=True)
    # Ensure deterministic fallback order
    seen = set()
    out: List[str] = []
    for f in ranked + _FAMILIES:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
