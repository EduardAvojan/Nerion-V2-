from typing import List, Optional, Dict
from pydantic import BaseModel
import yaml

class ToolParam(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    default: Optional[str] = None

class ToolSpec(BaseModel):
    name: str
    description: Optional[str] = None
    params: List[ToolParam]
    learned_weight: Optional[float] = None

class ToolsManifest(BaseModel):
    tools: List[ToolSpec]

    def as_prompt_block(self, learned_weights: Optional[Dict[str, float]] = None, *, epsilon: float = 0.0) -> str:
        """Render a concise, exact manifest block for prompts.

        Format:
          - tool_name: description
            params: name:type [default] — short desc; ...
            when: brief guidance
        """
        lines: List[str] = []
        items = list(self.tools)
        # Apply learned weights for annotation and optional epsilon-greedy ordering
        if learned_weights:
            for t in items:
                try:
                    w = float(learned_weights.get(t.name, 0.0))
                except Exception:
                    w = 0.0
                t.learned_weight = w
            # Order by weight descending; epsilon: small chance to promote lower-weight items
            try:
                import random as _rnd
                # Deterministic seeding for CI snapshots if requested
                try:
                    import os as _os
                    seed_raw = _os.getenv('NERION_BANDIT_SEED')
                    if seed_raw is not None and str(seed_raw).strip() != '':
                        _rnd.seed(int(seed_raw))
                except Exception:
                    pass
                items.sort(key=lambda x: (x.learned_weight or 0.0), reverse=True)
                if epsilon and 0.0 < float(epsilon) < 1.0 and len(items) >= 2:
                    if _rnd.random() < float(epsilon):
                        # pick a random item from the lower half and swap it near the top
                        low_idx = max(1, len(items)//2)
                        j = _rnd.randrange(low_idx, len(items))
                        items.insert(1, items.pop(j))
            except Exception:
                pass
        for t in items:
            lines.append(f"- {t.name}: {t.description or ''}".rstrip())
            if t.learned_weight is not None:
                lines.append(f"  learned_weight: {t.learned_weight:.2f}")
            if t.params:
                pbits = []
                for p in t.params:
                    bit = f"{p.name}:{p.type}"
                    if p.default is not None:
                        bit += f"={p.default}"
                    if p.description:
                        bit += f" — {p.description}"
                    pbits.append(bit)
                lines.append("  params: " + "; ".join(pbits))
            # Specific guidance for known tools
            when = None
            if t.name == "read_url":
                when = "use to fetch and summarize a single URL; prefer when the user provides a concrete link"
            elif t.name == "web_search":
                when = "use for open-ended queries or when no specific URL is provided; retrieve a small set of recent, reputable sources"
            elif t.name == "rename_symbol":
                when = "use to plan safe code renames (module/attribute) across the repo; prefer preview before apply"
            else:
                # Fallback on naming conventions
                if t.name.startswith("read_"):
                    when = "use to read a single resource of that type"
                elif t.name.endswith("search"):
                    when = "use for general search in that domain"
            if when:
                lines.append(f"  when: {when}")
        return "\n".join(lines)

def load_tools_manifest_from_yaml(yaml_path: str) -> ToolsManifest:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return ToolsManifest(**data)

# Example builtin manifest
builtin_manifest_yaml = """
tools:
  - name: echo
    description: Echo the input string
    params:
      - name: input
        type: string
        description: The string to echo
  - name: add
    description: Add two numbers
    params:
      - name: a
        type: int
        description: First number
      - name: b
        type: int
        description: Second number
"""

_builtin_data = yaml.safe_load(builtin_manifest_yaml)
builtin_manifest = ToolsManifest(**_builtin_data)
