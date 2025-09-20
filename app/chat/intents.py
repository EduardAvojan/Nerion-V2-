from __future__ import annotations
from typing import List, Optional, Dict, Any
import yaml
import re
from pydantic import BaseModel, Field
try:
    # pydantic v2
    from pydantic import ConfigDict  # type: ignore
except Exception:  # pragma: no cover
    ConfigDict = dict  # type: ignore

# --- Pydantic Models for a Validated, Type-Safe Intent Structure ---
# This brings the professional validation from your tools_manifest concept directly
# to your single source of truth: intents.yaml.

class IntentSpec(BaseModel):
    """Defines the structure for a single intent in intents.yaml."""
    name: str = Field(..., min_length=1)
    priority: int = 10
    is_triage_rule: bool = False
    description: Optional[str] = Field(default="", min_length=0)
    patterns: List[str] = Field(default_factory=list)
    handler: str
    regexes: List[re.Pattern] = Field(default_factory=list, repr=False)

    # Pydantic v2-compatible config
    try:
        model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        class Config:  # type: ignore
            arbitrary_types_allowed = True

class IntentManifest(BaseModel):
    """Represents the entire collection of intents loaded from YAML."""
    intents: List[IntentSpec]

# --- The Central Intent Registry ---

class IntentRegistry:
    """
    Loads, validates, and provides access to all of Nerion's capabilities
    from the single config/intents.yaml source of truth.
    """
    def __init__(self, path: str = "config/intents.yaml"):
        self.rules: List[IntentSpec] = self._load_and_validate(path)
        self.triage_rules: Dict[str, IntentSpec] = self._build_triage_rulebook()

    def _load_and_validate(self, path: str) -> List[IntentSpec]:
        """Loads intents from YAML and validates them using Pydantic models."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # Use Pydantic to parse and validate the entire structure
            manifest = IntentManifest(**data)
            
            # Compile regex patterns after validation
            for rule in manifest.intents:
                rule.regexes = [re.compile(p, re.IGNORECASE) for p in rule.patterns if p]
            
            # Sort by priority, highest first
            manifest.intents.sort(key=lambda r: r.priority, reverse=True)
            return manifest.intents
        except Exception as e:
            # In a real scenario, you'd want to log this error.
            print(f"[ERROR] Failed to load or validate intents from {path}: {e}")
            return []

    def _build_triage_rulebook(self) -> Dict[str, IntentSpec]:
        """Creates the simple trigger->intent mapping for the fast, deterministic check."""
        rulebook = {}
        for rule in self.rules:
            if rule.is_triage_rule:
                for pattern_str in rule.patterns:
                    # For the triage book, we use the simple string as the key
                    # This assumes triage patterns are simple, non-regex strings for direct matching.
                    # For more complex triage, the `detect_triage_intent` method uses regex.
                    trigger = re.sub(r'\\b|\^|\$', '', pattern_str).replace('(','').replace(')','').split('|')[0]
                    if trigger:
                        rulebook[trigger] = rule
        return rulebook

    def detect_triage_intent(self, text: str) -> Optional[IntentSpec]:
        """
        The "Triage Nurse" check. Matches against rules marked `is_triage_rule: true`.
        Uses compiled regex for more robust matching.
        """
        if not text:
            return None
        
        normalized_text = text.strip().lower()
        for rule in self.rules:
            if rule.is_triage_rule and rule.regexes:
                if any(rx.search(normalized_text) for rx in rule.regexes):
                    return rule
        return None
    
    def build_manifest_for_parent(self) -> str:
        """
        Builds the complete, formatted "Tool Manifest" string to be injected
        into the Parent LLM's prompt.
        """
        lines = []
        # Filter for tools the Parent should know about (e.g., not exit commands)
        parent_tools = [
            rule for rule in self.rules 
            if not rule.name.startswith("local.exit") and not rule.name.startswith("local.toggle")
        ]

        for tool in parent_tools:
            lines.append(f"- Tool Name: `{tool.name}`")
            lines.append(f"  Description: {tool.description}")
            lines.append(f"  Requires Network: {'yes' if 'web.' in tool.name else 'no'}")
        
        return "\n".join(lines)

# It's good practice to create a singleton instance that the rest of
# the application can import and use.
intent_registry = IntentRegistry()

# Helper function to load intents (for engine.py)
def load_intents(path: Optional[str] = None) -> List[IntentSpec]:
    """Load and return intent rules.

    - When `path` is provided, load directly from that YAML path.
    - Otherwise, return the singleton registry rules.
    """
    if path:
        reg = IntentRegistry(path)
        return reg.rules
    return intent_registry.rules

# Function to parse site query intent (for routes_web.py)
def parse_site_query_intent(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a site-query instruction from free text.

    Returns dict with at least:
      - url: str
      - query: str (defaults to full text if no explicit question extracted)
      - augment: bool (optional)
      - allow: Optional[list[str]] (optional domain allowlist)
    or None if no URL is present.
    """
    if not text:
        return None

    url_pattern = re.compile(r"(https?://[^\s]+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")
    m = url_pattern.search(text)
    if not m:
        return None
    url = m.group(1)

    # Very light query extraction: remove the URL from text and trim
    query = (text[:m.start()] + text[m.end():]).strip() or text.strip()

    # Optional flags (best-effort)
    augment = bool(re.search(r"\bwith\s+(?:external\s+)?sources?\b|\baugment\b", text, flags=re.I))
    allow = None
    allow_match = re.search(r"sources?\s+from\s+([^.;]+)", text, flags=re.I)
    if allow_match:
        allow = [s.strip() for s in re.split(r",|;|\band\b", allow_match.group(1)) if s.strip()]

    return {"url": url, "query": query, "augment": augment, "allow": allow}

def detect_intent(text: str, rules=None) -> Optional[IntentSpec]:
    """
    Detect an intent based on the provided text.
    If rules are provided, search them directly; otherwise use the intent registry.
    """
    if not text:
        return None
    
    # If rules are provided directly (from STATE._intent_rules)
    if rules and isinstance(rules, list):
        normalized_text = text.strip().lower()
        for rule in rules:
            if hasattr(rule, 'regexes') and rule.regexes:
                if any(rx.search(normalized_text) for rx in rule.regexes):
                    return rule
        return None
    
    # Otherwise use the registry
    return intent_registry.detect_triage_intent(text)

def call_handler(intent_spec: IntentSpec, text: str, **kwargs) -> Any:
    """
    Call the appropriate handler for an intent with the provided text and arguments.
    """
    if not intent_spec or not intent_spec.handler:
        return None
        
    try:
        # Split the handler path into module and function
        module_path, func_name = intent_spec.handler.rsplit(':', 1)
        
        # Import the module dynamically
        import importlib
        module = importlib.import_module(module_path)
        
        # Get the function
        handler_func = getattr(module, func_name)
        
        # Call the handler with the text and any additional arguments
        return handler_func(text, **kwargs)
    except Exception as e:
        print(f"Error calling handler for intent {intent_spec.name}: {e}")
        return None

def maybe_offline_intent(text: str) -> tuple[Optional[IntentSpec], Optional[str]]:
    """
    Check if the text matches any offline intent.
    Returns a tuple of (intent_spec, handler_name) if found, otherwise (None, None).
    """
    if not text:
        return None, None
        
    intent = detect_intent(text)
    if not intent:
        return None, None
        
    # Only consider local intents that don't require network
    if intent.name.startswith('local.'):
        handler_name = intent.handler.split(':')[-1] if ':' in intent.handler else intent.handler
        return intent, handler_name
        
    return None, None
