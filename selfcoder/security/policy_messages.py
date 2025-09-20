from typing import Dict

_DOCS = {
    "plugins": "docs/policy.md#plugins",
    "paths": "docs/policy.md#paths",
    "actions": "docs/policy.md#actions",
    "size": "docs/policy.md#limits",
    "secrets": "docs/policy.md#secrets",
}

def explain_policy(block: Dict) -> Dict:
    """
    Transform a raw policy block dict into a human message.
    block keys expected: kind ('paths'|'actions'|'size'|'secrets'), rule, value, path/action, detail
    Returns: dict(rule, reason, hint, doc)
    """
    kind = block.get("kind", "paths")
    rule = block.get("rule", "deny_paths")
    value = block.get("value", "")
    subject = block.get("path") or block.get("action") or ""
    reason = ""
    hint = ""
    if kind == "paths":
        reason = f"{subject} is denied by {rule} '{value}'"
        hint = "Move file or adjust .nerion/policy.yaml (allow_paths/deny_paths)."
        doc = _DOCS["paths"]
    elif kind == "actions":
        reason = f"Action '{subject}' is denied by {rule}"
        hint = "Enable in actions.allow or remove from actions.deny."
        doc = _DOCS["actions"]
    elif kind == "size":
        reason = f"Change exceeds limit '{rule}'={value}"
        hint = "Reduce diff size or raise limit in policy.limits."
        doc = _DOCS["size"]
    elif kind == "secrets":
        reason = "Secret/PII scan blocked write"
        hint = "Remove secrets/PII or disable only if you accept the risk."
        doc = _DOCS["secrets"]
    else:
        reason = "Policy violation"
        hint = "Check your policy file and logs."
        doc = "docs/policy.md"
    return {"rule": rule, "reason": reason, "hint": hint, "doc": doc}
