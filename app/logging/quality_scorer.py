"""Response quality scoring for learning system.

Evaluates whether an LLM response was actually helpful or just a cop-out.
"""
from __future__ import annotations
import re
from typing import Dict, Any, Optional


def evaluate_response_quality(
    user_query: str,
    response_text: str,
    action_taken: Dict[str, Any],
    error: Optional[str] = None,
) -> bool:
    """Evaluate if a response was genuinely helpful.

    Args:
        user_query: What the user asked for
        response_text: The assistant's response
        action_taken: Dictionary of actions/tools used
        error: Any error that occurred

    Returns:
        True if quality response, False if unhelpful/cop-out
    """
    # Hard failures
    if error:
        return False

    if not response_text or not response_text.strip():
        return False

    # Detect cop-out patterns (case-insensitive)
    response_lower = response_text.lower()

    # Common unhelpful patterns
    unhelpful_patterns = [
        r"i don'?t have access to",
        r"i can'?t (access|see|view|read|tell you)",
        r"i'?m unable to",
        r"i do not have (access|information)",
        r"you'?ll need to (run|use|check)",
        r"i don'?t know",
        r"i cannot (access|help|provide)",
    ]

    for pattern in unhelpful_patterns:
        if re.search(pattern, response_lower):
            # Check if they at least provided an alternative
            helpful_followup = any([
                "however" in response_lower,
                "instead" in response_lower,
                "alternatively" in response_lower,
                "you can" in response_lower,
            ])

            if not helpful_followup:
                return False  # Cop-out without alternative

    # Check for suspiciously short responses to complex queries
    query_words = len(user_query.split())
    response_chars = len(response_text.strip())

    # Complex query (>5 words) with very short response (<50 chars) = likely unhelpful
    if query_words > 5 and response_chars < 50:
        return False

    # Check if tools were used for actionable queries
    action_keywords_in_query = any([
        re.search(r"\b(review|check|analyze|find|search|show|list)\b", user_query, re.I),
        re.search(r"\b(file|code|function|class)\b", user_query, re.I),
        re.search(r"\b(recent|modified|changed)\b", user_query, re.I),
    ])

    if action_keywords_in_query:
        # Should have used tools or provided specific info
        routed = action_taken.get("routed", "")
        steps = action_taken.get("steps", [])

        # If it's just llm_fallback with no tools, that's suspicious for actionable queries
        if routed == "llm_fallback" and not steps:
            # Check if response has specific content (code, file paths, data)
            has_specific_content = any([
                re.search(r"```", response_text),  # Code blocks
                re.search(r"/[\w/]+\.\w+", response_text),  # File paths
                re.search(r"\d+", response_text),  # Numbers/data (any digits)
                len(response_text) > 150,  # Substantial response
            ])

            if not has_specific_content:
                return False  # Actionable query with no action = failure

    # If we got here, it seems like a reasonable response
    return True


def calculate_response_score(
    user_query: str,
    response_text: str,
    action_taken: Dict[str, Any],
    error: Optional[str] = None,
) -> float:
    """Calculate a 0.0-1.0 quality score.

    Returns:
        Float between 0.0 (terrible) and 1.0 (excellent)
    """
    if not evaluate_response_quality(user_query, response_text, action_taken, error):
        return 0.0  # Complete failure

    # Start with base score
    score = 0.7

    # Bonus for tool usage on actionable queries
    steps = action_taken.get("steps", [])
    if steps and len(steps) > 0:
        score += 0.15

    # Bonus for substantive responses
    if len(response_text) > 200:
        score += 0.1

    # Bonus for code/data/examples
    if re.search(r"```", response_text):
        score += 0.05

    return min(1.0, score)
