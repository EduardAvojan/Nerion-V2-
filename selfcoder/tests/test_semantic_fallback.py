# selfcoder/tests/test_semantic_fallback.py
from app.chat.semantic_parser import configure_intents, configure_model, parse_intent_by_similarity

def test_semantic_fallback_without_model_offline():
    # No model directory provided → it will use fuzzy matching
    ok = configure_model(local_model_dir=None)
    assert ok is False

    # Provide example phrases as data (no literals in codebase)
    configure_intents(examples={
        "local.get_current_time": [
            "what time is it",
            "do you have the time",
            "time please",
        ],
        "web.search": [
            "search the web",
            "look up the news",
        ],
    })

    # Fuzzy phrasing that doesn't exactly match regex
    intent = parse_intent_by_similarity("could you tell me the time now?")
    assert intent == "local.get_current_time"