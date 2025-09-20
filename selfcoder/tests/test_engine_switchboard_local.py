# selfcoder/tests/test_engine_switchboard_local.py
import types
import textwrap
from pathlib import Path

from app.chat.intents import load_intents, detect_intent, call_handler

# We unit-test the same flow engine uses (rules -> handler) without spinning voice.
def test_engine_like_local_dispatch(tmp_path: Path, monkeypatch):
    # temp intents.yaml with a local rule
    yaml = textwrap.dedent(r"""
    intents:
      - name: local.get_current_date
        priority: 100
        patterns:
          - "\\b(what's today's date|what is the date|date please)\\b"
        handler: app.chat.offline_tools:get_current_date
    """).strip()
    (tmp_path / "config").mkdir(exist_ok=True)
    cfg = tmp_path / "config" / "intents.yaml"
    cfg.write_text(yaml, encoding="utf-8")

    # Mimic engine's rule load
    rules = load_intents(str(cfg))
    utterance = "hey nerion, what's today's date?"
    rule = detect_intent(utterance, rules)
    assert rule and rule.name == "local.get_current_date"

    # Call handler like engine would
    out = call_handler(rule, utterance)
    assert isinstance(out, str) and "date" in out.lower()