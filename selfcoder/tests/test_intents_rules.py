# selfcoder/tests/test_intents_rules.py
import textwrap
from pathlib import Path

from app.chat.intents import load_intents, detect_intent, call_handler

def test_rules_load_match_and_call(tmp_path: Path):
    yaml = textwrap.dedent(r"""
    intents:
      - name: local.get_current_time
        priority: 100
        patterns:
          - "\\b(what time is it|time please)\\b"
        handler: app.chat.offline_tools:get_current_time
    """).strip()
    (tmp_path / "config").mkdir(exist_ok=True)
    f = tmp_path / "config" / "intents.yaml"
    f.write_text(yaml, encoding="utf-8")

    # Load rules from temp config
    rules = load_intents(str(f))
    assert rules and rules[0].name == "local.get_current_time"

    # Detect rule by text
    rule = detect_intent("hey, what time is it", rules)
    assert rule and rule.name == "local.get_current_time"

    # Call handler (offline)
    out = call_handler(rule, "what time is it")
    assert isinstance(out, str) and out  # should produce a friendly time string