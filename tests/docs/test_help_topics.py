import os
def test_help_topics_exist():
    for f in ["policy-blocked.md","gate-high-risk.md","tests-failed.md","node-bridge-missing.md","prettier-conflict.md"]:
        assert os.path.exists(f"docs/help/{f}")
