from selfcoder.cli_ext.trace import summarize_since
def test_summarize_since():
    lines = [
      '{"ts": 9999999999, "action":"gate","ms":120,"result":"OK"}',
      '{"ts": 9999999999, "action":"apply","ms":50,"result":"BLOCKED"}'
    ]
    s = summarize_since(lines, since_sec=10**10)  # include both
    assert s["actions"] == 2 and s["blocked"] == 1 and s["time_ms"] == 170
