#!/usr/bin/env python3
"""Check learning system quality tracking.

Analyzes recent experience logs to show quality scoring in action.
"""
import json
from pathlib import Path

LOG_PATH = Path('out/experience/log.jsonl')

def analyze_recent_logs(n=20):
    """Analyze the last N log entries."""
    if not LOG_PATH.exists():
        print(f"No log file found at {LOG_PATH}")
        return

    lines = LOG_PATH.read_text(encoding='utf-8').splitlines()
    recent = lines[-n:]

    successes = 0
    failures = 0
    details = []

    for ln in recent:
        try:
            rec = json.loads(ln)
        except Exception:
            continue

        query = rec.get('user_query', '')[:50]
        routed = rec.get('action_taken', {}).get('routed', '')
        success = rec.get('outcome_success', False)
        error = rec.get('error')

        if success:
            successes += 1
            status = '✓'
        else:
            failures += 1
            status = '✗'

        details.append({
            'status': status,
            'query': query,
            'routed': routed,
            'error': error
        })

    print(f"\n{'='*70}")
    print(f"Learning Quality Report (last {n} interactions)")
    print(f"{'='*70}\n")

    total = successes + failures
    if total > 0:
        success_rate = (successes / total) * 100
        print(f"Success Rate: {successes}/{total} ({success_rate:.1f}%)")
        print(f"Failures: {failures}\n")
    else:
        print("No interactions found\n")

    print(f"{'Status':<8} {'Route':<20} {'Query':<45}")
    print(f"{'-'*70}")

    for d in details:
        print(f"{d['status']:<8} {d['routed']:<20} {d['query']:<45}")
        if d['error']:
            print(f"         Error: {d['error'][:60]}")

    print(f"\n{'='*70}")
    print("\nKey:")
    print("  ✓ = Quality response (helpful, used tools when needed)")
    print("  ✗ = Poor quality (cop-out, error, or didn't use tools for actionable query)")
    print(f"\nNOTE: Only 'llm_fallback' routes use quality scoring.")
    print(f"      Other routes (offline_fast, local_rule, etc.) are hardcoded as successful.\n")

if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    analyze_recent_logs(n)
