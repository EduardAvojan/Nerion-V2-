#!/usr/bin/env python3
"""Test which queries actually return results."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from nerion_digital_physicist.data_mining.github_api_connector import GitHubAPIConnector

token = os.getenv("GITHUB_TOKEN")
connector = GitHubAPIConnector(token=token)

# Test different query strategies
test_queries = [
    # Broad queries
    "language:python fix",
    "language:python fixed",
    "language:python bug",
    "language:python bugfix",
    "language:python refactor",
    "language:python improve",

    # With date ranges (quarterly)
    "language:python fix pushed:2024-07-01..2024-09-30",
    "language:python fix pushed:2024-04-01..2024-06-30",
    "language:python fix pushed:2024-01-01..2024-03-31",

    # With size filters
    "language:python fix size:1..10000",
    "language:python fix size:10000..50000",

    # With file patterns
    "language:python fix filename:.py",

    # Framework specific
    "language:python fix django",
    "language:python fix flask",
    "language:python fix fastapi",
]

print("Testing query effectiveness...\n")
results = {}

for query in test_queries:
    print(f"Testing: {query}")
    count = 0
    try:
        for commit in connector.search_commits(query, max_results=100):
            count += 1
            if count >= 100:
                break
        results[query] = count
        print(f"  ✅ Found {count} commits\n")
    except Exception as e:
        results[query] = 0
        print(f"  ❌ Error: {e}\n")

print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Sort by count
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for query, count in sorted_results:
    status = "✅" if count >= 50 else "⚠️" if count > 0 else "❌"
    print(f"{status} {count:3d} commits: {query}")

print(f"\n✅ Good queries (50+): {len([c for c in results.values() if c >= 50])}")
print(f"⚠️  Weak queries (1-49): {len([c for c in results.values() if 0 < c < 50])}")
print(f"❌ Failed queries (0): {len([c for c in results.values() if c == 0])}")
