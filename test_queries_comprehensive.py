#!/usr/bin/env python3
"""Comprehensive query testing to find ALL patterns that work."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from nerion_digital_physicist.data_mining.github_api_connector import GitHubAPIConnector

token = os.getenv("GITHUB_TOKEN")
connector = GitHubAPIConnector(token=token)

# Test EVERYTHING
action_words = [
    "fix", "fixed", "fixes", "fixing",
    "bug", "bugfix", "bugs",
    "refactor", "refactored", "refactoring",
    "improve", "improved", "improvement",
    "update", "updated", "updates",
    "patch", "patched", "patches",
    "correct", "corrected", "correction",
    "resolve", "resolved", "resolves",
    "address", "addressed",
    "handle", "handled", "handles",
    "repair", "repaired",
    "enhance", "enhanced",
    "optimize", "optimized",
    "cleanup", "clean",
    "simplify", "simplified",
]

problem_words = [
    "bug", "issue", "problem", "error", "exception",
    "crash", "failure", "fault", "defect",
    "leak", "memory", "performance",
    "security", "vulnerability",
    "race", "deadlock", "thread",
    "null", "none", "missing",
    "wrong", "incorrect", "invalid",
    "broken", "bad", "outdated",
]

frameworks = [
    # Web frameworks
    "django", "flask", "fastapi", "tornado", "pyramid", "bottle",
    "starlette", "sanic", "quart", "falcon",

    # HTTP clients
    "requests", "httpx", "aiohttp", "urllib3",

    # Data science
    "numpy", "pandas", "scipy", "matplotlib", "seaborn",
    "sklearn", "scikit-learn", "tensorflow", "pytorch", "keras",

    # Testing
    "pytest", "unittest", "nose", "mock", "hypothesis",

    # Databases
    "sqlalchemy", "django-orm", "peewee", "pony",
    "psycopg2", "pymongo", "redis", "elasticsearch",

    # Async
    "asyncio", "celery", "dramatiq", "rq",

    # CLI
    "click", "argparse", "typer",

    # Web scraping
    "scrapy", "beautifulsoup", "selenium", "playwright",

    # Common libraries
    "json", "yaml", "xml", "csv", "configparser",
    "logging", "datetime", "pathlib", "typing",
]

print(f"Testing {len(action_words)} action words...")
print(f"Testing {len(problem_words)} problem words...")
print(f"Testing {len(frameworks)} frameworks...")
print(f"Testing combinations...\n")

results = {}

def test_query(query):
    """Test a query and return commit count."""
    try:
        count = 0
        for commit in connector.search_commits(query, max_results=100):
            count += 1
            if count >= 100:
                return 100  # Max out at 100
        return count
    except Exception:
        return 0

# Test single action words
print("Testing action words...")
for word in action_words[:10]:  # Test first 10 to avoid rate limit
    query = f"language:python {word}"
    count = test_query(query)
    if count > 0:
        results[query] = count
        print(f"  ✅ {count:3d}: {query}")

# Test problem words
print("\nTesting problem words...")
for word in problem_words[:10]:
    query = f"language:python {word}"
    count = test_query(query)
    if count > 0:
        results[query] = count
        print(f"  ✅ {count:3d}: {query}")

# Test frameworks
print("\nTesting frameworks...")
for fw in frameworks[:15]:
    query = f"language:python {fw}"
    count = test_query(query)
    if count > 0:
        results[query] = count
        print(f"  ✅ {count:3d}: {query}")

# Test action + problem combinations
print("\nTesting action + problem combinations...")
top_actions = ["fix", "resolve", "handle", "patch", "correct"]
top_problems = ["bug", "error", "issue", "crash", "leak"]
for action in top_actions:
    for problem in top_problems[:3]:  # Test 3 problems per action
        query = f"language:python {action} {problem}"
        count = test_query(query)
        if count > 0:
            results[query] = count
            print(f"  ✅ {count:3d}: {query}")

# Test framework combinations
print("\nTesting framework combinations...")
top_frameworks = ["django", "flask", "pytest", "numpy", "pandas"]
for action in ["fix", "update", "improve"][:2]:
    for fw in top_frameworks[:3]:
        query = f"language:python {action} {fw}"
        count = test_query(query)
        if count > 0:
            results[query] = count
            print(f"  ✅ {count:3d}: {query}")

print("\n" + "="*70)
print("SUCCESSFUL PATTERNS")
print("="*70)

# Group by count
excellent = [(q, c) for q, c in results.items() if c >= 80]
good = [(q, c) for q, c in results.items() if 50 <= c < 80]
okay = [(q, c) for q, c in results.items() if 20 <= c < 50]

print(f"\n🌟 EXCELLENT (80-100 commits): {len(excellent)}")
for q, c in sorted(excellent, key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {c:3d}: {q}")

print(f"\n✅ GOOD (50-79 commits): {len(good)}")
for q, c in sorted(good, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {c:3d}: {q}")

print(f"\n⚠️  OKAY (20-49 commits): {len(okay)}")
for q, c in sorted(okay, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {c:3d}: {q}")

print(f"\n📊 Total successful queries: {len(results)}")
print(f"   Estimated total commits: {sum(results.values()) * 10:,} (scaled to 1000 per query)")
