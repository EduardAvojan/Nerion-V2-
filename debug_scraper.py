#!/usr/bin/env python3
"""Debug why scraper returns 0 commits."""

import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

token = os.getenv("GITHUB_TOKEN")
print(f"Token loaded: {bool(token)}")
if token:
    print(f"Token: {token[:10]}...")

# Test API connector
from nerion_digital_physicist.data_mining.github_api_connector import GitHubAPIConnector

connector = GitHubAPIConnector(token=token)

# Test the exact query from build_search_queries
query = "language:python fix bug is:public stars:>10"
print(f"\nTesting query: {query}")

count = 0
for commit in connector.search_commits(query, max_results=5):
    count += 1
    print(f"Commit {count}: {commit['sha'][:8]} - {commit['commit']['message'][:50]}")

print(f"\nTotal fetched: {count}")

if count == 0:
    print("\n⚠️  Query returned 0 results. Testing simpler query...")
    query2 = "language:python fix bug"
    print(f"Query: {query2}")

    count2 = 0
    for commit in connector.search_commits(query2, max_results=5):
        count2 += 1
        print(f"Commit {count2}: {commit['sha'][:8]}")

    print(f"Simpler query returned: {count2} commits")
