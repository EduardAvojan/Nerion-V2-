#!/usr/bin/env python3
"""Quick test of GitHub API to debug the scraper."""

import requests
import sys

# You can pass token as argument
token = sys.argv[1] if len(sys.argv) > 1 else None

if not token:
    print("❌ No token provided")
    print("Usage: python test_github_api.py ghp_your_token")
    sys.exit(1)

print(f"✅ Token: {token[:10]}...")

url = "https://api.github.com/search/commits"
params = {
    "q": "language:python fix bug",
    "per_page": 3,
    "page": 1,
    "sort": "committer-date",
    "order": "desc"
}
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.cloak-preview+json"
}

print(f"\n🔍 Searching: {params['q']}")
response = requests.get(url, params=params, headers=headers, timeout=30)

print(f"📊 Status: {response.status_code}")

if response.status_code != 200:
    print(f"❌ Error: {response.text}")
    sys.exit(1)

data = response.json()
total = data.get('total_count', 0)
items = data.get('items', [])

print(f"✅ Total available: {total:,}")
print(f"✅ Fetched: {len(items)}")

if items:
    print("\n📝 Sample commits:")
    for i, commit in enumerate(items[:3], 1):
        sha = commit['sha'][:8]
        message = commit['commit']['message'].split('\n')[0][:60]
        print(f"  {i}. {sha} - {message}")
else:
    print("❌ No commits returned!")
    print("Response:", data)
