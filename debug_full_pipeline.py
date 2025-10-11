#!/usr/bin/env python3
"""Debug the full scraper pipeline to see where commits fail."""

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

from nerion_digital_physicist.data_mining.github_api_connector import GitHubAPIConnector
from nerion_digital_physicist.data_mining.github_quality_scraper import (
    GitHubQualityScraper,
    QualityThresholds,
)

token = os.getenv("GITHUB_TOKEN")
connector = GitHubAPIConnector(token=token)
scraper = GitHubQualityScraper(
    db_path=Path("debug_test.db"),
    github_token=token,
    thresholds=QualityThresholds(min_quality_score=60)
)

query = "language:python fix bug"
print(f"🔍 Testing full pipeline: {query}\n")

stats = {
    "fetched": 0,
    "message_filter_rejected": 0,
    "file_filter_rejected": 0,
    "no_python_files": 0,
    "extraction_failed": 0,
    "size_filter_rejected": 0,
    "syntax_filter_rejected": 0,
    "quality_filter_rejected": 0,
    "accepted": 0,
}

for commit_json in connector.search_commits(query, max_results=20):
    stats["fetched"] += 1

    sha = commit_json["sha"][:8]
    repo = commit_json.get("repository", {}).get("full_name", "unknown")
    message = commit_json["commit"]["message"].split('\n')[0][:60]

    print(f"\n{'='*70}")
    print(f"Commit {stats['fetched']}: {sha} - {repo}")
    print(f"Message: {message}")

    # Extract commit metadata
    commit_data = connector.extract_commit_data(commit_json)
    if not commit_data:
        print("  ❌ Failed to extract commit data")
        continue

    # Stage 1: Message filter
    if not scraper.passes_message_filter(commit_data):
        print("  ❌ REJECTED: Message filter")
        stats["message_filter_rejected"] += 1
        continue
    print("  ✅ Passed: Message filter")

    # Stage 2: File filter
    if not scraper.passes_file_filter(commit_data):
        print("  ❌ REJECTED: File filter")
        stats["file_filter_rejected"] += 1
        continue
    print("  ✅ Passed: File filter")

    # Fetch full commit details
    full_commit = connector.fetch_commit_diff(repo, commit_data.sha)
    if not full_commit:
        print("  ❌ Failed to fetch full commit details")
        continue

    # Update files list from full commit (search results don't include files)
    commit_data.files = [f["filename"] for f in full_commit.get("files", [])]

    # Extract code changes
    python_files = [f for f in commit_data.files if f.endswith('.py')]
    if not python_files:
        print("  ❌ No Python files")
        stats["no_python_files"] += 1
        continue

    print(f"  📁 Python files: {len(python_files)}")

    extracted = False
    for file_path in python_files:
        print(f"    Trying: {file_path}")

        changes = connector.extract_file_changes(full_commit, file_path, repo)
        if not changes:
            print(f"      ❌ Extraction failed")
            continue

        before_code, after_code = changes
        print(f"      ✅ Extracted: {len(before_code)} -> {len(after_code)} chars")

        commit_data.before_code = before_code
        commit_data.after_code = after_code

        # Stage 3: Size filter
        if not scraper.passes_size_filter(before_code, after_code):
            print(f"      ❌ REJECTED: Size filter")
            stats["size_filter_rejected"] += 1
            continue
        print(f"      ✅ Passed: Size filter")

        # Stage 4: Syntax validation
        if not scraper.validate_syntax(before_code, after_code):
            print(f"      ❌ REJECTED: Syntax validation")
            stats["syntax_filter_rejected"] += 1
            continue
        print(f"      ✅ Passed: Syntax validation")

        # Stage 5: Quality assessment
        if not scraper.assess_quality(commit_data):
            print(f"      ❌ REJECTED: Quality score {commit_data.quality_score}/100 < 60")
            stats["quality_filter_rejected"] += 1
            continue
        print(f"      ✅ Passed: Quality score {commit_data.quality_score}/100")

        # Success!
        stats["accepted"] += 1
        extracted = True
        print(f"      🎉 ACCEPTED!")
        break

    if not extracted:
        stats["extraction_failed"] += 1

print(f"\n\n{'='*70}")
print("PIPELINE STATISTICS")
print(f"{'='*70}")
print(f"Fetched: {stats['fetched']}")
print(f"  ├─ Message filter rejected: {stats['message_filter_rejected']}")
print(f"  ├─ File filter rejected: {stats['file_filter_rejected']}")
print(f"  ├─ No Python files: {stats['no_python_files']}")
print(f"  ├─ Extraction failed (all files): {stats['extraction_failed']}")
print(f"  ├─ Size filter rejected: {stats['size_filter_rejected']}")
print(f"  ├─ Syntax filter rejected: {stats['syntax_filter_rejected']}")
print(f"  ├─ Quality filter rejected: {stats['quality_filter_rejected']}")
print(f"  └─ ACCEPTED: {stats['accepted']}")
print(f"\nAcceptance rate: {stats['accepted']/stats['fetched']*100:.1f}%")
