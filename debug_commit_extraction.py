#!/usr/bin/env python3
"""Debug why code extraction is failing for all commits."""

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

token = os.getenv("GITHUB_TOKEN")
connector = GitHubAPIConnector(token=token)

query = "language:python fix bug"
print(f"🔍 Testing query: {query}\n")

commit_count = 0
extraction_failures = {
    "no_files": 0,
    "non_python": 0,
    "added_file": 0,
    "removed_file": 0,
    "no_patch": 0,
    "fetch_failed": 0,
    "extraction_success": 0,
}

for commit_json in connector.search_commits(query, max_results=20):
    commit_count += 1

    sha = commit_json["sha"][:8]
    repo = commit_json.get("repository", {}).get("full_name", "unknown")
    message = commit_json["commit"]["message"].split('\n')[0][:60]

    print(f"\n{'='*70}")
    print(f"Commit {commit_count}: {sha}")
    print(f"Repo: {repo}")
    print(f"Message: {message}")

    # Fetch full commit with file details
    full_commit = connector.fetch_commit_diff(repo, sha)

    if not full_commit:
        print(f"  ❌ Failed to fetch full commit details")
        extraction_failures["fetch_failed"] += 1
        continue

    files = full_commit.get("files", [])
    print(f"  📁 Files changed: {len(files)}")

    if not files:
        print(f"  ❌ No files in commit")
        extraction_failures["no_files"] += 1
        continue

    # Check each file
    python_files = [f for f in files if f["filename"].endswith(".py")]
    print(f"  🐍 Python files: {len(python_files)}")

    if not python_files:
        print(f"  ❌ No Python files")
        extraction_failures["non_python"] += 1
        continue

    # Try to extract changes from first Python file
    for file_data in python_files[:1]:  # Just check first one
        filename = file_data["filename"]
        status = file_data["status"]

        print(f"\n  File: {filename}")
        print(f"  Status: {status}")

        if status == "added":
            print(f"    ❌ SKIPPED: New file (we only want modifications)")
            extraction_failures["added_file"] += 1
            continue

        if status == "removed":
            print(f"    ❌ SKIPPED: Deleted file")
            extraction_failures["removed_file"] += 1
            continue

        # status is "modified" or "renamed" - try to extract
        patch = file_data.get("patch", "")
        if not patch:
            print(f"    ❌ SKIPPED: No patch data")
            extraction_failures["no_patch"] += 1
            continue

        patch_lines = len(patch.split('\n'))
        print(f"    ✅ Has patch: {patch_lines} lines")

        # Try extraction
        changes = connector.extract_file_changes(full_commit, filename, repo)

        if changes:
            before, after = changes
            print(f"    ✅ EXTRACTION SUCCESS!")
            print(f"       Before: {len(before)} chars")
            print(f"       After: {len(after)} chars")
            extraction_failures["extraction_success"] += 1
        else:
            print(f"    ❌ Extraction returned None")
            extraction_failures["fetch_failed"] += 1

        break  # Only check first Python file

print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total commits analyzed: {commit_count}")
print(f"\nFailure breakdown:")
print(f"  - Fetch failed: {extraction_failures['fetch_failed']}")
print(f"  - No files: {extraction_failures['no_files']}")
print(f"  - No Python files: {extraction_failures['non_python']}")
print(f"  - Added file (new): {extraction_failures['added_file']}")
print(f"  - Removed file: {extraction_failures['removed_file']}")
print(f"  - No patch: {extraction_failures['no_patch']}")
print(f"\n  ✅ Extraction SUCCESS: {extraction_failures['extraction_success']}")
print(f"\nSuccess rate: {extraction_failures['extraction_success']/commit_count*100:.1f}%")
