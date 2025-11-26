#!/usr/bin/env python3
"""
Bug Miner - Extract real bug fixes from git history for GNN training.

This script mines bug fix commits from training_ground repositories,
extracting before/after code pairs to train Nerion on REAL bugs.

Categories (all 20):
- Bug fixes: attribute_error, type_error, value_error, index_error, key_error,
             import_error, syntax_error, logic_error
- Code quality: complexity_reduction, naming_improvement, code_duplication,
                maintainability, readability
- Architecture: design_pattern, dependency_management, modularity
- Security: injection_prevention, secret_management
- Performance & Types: performance_optimization, type_safety
"""

import subprocess
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_GROUND = PROJECT_ROOT / "training_ground"
MINED_BUGS_PATH = PROJECT_ROOT / "data" / "mined_bugs.json"


@dataclass
class MinedBug:
    """A bug extracted from git history."""
    bug_id: str
    repo: str
    commit_hash: str
    commit_message: str
    file_path: str
    language: str
    code_before: str
    code_after: str
    bug_category: str
    error_type: str  # One of the 20 categories
    confidence: float  # How confident we are in the classification


# Keywords for classifying bug types (maps to all 20 categories)
BUG_PATTERNS = {
    # Bug fixes (8 classes) - from actual runtime errors
    'attribute_error': [
        r'AttributeError', r'has no attribute', r'attribute.*error',
        r'\.(\w+)\s+not found', r'object has no'
    ],
    'type_error': [
        r'TypeError', r'type.*error', r'expected.*got', r'not.*callable',
        r'unsupported operand', r'argument.*type', r'NoneType'
    ],
    'value_error': [
        r'ValueError', r'value.*error', r'invalid.*value', r'out of range',
        r'invalid literal', r'could not convert'
    ],
    'index_error': [
        r'IndexError', r'index.*out', r'list index', r'tuple index',
        r'index.*range', r'out of bounds'
    ],
    'key_error': [
        r'KeyError', r'key.*error', r'key.*not found', r'missing key',
        r'dict.*key', r'no such key'
    ],
    'import_error': [
        r'ImportError', r'ModuleNotFoundError', r'import.*error',
        r'cannot import', r'no module named', r'circular import'
    ],
    'syntax_error': [
        r'SyntaxError', r'IndentationError', r'syntax.*error',
        r'invalid syntax', r'unexpected indent', r'parsing error'
    ],
    'logic_error': [
        r'logic.*error', r'wrong.*result', r'incorrect.*behavior',
        r'off.by.one', r'infinite loop', r'race condition', r'deadlock',
        r'assertion.*fail', r'AssertionError'
    ],

    # Code quality (5 classes)
    'complexity_reduction': [
        r'simplif', r'refactor', r'clean.?up', r'reduce.*complexity',
        r'cyclomatic', r'nested.*too', r'extract.*method'
    ],
    'naming_improvement': [
        r'rename', r'naming', r'typo', r'spell', r'variable.*name',
        r'clarif.*name', r'better.*name'
    ],
    'code_duplication': [
        r'duplicat', r'DRY', r'repeated', r'redundant', r'consolidat',
        r'shared.*code', r'common.*code'
    ],
    'maintainability': [
        r'maintain', r'readab', r'clean', r'SOLID', r'decouple',
        r'modular', r'organiz'
    ],
    'readability': [
        r'readab', r'document', r'comment', r'docstring', r'clarity',
        r'understandab'
    ],

    # Architecture (3 classes)
    'design_pattern': [
        r'pattern', r'factory', r'singleton', r'observer', r'strategy',
        r'decorator', r'adapter', r'facade'
    ],
    'dependency_management': [
        r'depend', r'coupling', r'injection', r'import.*order',
        r'circular.*dep', r'version.*bump'
    ],
    'modularity': [
        r'modular', r'split', r'separate', r'extract.*class',
        r'god.*class', r'single.*responsibility'
    ],

    # Security (2 classes)
    'injection_prevention': [
        r'SQL.*inject', r'XSS', r'injection', r'sanitiz', r'escap',
        r'CSRF', r'security.*fix', r'vulnerab', r'CVE'
    ],
    'secret_management': [
        r'secret', r'password', r'credential', r'API.*key', r'token',
        r'hardcoded', r'env.*var', r'config.*leak'
    ],

    # Performance & Type Safety (2 classes)
    'performance_optimization': [
        r'perf', r'optimi', r'speed', r'faster', r'slow', r'memory.*leak',
        r'cache', r'O\(n\)', r'bottleneck', r'efficient'
    ],
    'type_safety': [
        r'type.*hint', r'typing', r'mypy', r'annotation', r'type.*check',
        r'generic', r'TypeVar', r'Protocol'
    ],
}


def classify_bug(commit_message: str, diff_content: str) -> Tuple[str, float]:
    """
    Classify a bug into one of 20 categories based on commit message and diff.
    Returns (category, confidence).
    """
    text = f"{commit_message}\n{diff_content}".lower()

    scores: Dict[str, int] = {cat: 0 for cat in BUG_PATTERNS}

    for category, patterns in BUG_PATTERNS.items():
        for pattern in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            scores[category] += matches

    # Get best match
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    if best_score == 0:
        # Default to logic_error for generic "fix" commits
        return 'logic_error', 0.3

    # Confidence based on score
    confidence = min(1.0, best_score / 5.0)
    return best_category, confidence


def get_file_at_commit(repo_path: Path, commit: str, file_path: str) -> Optional[str]:
    """Get file content at a specific commit."""
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{file_path}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def get_bug_fix_commits(repo_path: Path, max_commits: int = 100) -> List[dict]:
    """Get commits that look like bug fixes."""
    try:
        # Find commits with fix/bug/error in message
        result = subprocess.run(
            ['git', 'log', '--oneline', f'-{max_commits * 3}',
             '--grep=fix', '--grep=bug', '--grep=error', '--all-match'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Also get commits without all-match (OR instead of AND)
        result2 = subprocess.run(
            ['git', 'log', '--oneline', f'-{max_commits}', '--grep=fix'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        commits = []
        seen = set()

        for line in (result.stdout + '\n' + result2.stdout).strip().split('\n'):
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) >= 2:
                commit_hash, message = parts
                if commit_hash not in seen:
                    seen.add(commit_hash)
                    commits.append({'hash': commit_hash, 'message': message})

        return commits[:max_commits]
    except Exception as e:
        print(f"Error getting commits from {repo_path}: {e}")
        return []


def get_changed_files(repo_path: Path, commit: str) -> List[Tuple[str, str]]:
    """Get list of changed files in a commit with their status."""
    try:
        result = subprocess.run(
            ['git', 'diff-tree', '--no-commit-id', '--name-status', '-r', commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        files = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                status, filepath = parts[0], parts[-1]
                # Only include modified files (not added/deleted)
                if status == 'M':
                    files.append((status, filepath))
        return files
    except Exception:
        return []


def get_diff_for_file(repo_path: Path, commit: str, file_path: str) -> Optional[str]:
    """Get the diff for a specific file in a commit."""
    try:
        result = subprocess.run(
            ['git', 'show', commit, '--', file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def mine_bugs_from_repo(repo_name: str, max_bugs: int = 50) -> List[MinedBug]:
    """Mine bug fixes from a single repository."""
    repo_path = TRAINING_GROUND / repo_name

    if not repo_path.exists():
        print(f"  Repo not found: {repo_path}")
        return []

    # Check if it has git history
    if not (repo_path / '.git').exists():
        print(f"  No git history in {repo_name}")
        return []

    print(f"  Mining {repo_name}...")

    commits = get_bug_fix_commits(repo_path, max_commits=max_bugs * 2)
    print(f"    Found {len(commits)} fix commits")

    bugs = []

    for commit_info in commits:
        if len(bugs) >= max_bugs:
            break

        commit = commit_info['hash']
        message = commit_info['message']

        # Get changed files
        changed_files = get_changed_files(repo_path, commit)

        for status, filepath in changed_files:
            # Only Python and JavaScript files
            ext = Path(filepath).suffix.lower()
            if ext not in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                continue

            language = 'python' if ext == '.py' else 'javascript'

            # Get before/after code
            code_after = get_file_at_commit(repo_path, commit, filepath)
            code_before = get_file_at_commit(repo_path, f'{commit}^', filepath)

            if not code_before or not code_after:
                continue

            # Skip if files are too large or too similar
            if len(code_before) > 50000 or len(code_after) > 50000:
                continue
            if code_before == code_after:
                continue

            # Get diff for classification
            diff = get_diff_for_file(repo_path, commit, filepath)

            # Classify the bug
            error_type, confidence = classify_bug(message, diff or '')

            # Create bug ID
            bug_id = hashlib.md5(f"{repo_name}:{commit}:{filepath}".encode()).hexdigest()[:12]

            bug = MinedBug(
                bug_id=bug_id,
                repo=repo_name,
                commit_hash=commit,
                commit_message=message,
                file_path=filepath,
                language=language,
                code_before=code_before,
                code_after=code_after,
                bug_category=error_type.split('_')[0],  # e.g., "attribute" from "attribute_error"
                error_type=error_type,
                confidence=confidence
            )
            bugs.append(bug)

            if len(bugs) >= max_bugs:
                break

    print(f"    Mined {len(bugs)} bugs from {repo_name}")
    return bugs


def mine_all_repos(max_bugs_per_repo: int = 30) -> List[MinedBug]:
    """Mine bugs from all training_ground repositories."""
    repos = ['flask', 'rich', 'click', 'httpx', 'requests', 'express', 'lodash']

    all_bugs = []

    print("Mining bugs from git history...")
    for repo in repos:
        bugs = mine_bugs_from_repo(repo, max_bugs=max_bugs_per_repo)
        all_bugs.extend(bugs)

    print(f"\nTotal bugs mined: {len(all_bugs)}")

    # Print distribution
    from collections import Counter
    dist = Counter(b.error_type for b in all_bugs)
    print("\nCategory distribution:")
    for cat, count in dist.most_common():
        print(f"  {cat}: {count}")

    return all_bugs


def save_mined_bugs(bugs: List[MinedBug]):
    """Save mined bugs to JSON file."""
    MINED_BUGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'mined_at': datetime.now().isoformat(),
        'total_bugs': len(bugs),
        'bugs': [asdict(b) for b in bugs]
    }

    with open(MINED_BUGS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(bugs)} bugs to {MINED_BUGS_PATH}")


def load_mined_bugs() -> List[MinedBug]:
    """Load previously mined bugs."""
    if not MINED_BUGS_PATH.exists():
        return []

    with open(MINED_BUGS_PATH) as f:
        data = json.load(f)

    return [MinedBug(**b) for b in data['bugs']]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mine bugs from git history')
    parser.add_argument('--max-per-repo', type=int, default=30,
                        help='Max bugs to mine per repo')
    parser.add_argument('--repo', type=str, default=None,
                        help='Mine only this repo')
    args = parser.parse_args()

    if args.repo:
        bugs = mine_bugs_from_repo(args.repo, max_bugs=args.max_per_repo)
    else:
        bugs = mine_all_repos(max_bugs_per_repo=args.max_per_repo)

    if bugs:
        save_mined_bugs(bugs)
