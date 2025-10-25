#!/usr/bin/env python3
"""
Duplication Checker for Lesson Generation

Checks if a proposed lesson already exists in production database
BEFORE agent wastes time generating it.

Usage:
    python scripts/check_duplicates.py --name "a1_variable_scope_001"
    python scripts/check_duplicates.py --content-hash "abc123..."
    python scripts/check_duplicates.py --topic "variable scope"
"""
import sys
import sqlite3
import hashlib
from pathlib import Path
import argparse

PRODUCTION_DB = Path("out/learning/curriculum.sqlite")


def normalize_code(code: str) -> str:
    """Normalize code for hash comparison (same as SafeCurriculumDB)."""
    import ast
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except:
        return code.strip()


def calculate_content_hash(before_code: str, after_code: str, test_code: str) -> str:
    """Calculate SHA256 hash (same algorithm as SafeCurriculumDB)."""
    norm_before = normalize_code(before_code)
    norm_after = normalize_code(after_code)
    norm_test = normalize_code(test_code)

    combined = f"{norm_before}|||{norm_after}|||{norm_test}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def check_name_exists(name: str) -> tuple[bool, dict]:
    """Check if lesson name already exists."""
    if not PRODUCTION_DB.exists():
        return False, {}

    conn = sqlite3.connect(PRODUCTION_DB)
    cursor = conn.execute(
        "SELECT id, name, description, focus_area FROM lessons WHERE name = ?",
        (name,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return True, {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "focus_area": row[3]
        }
    return False, {}


def check_content_hash_exists(content_hash: str) -> tuple[bool, dict]:
    """Check if lesson content hash already exists."""
    if not PRODUCTION_DB.exists():
        return False, {}

    conn = sqlite3.connect(PRODUCTION_DB)
    cursor = conn.execute(
        "SELECT id, name, description, content_hash FROM lessons WHERE content_hash = ?",
        (content_hash,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return True, {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "content_hash": row[3]
        }
    return False, {}


def search_by_topic(topic: str) -> list[dict]:
    """Search for lessons by topic (fuzzy search in name, description, focus_area)."""
    if not PRODUCTION_DB.exists():
        return []

    conn = sqlite3.connect(PRODUCTION_DB)
    cursor = conn.execute("""
        SELECT id, name, description, focus_area
        FROM lessons
        WHERE name LIKE ? OR description LIKE ? OR focus_area LIKE ?
        LIMIT 20
    """, (f"%{topic}%", f"%{topic}%", f"%{topic}%"))

    results = []
    for row in cursor.fetchall():
        results.append({
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "focus_area": row[3]
        })
    conn.close()
    return results


def get_statistics() -> dict:
    """Get production database statistics."""
    if not PRODUCTION_DB.exists():
        return {"total": 0, "by_level": {}}

    conn = sqlite3.connect(PRODUCTION_DB)

    total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

    # By CERF level
    levels = conn.execute("""
        SELECT
            SUBSTR(name, 1, 2) as level,
            COUNT(*) as count
        FROM lessons
        WHERE name LIKE 'a1_%' OR name LIKE 'a2_%'
            OR name LIKE 'b1_%' OR name LIKE 'b2_%'
            OR name LIKE 'c1_%' OR name LIKE 'c2_%'
        GROUP BY level
        ORDER BY level
    """).fetchall()

    conn.close()

    return {
        "total": total,
        "by_level": {level: count for level, count in levels}
    }


def main():
    parser = argparse.ArgumentParser(description="Check for duplicate lessons")
    parser.add_argument('--name', help='Check if lesson name exists')
    parser.add_argument('--content-hash', help='Check if content hash exists')
    parser.add_argument('--topic', help='Search for lessons by topic')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')

    args = parser.parse_args()

    if args.stats or (not args.name and not args.content_hash and not args.topic):
        stats = get_statistics()
        print(f"Production Database Statistics:")
        print(f"  Total lessons: {stats['total']}")
        print(f"  By CERF level:")
        for level, count in stats['by_level'].items():
            print(f"    {level.upper()}: {count}")
        return

    if args.name:
        exists, info = check_name_exists(args.name)
        if exists:
            print(f"❌ DUPLICATE: Lesson '{args.name}' already exists!")
            print(f"   ID: {info['id']}")
            print(f"   Description: {info['description']}")
            print(f"   Focus Area: {info['focus_area']}")
            sys.exit(1)
        else:
            print(f"✅ AVAILABLE: Lesson name '{args.name}' is available")
            sys.exit(0)

    if args.content_hash:
        exists, info = check_content_hash_exists(args.content_hash)
        if exists:
            print(f"❌ DUPLICATE CONTENT: A lesson with this content already exists!")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Content Hash: {info['content_hash'][:16]}...")
            sys.exit(1)
        else:
            print(f"✅ UNIQUE: Content hash not found in production")
            sys.exit(0)

    if args.topic:
        results = search_by_topic(args.topic)
        if results:
            print(f"Found {len(results)} lessons related to '{args.topic}':")
            for lesson in results:
                print(f"  - {lesson['name']}: {lesson['description']}")
        else:
            print(f"No lessons found for topic '{args.topic}'")


if __name__ == "__main__":
    main()
