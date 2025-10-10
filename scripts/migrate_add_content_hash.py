#!/usr/bin/env python3
"""
Database migration: Add content_hash column and populate it for existing lessons.

This migration:
1. Adds content_hash column to lessons table
2. Calculates and stores content hash for all existing lessons
3. Identifies and reports any duplicate content
"""
import sqlite3
import hashlib
import ast
from pathlib import Path

DB_PATH = Path("out/learning/curriculum.sqlite")

def normalize_code(code: str) -> str:
    """Normalize Python code for consistent hashing."""
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except:
        return "\n".join(line.strip() for line in code.split("\n") if line.strip())

def calculate_content_hash(before_code: str, after_code: str, test_code: str) -> str:
    """Calculate SHA256 hash of lesson content."""
    norm_before = normalize_code(before_code)
    norm_after = normalize_code(after_code)
    norm_test = normalize_code(test_code)
    combined = f"{norm_before}|||{norm_after}|||{norm_test}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()

def migrate():
    """Run the migration."""
    print("🔄 Starting migration: Add content_hash column")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Step 1: Check if column already exists
    cursor.execute("PRAGMA table_info(lessons)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'content_hash' in columns:
        print("✅ content_hash column already exists")
    else:
        print("📝 Adding content_hash column...")
        cursor.execute("ALTER TABLE lessons ADD COLUMN content_hash TEXT")
        conn.commit()
        print("✅ Column added")

    # Step 2: Calculate and update content_hash for all lessons
    print("\n📊 Calculating content hashes for existing lessons...")
    cursor.execute("SELECT id, name, before_code, after_code, test_code FROM lessons")
    lessons = cursor.fetchall()

    print(f"   Found {len(lessons)} lessons to process")

    hash_map = {}  # Track duplicates: hash -> [lesson_names]
    updated = 0

    for lesson_id, name, before_code, after_code, test_code in lessons:
        content_hash = calculate_content_hash(before_code, after_code, test_code)

        # Track duplicates
        if content_hash not in hash_map:
            hash_map[content_hash] = []
        hash_map[content_hash].append(name)

        # Update the hash
        cursor.execute(
            "UPDATE lessons SET content_hash = ? WHERE id = ?",
            (content_hash, lesson_id)
        )
        updated += 1

        if updated % 100 == 0:
            print(f"   Processed {updated}/{len(lessons)} lessons...")

    conn.commit()
    print(f"✅ Updated {updated} lessons with content hashes")

    # Step 3: Report duplicates
    duplicates = {h: names for h, names in hash_map.items() if len(names) > 1}

    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} duplicate content groups:")
        for hash_val, names in duplicates.items():
            print(f"\n   Hash: {hash_val[:16]}...")
            print(f"   Duplicate lessons ({len(names)}):")
            for name in names:
                print(f"     - {name}")
    else:
        print("\n✅ No duplicate content found!")

    # Step 4: Create index on content_hash for fast lookups
    print("\n📇 Creating index on content_hash...")
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lessons_content_hash ON lessons(content_hash)")
        conn.commit()
        print("✅ Index created")
    except Exception as e:
        print(f"⚠️  Index creation skipped: {e}")

    # Step 5: Verify
    cursor.execute("SELECT COUNT(*) FROM lessons WHERE content_hash IS NOT NULL")
    count_with_hash = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM lessons")
    total_count = cursor.fetchone()[0]

    conn.close()

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print(f"   Total lessons: {total_count}")
    print(f"   With content_hash: {count_with_hash}")
    print(f"   Duplicate groups: {len(duplicates)}")
    print("=" * 60)

if __name__ == "__main__":
    migrate()
