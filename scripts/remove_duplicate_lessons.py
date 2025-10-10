#!/usr/bin/env python3
"""
Remove duplicate lessons (keeping the first one by timestamp).

Found 47 lessons with identical content. This script:
1. Identifies all duplicate groups
2. Keeps the OLDEST lesson in each group (by timestamp)
3. Removes the rest
4. Creates a backup before deletion
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("out/learning/curriculum.sqlite")

def remove_duplicates():
    """Remove duplicate lessons, keeping the oldest in each group."""
    print("🔄 Removing duplicate lessons")
    print("=" * 60)

    # Create backup first
    backup_path = Path(f"backups/curriculum/before_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite")
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(DB_PATH, backup_path)
    print(f"✅ Backup created: {backup_path}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find duplicate groups (content_hash appears more than once)
    cursor.execute("""
        SELECT content_hash, COUNT(*) as cnt
        FROM lessons
        WHERE content_hash IS NOT NULL
        GROUP BY content_hash
        HAVING cnt > 1
    """)

    duplicate_groups = cursor.fetchall()
    print(f"\n📊 Found {len(duplicate_groups)} duplicate content groups")

    total_to_remove = 0
    for content_hash, count in duplicate_groups:
        total_to_remove += count - 1  # Keep one, remove the rest

    print(f"   Will remove {total_to_remove} duplicate lessons")
    print(f"   Will keep {len(duplicate_groups)} lessons (oldest in each group)")

    # For each duplicate group, keep the oldest and remove the rest
    removed_lessons = []

    for content_hash, count in duplicate_groups:
        # Get all lessons with this hash, ordered by timestamp (oldest first)
        cursor.execute("""
            SELECT id, name, timestamp
            FROM lessons
            WHERE content_hash = ?
            ORDER BY timestamp ASC
        """, (content_hash,))

        lessons = cursor.fetchall()
        keep_lesson = lessons[0]  # Keep the oldest
        remove_lessons = lessons[1:]  # Remove the rest

        print(f"\n   Hash {content_hash[:16]}... ({len(lessons)} lessons):")
        print(f"     KEEP: {keep_lesson[1]} (created {keep_lesson[2]})")

        for lesson_id, name, timestamp in remove_lessons:
            print(f"     REMOVE: {name} (created {timestamp})")
            cursor.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
            removed_lessons.append((name, timestamp))

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM lessons")
    final_count = cursor.fetchone()[0]

    conn.close()

    print("\n" + "=" * 60)
    print("✅ Cleanup complete!")
    print(f"   Removed: {len(removed_lessons)} duplicate lessons")
    print(f"   Remaining: {final_count} unique lessons")
    print(f"   Backup: {backup_path}")
    print("=" * 60)

if __name__ == "__main__":
    remove_duplicates()
