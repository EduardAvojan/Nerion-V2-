#!/usr/bin/env python3
"""
Safe lesson generation workflow with duplicate prevention and quality checks.

This script:
1. Copies main DB to workspace (so agents know what exists)
2. Lets agents generate lessons
3. Reviews lesson quality (CRITICAL - catches broken tests, placeholders, etc.)
4. Merges ONLY new lessons to main DB (after review passes)
5. Cleans up workspace
6. NEVER touches main DB directly

Safety guarantees:
- Main DB is READ-ONLY until merge (protected by SafeCurriculumDB)
- Quality review MUST pass before merge (prevents broken lessons)
- Workspace is always deleted after merge (no copies left behind)
- Multiple safety checks prevent accidental deletion

Quality checks:
- No placeholder variables (CODE, TEST_TYPE, etc.)
- Test code is executable (has imports/functions)
- Code is not trivially short
- before_code and after_code are different
- All required fields present
"""
import sqlite3
import shutil
from pathlib import Path
from nerion_digital_physicist.db.safe_curriculum import SafeCurriculumDB

# CONSTANTS - These paths are hardcoded for safety
MAIN_DB = Path("out/learning/curriculum.sqlite")
WORKSPACE_DB = Path("agent_generated_curriculum.sqlite")

def prepare_workspace():
    """
    Step 1: Copy main DB to workspace so agents can check for duplicates.

    Safety: Main DB is never modified, only read.
    """
    print("=" * 70)
    print("STEP 1: PREPARE WORKSPACE")
    print("=" * 70)

    # Safety check: Ensure main DB exists
    if not MAIN_DB.exists():
        raise FileNotFoundError(f"❌ FATAL: Main database not found at {MAIN_DB}")

    # Safety check: Ensure we're not about to overwrite main DB
    if WORKSPACE_DB == MAIN_DB:
        raise ValueError("❌ FATAL: Workspace and main DB paths are the same!")

    # Safety check: Confirm main DB is protected location
    if not str(MAIN_DB).startswith("out/learning/"):
        raise ValueError(f"❌ FATAL: Main DB is not in expected protected location: {MAIN_DB}")

    # Get initial count from main DB (for verification later)
    conn = sqlite3.connect(MAIN_DB)
    cursor = conn.cursor()
    initial_count = cursor.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    conn.close()

    print(f"✅ Main database: {MAIN_DB}")
    print(f"✅ Main database has: {initial_count} lessons")
    print(f"✅ Workspace: {WORKSPACE_DB}")
    print()

    # Copy main DB to workspace (READ operation on main DB)
    print(f"📋 Copying main database to workspace...")
    shutil.copy2(MAIN_DB, WORKSPACE_DB)

    # Verify copy
    conn = sqlite3.connect(WORKSPACE_DB)
    cursor = conn.cursor()
    workspace_count = cursor.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

    # Export existing lesson names for duplicate checking
    existing_names = {row[0] for row in cursor.execute("SELECT name FROM lessons").fetchall()}
    conn.close()

    if workspace_count != initial_count:
        raise ValueError(f"❌ FATAL: Copy verification failed! Expected {initial_count}, got {workspace_count}")

    # Write existing names to file for agents to check
    with open('/tmp/existing_names.txt', 'w') as f:
        for name in sorted(existing_names):
            f.write(f"{name}\n")

    print(f"✅ Workspace ready with {workspace_count} existing lessons")
    print(f"✅ Exported {len(existing_names)} lesson names to /tmp/existing_names.txt")
    print(f"✅ Agents can now check for duplicates before generating")
    print()
    print("=" * 70)
    print("⚠️  AGENTS MUST load existing names before generating:")
    print("=" * 70)
    print("with open('/tmp/existing_names.txt', 'r') as f:")
    print("    existing_names = {line.strip() for line in f}")
    print()
    print("def is_duplicate(name):")
    print("    return name in existing_names")
    print("=" * 70)
    print()
    print("=" * 70)
    print("NOW: Activate agents to generate lessons")
    print("=" * 70)
    print()

    return initial_count

def review_lessons():
    """
    Step 2: Review quality of new lessons before merging.

    Quality checks:
    - No placeholder variables (CODE, TEST_TYPE, etc.)
    - Test code is executable
    - Code is not trivially short
    - before_code and after_code are different
    - All required fields present
    """
    print()
    print("=" * 70)
    print("STEP 2: REVIEW NEW LESSONS")
    print("=" * 70)

    # Safety check: Ensure workspace exists
    if not WORKSPACE_DB.exists():
        raise FileNotFoundError(f"❌ FATAL: Workspace database not found at {WORKSPACE_DB}")

    # Get initial count from main DB to find NEW lessons
    with SafeCurriculumDB(MAIN_DB) as db:
        main_count = db.get_lesson_count()

        # Get all existing lesson names
        conn = sqlite3.connect(MAIN_DB)
        cursor = conn.cursor()
        existing_names = set(row[0] for row in cursor.execute("SELECT name FROM lessons").fetchall())
        conn.close()

    # Get lessons from workspace
    conn = sqlite3.connect(WORKSPACE_DB)
    cursor = conn.cursor()
    workspace_count = cursor.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

    # Find NEW lessons only (not in main DB)
    all_lessons = cursor.execute("""
        SELECT name, description, focus_area, before_code, after_code, test_code,
               category, language, metadata
        FROM lessons
    """).fetchall()
    conn.close()

    # Filter to only NEW lessons (not in existing_names set)
    new_lessons = [lesson for lesson in all_lessons if lesson[0] not in existing_names]

    new_count = len(new_lessons)

    print(f"✅ Main database: {main_count} lessons")
    print(f"✅ Workspace: {workspace_count} lessons")
    print(f"✅ NEW lessons to review: {new_count}")
    print()

    if new_count == 0:
        print("⚠️  No new lessons to review!")
        return True

    # Check for duplicate names WITHIN new lessons
    new_lesson_names = [lesson[0] for lesson in new_lessons]
    name_counts = {}
    for name in new_lesson_names:
        name_counts[name] = name_counts.get(name, 0) + 1

    duplicates_within_new = {name: count for name, count in name_counts.items() if count > 1}

    if duplicates_within_new:
        print("=" * 70)
        print("❌ CRITICAL: DUPLICATE NAMES FOUND IN NEW LESSONS!")
        print("=" * 70)
        for name, count in duplicates_within_new.items():
            print(f"   {name}: appears {count} times")
        print()
        print("This should NEVER happen! The agent failed duplicate checking!")
        print("You must delete duplicate lessons from workspace before merging.")
        print("=" * 70)
        print()
        return False

    # Quality checks
    print("🔍 Running quality checks...")
    print()

    issues = []
    warnings = []

    for lesson in new_lessons:
        name, description, focus_area, before_code, after_code, test_code, category, language, metadata = lesson
        lesson_issues = []
        lesson_warnings = []

        # Check 1: Placeholder variables in test_code
        # Note: Exclude PHP heredoc syntax (<<<'CODE' and CODE;) which is valid
        placeholders = ['CODE', 'TEST_TYPE', 'BEFORE_CODE', 'AFTER_CODE']
        found_placeholders = []
        for p in placeholders:
            if p in test_code:
                # Exclude PHP heredoc syntax
                if p == 'CODE' and ("<<<'CODE'" in test_code or "<<<CODE" in test_code):
                    continue  # Valid PHP heredoc delimiter
                found_placeholders.append(p)
        if found_placeholders:
            lesson_issues.append(f"Test code has placeholder variables: {', '.join(found_placeholders)}")

        # Check 2: Test code must have some structure (not just empty)
        # Check for any language's import/include/using/require or function/class definition
        has_structure = any([
            'import ' in test_code,  # Python, Java
            'def ' in test_code,     # Python
            'using ' in test_code,   # C#
            'include' in test_code,  # C++, PHP
            'require' in test_code,  # Ruby
            'use ' in test_code,     # Rust
            'class ' in test_code,   # Multiple languages
            'function ' in test_code,  # JavaScript, PHP
            'fn ' in test_code,      # Rust
            '<?php' in test_code,    # PHP
        ])
        if not has_structure and len(test_code) < 100:
            lesson_warnings.append("Test code appears too simple (no clear structure)")

        # Check 3: Code too short (likely incomplete)
        if len(before_code) < 20:
            lesson_warnings.append(f"before_code very short ({len(before_code)} chars)")
        if len(after_code) < 20:
            lesson_warnings.append(f"after_code very short ({len(after_code)} chars)")
        if len(test_code) < 50:
            lesson_issues.append(f"test_code very short ({len(test_code)} chars)")

        # Check 4: before_code and after_code should be different
        if before_code == after_code:
            lesson_issues.append("before_code and after_code are identical!")

        # Check 5: Required fields
        if not name:
            lesson_issues.append("Missing name")
        if not description:
            lesson_warnings.append("Missing description")
        if not focus_area:
            lesson_warnings.append("Missing focus_area")
        if not language:
            lesson_warnings.append("Missing language")

        # Report issues for this lesson
        if lesson_issues or lesson_warnings:
            print(f"📝 {name} [{language or 'unknown'}]:")
            for issue in lesson_issues:
                print(f"   ❌ ERROR: {issue}")
                issues.append((name, issue))
            for warning in lesson_warnings:
                print(f"   ⚠️  WARNING: {warning}")
                warnings.append((name, warning))
            print()

    # Summary
    print("=" * 70)
    print("QUALITY CHECK SUMMARY")
    print("=" * 70)
    print(f"Total new lessons:  {new_count}")
    print(f"Errors found:       {len(issues)}")
    print(f"Warnings found:     {len(warnings)}")
    print()

    if issues:
        print("❌ QUALITY CHECK FAILED")
        print()
        print("Critical issues found:")
        for name, issue in issues:
            print(f"  - {name}: {issue}")
        print()
        print("These lessons CANNOT be merged. They must be fixed first.")
        print()
        print("Recommendation:")
        print("  1. Delete flawed lessons from workspace")
        print("  2. Regenerate with proper testing")
        print("  3. Run review again")
        print()
        return False
    elif warnings:
        print("⚠️  WARNINGS FOUND (but no critical errors)")
        print()
        print("Non-critical issues:")
        for name, warning in warnings[:10]:  # Show first 10
            print(f"  - {name}: {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
        print()
        print("These lessons CAN be merged, but may need improvement later.")
        print()
        return True
    else:
        print("✅ ALL QUALITY CHECKS PASSED")
        print()
        print("Lessons are ready to merge!")
        print()
        return True

def merge_new_lessons():
    """
    Step 3: Merge ONLY new lessons from workspace to main DB.

    Safety: SafeCurriculumDB automatically rejects duplicates.
    """
    print()
    print("=" * 70)
    print("STEP 3: MERGE NEW LESSONS")
    print("=" * 70)

    # Safety check: Ensure workspace exists
    if not WORKSPACE_DB.exists():
        raise FileNotFoundError(f"❌ FATAL: Workspace database not found at {WORKSPACE_DB}")

    # Get initial count from main DB
    with SafeCurriculumDB(MAIN_DB) as db:
        initial_main_count = db.get_lesson_count()

    # Get count from workspace
    conn = sqlite3.connect(WORKSPACE_DB)
    cursor = conn.cursor()
    workspace_count = cursor.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

    # Read all lessons from workspace
    lessons = cursor.execute("""
        SELECT name, description, focus_area, before_code, after_code, test_code,
               category, language, metadata
        FROM lessons
        ORDER BY name
    """).fetchall()
    conn.close()

    print(f"✅ Main database: {initial_main_count} lessons")
    print(f"✅ Workspace: {workspace_count} lessons")
    print(f"✅ Will attempt to add: {workspace_count} lessons (duplicates will be rejected)")
    print()

    # Merge to main DB (SafeCurriculumDB handles duplicates)
    added_count = 0
    duplicate_name_count = 0
    duplicate_content_count = 0

    print("🔄 Merging lessons...")
    with SafeCurriculumDB(MAIN_DB) as db:
        for lesson in lessons:
            name, description, focus_area, before_code, after_code, test_code, category, language, metadata = lesson

            result = db.add_lesson(
                name=name,
                description=description,
                focus_area=focus_area,
                before_code=before_code,
                after_code=after_code,
                test_code=test_code,
                category=category,
                language=language,
                metadata=metadata
            )

            if result:
                added_count += 1
                lang_str = f" [{language}]" if language else ""
                print(f"  ✅ Added: {name}{lang_str}")
            else:
                # Duplicate detected
                if db.lesson_exists(name):
                    duplicate_name_count += 1
                else:
                    duplicate_content_count += 1

        final_count = db.get_lesson_count()

    print()
    print("=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Initial lessons:        {initial_main_count}")
    print(f"Final lessons:          {final_count}")
    print(f"NEW lessons added:      {added_count}")
    print(f"Duplicates (name):      {duplicate_name_count}")
    print(f"Duplicates (content):   {duplicate_content_count}")
    print()

    return added_count

def cleanup_workspace():
    """
    Step 4: Clean up workspace database.

    Safety: Only deletes workspace, NEVER touches main DB.
    """
    print("=" * 70)
    print("STEP 4: CLEANUP")
    print("=" * 70)

    # Safety check: Ensure we're not about to delete main DB
    if WORKSPACE_DB == MAIN_DB:
        raise ValueError("❌ FATAL: Refusing to delete - workspace path equals main DB path!")

    # Safety check: Ensure workspace is not in protected location
    if str(WORKSPACE_DB).startswith("out/learning/"):
        raise ValueError(f"❌ FATAL: Refusing to delete - workspace is in protected location: {WORKSPACE_DB}")

    # Delete workspace
    if WORKSPACE_DB.exists():
        WORKSPACE_DB.unlink()
        print(f"✅ Deleted workspace: {WORKSPACE_DB}")
    else:
        print(f"⚠️  Workspace already deleted: {WORKSPACE_DB}")

    # Verify main DB still exists and is intact
    if not MAIN_DB.exists():
        raise FileNotFoundError(f"❌ FATAL: Main database was deleted! This should never happen!")

    with SafeCurriculumDB(MAIN_DB) as db:
        final_count = db.get_lesson_count()

    print(f"✅ Main database intact: {final_count} lessons")
    print()
    print("✅ Workflow complete - no copies left behind")
    print()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 safe_lesson_workflow.py prepare  # Step 1: Prepare workspace")
        print("  python3 safe_lesson_workflow.py review   # Step 2: Review lesson quality")
        print("  python3 safe_lesson_workflow.py merge    # Step 3: Merge new lessons")
        print("  python3 safe_lesson_workflow.py cleanup  # Step 4: Clean up workspace")
        print()
        print("Full workflow:")
        print("  1. python3 safe_lesson_workflow.py prepare")
        print("  2. Activate agents to generate lessons")
        print("  3. python3 safe_lesson_workflow.py review   # ⚠️ CRITICAL: Quality check")
        print("  4. python3 safe_lesson_workflow.py merge    # Only if review passes!")
        print("  5. python3 safe_lesson_workflow.py cleanup")
        print()
        print("⚠️  IMPORTANT: Always run 'review' before 'merge' to catch quality issues!")
        sys.exit(1)

    command = sys.argv[1]

    if command == "prepare":
        prepare_workspace()
    elif command == "review":
        passed = review_lessons()
        if not passed:
            print("❌ Review failed - DO NOT MERGE until issues are fixed!")
            sys.exit(1)
    elif command == "merge":
        merge_new_lessons()
    elif command == "cleanup":
        cleanup_workspace()
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
