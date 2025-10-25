# Curriculum Database Protection System

## Overview

This document describes the multi-layered protection system that ensures the curriculum database is **NEVER lost**.

## Protection Layers

### 1. Write-Ahead Logging (WAL Mode)

**Status**: ✅ Enabled

SQLite WAL mode provides:
- Better concurrency (readers don't block writers)
- Atomic commits
- Crash recovery
- Checkpointing for data integrity

**Verification**:
```bash
sqlite3 out/learning/curriculum.sqlite "PRAGMA journal_mode;"
# Should output: wal
```

### 2. Safe Database Wrapper with Duplicate Prevention

**Location**: `nerion_digital_physicist/db/safe_curriculum.py`

**Features**:
- **Automatic backup before EVERY write**
- **Automatic rollback on errors**
- **Integrity checking before writes**
- **Duplicate prevention (2-layer)**:
  - Name uniqueness (UNIQUE constraint on `name` field)
  - Content uniqueness (SHA256 hash of normalized code)
- **Context manager for safe operations**

**How Content Deduplication Works**:
1. Before inserting, SafeCurriculumDB normalizes the Python code (removes comments, normalizes whitespace)
2. Calculates SHA256 hash of `before_code + after_code + test_code`
3. Checks if this exact code already exists in database
4. Rejects duplicate content even with different names

**Usage**:
```python
from nerion_digital_physicist.db.safe_curriculum import SafeCurriculumDB

with SafeCurriculumDB() as db:
    # Check if lesson exists
    if db.lesson_exists("a1_example"):
        print("Lesson already exists, skipping")
    else:
        # Add lesson (returns False if duplicate)
        success = db.add_lesson(
            name="a1_example",  # MUST be unique!
            description="...",
            focus_area="a1",
            before_code="...",
            after_code="...",
            test_code="..."
        )
        if not success:
            print("Duplicate prevented")
    # If ANY error occurs, automatically restores from backup
```

### 3. Automated Backups

#### Hourly Backups (Local)

**Script**: `scripts/backup_curriculum.sh`
**Schedule**: Every hour via cron
**Location**: `backups/curriculum/daily/`
**Retention**: 30 days

**Manual backup**:
```bash
./scripts/backup_curriculum.sh
```

#### Before-Write Backups

**Automatic**: Created by `SafeCurriculumDB` before every write
**Location**: `backups/curriculum/before_write/`
**Purpose**: Immediate rollback if write fails

#### GCS Cloud Backups

**Schedule**: 
- Every hour (with local backups)
- Every 30 minutes (monitoring sync)
- Daily full backup at 3 AM

**Location**: `gs://nerion-training-data/curriculum/`

**Types**:
- `curriculum_<timestamp>.sqlite` - Manual backups
- `auto_sync_<timestamp>.sqlite` - Automated syncs (keep 50)
- `daily_backup_<date>.sqlite` - Daily snapshots

### 4. Database Monitoring

**Script**: `scripts/monitor_curriculum.sh`
**Schedule**: Every 30 minutes
**Log**: `out/learning/monitor.log`

**Checks**:
- ✅ Database file exists
- ✅ Integrity check passes
- ✅ Lesson count > 500 threshold
- ✅ WAL mode enabled
- ☁️  Auto-sync to GCS

**Alerts**: Logged to `out/learning/curriculum_alerts.log`

### 5. Easy Restoration

**Script**: `scripts/restore_curriculum.sh`

**Restore options**:
```bash
# Restore latest local backup
./scripts/restore_curriculum.sh
# Enter: latest

# Restore latest GCS backup
./scripts/restore_curriculum.sh
# Enter: gcs:latest

# Restore specific backup
./scripts/restore_curriculum.sh
# Enter: /path/to/backup.sqlite
```

## Setup Instructions

### Initial Setup

```bash
# 1. Ensure WAL mode is enabled (already done)
sqlite3 out/learning/curriculum.sqlite "PRAGMA journal_mode=WAL;"

# 2. Run initial backup
./scripts/backup_curriculum.sh

# 3. Set up automated backups (optional)
./scripts/setup_backup_cron.sh
```

### Verification

```bash
# Check backup system
ls -lh backups/curriculum/daily/

# Check GCS backups
gcloud storage ls gs://nerion-training-data/curriculum/

# Check monitoring
./scripts/monitor_curriculum.sh

# Verify lesson count
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"
```

## Recovery Procedures

### Scenario 1: Accidental Data Loss (Within 1 hour)

```bash
# Use before-write backup
LATEST=$(ls -t backups/curriculum/before_write/*.sqlite | head -1)
cp "$LATEST" out/learning/curriculum.sqlite
```

### Scenario 2: Data Loss Discovered Later

```bash
# Interactive restoration
./scripts/restore_curriculum.sh
```

### Scenario 3: Complete System Failure

```bash
# Restore from GCS
gcloud storage ls gs://nerion-training-data/curriculum/*.sqlite | sort -r | head -1
gcloud storage cp <GCS_PATH> out/learning/curriculum.sqlite
```

### Scenario 4: Database Corruption

```bash
# 1. Check integrity
sqlite3 out/learning/curriculum.sqlite "PRAGMA integrity_check;"

# 2. If corrupted, restore from backup
./scripts/restore_curriculum.sh

# 3. Re-enable WAL mode
sqlite3 out/learning/curriculum.sqlite "PRAGMA journal_mode=WAL;"
```

## Protection Features Summary

| Feature | Status | Frequency | Retention |
|---------|--------|-----------|-----------|
| WAL Mode | ✅ Enabled | Always | N/A |
| Before-Write Backup | ✅ Enabled | Every write | Until next write |
| Local Hourly Backup | ✅ Ready | Every hour | 30 days |
| GCS Sync | ✅ Ready | Every 30 min | 50 recent |
| Daily GCS Backup | ✅ Ready | 3 AM daily | Indefinite |
| Integrity Monitoring | ✅ Ready | Every 30 min | N/A |
| Safe Write Wrapper | ✅ Implemented | Every write | N/A |

## Best Practices

### For Developers

1. **ALWAYS use SafeCurriculumDB** for writes:
   ```python
   with SafeCurriculumDB() as db:
       result = db.add_lesson(name="unique_lesson_name", ...)
       if not result:
           print("Lesson already exists (duplicate prevented)")
   ```

2. **NEVER directly modify the database** with:
   - `sqlite3.connect()` without backup
   - SQL DROP/TRUNCATE commands
   - Database recreation

3. **Ensure lesson names are UNIQUE**:
   - The database has a UNIQUE constraint on `name` field
   - Duplicates are automatically prevented
   - Use descriptive, unique names (e.g., `a1_variable_scope_001`, `b2_async_context_manager_race_condition`)

4. **Check backup logs regularly**:
   ```bash
   tail -f out/learning/backup.log
   tail -f out/learning/monitor.log
   ```

### For Operations

1. **Verify backups weekly**:
   ```bash
   ./scripts/monitor_curriculum.sh
   gcloud storage ls gs://nerion-training-data/curriculum/
   ```

2. **Test restoration quarterly**:
   ```bash
   # In a test environment
   ./scripts/restore_curriculum.sh
   ```

3. **Monitor disk space**:
   ```bash
   du -sh backups/curriculum/
   ```

## Monitoring Alerts

The system alerts on:

- ⚠️ Lesson count drops below 500
- ❌ Database file missing
- ❌ Integrity check fails
- ⚠️ WAL mode disabled
- ⚠️ GCS sync fails

**Alert log**: `out/learning/curriculum_alerts.log`

## Duplicate Prevention System

### Two Layers of Protection

1. **Name Uniqueness** (Database Constraint)
   - `name` field has UNIQUE constraint
   - Prevents lessons with identical names

2. **Content Uniqueness** (Hash-Based)
   - SHA256 hash of normalized code (`before_code + after_code + test_code`)
   - Prevents lessons with identical code but different names
   - Code normalization: removes comments, normalizes whitespace, parses AST

### Migration History

**October 9, 2025**: Content deduplication implemented
- Added `content_hash` column to lessons table
- Calculated hashes for all 523 existing lessons
- **Found and removed 46 duplicate lessons**
- Database cleaned: 523 → 477 unique lessons
- One duplicate group contained 47 lessons with identical code

### Scripts

- `scripts/migrate_add_content_hash.py` - Add content_hash column and populate
- `scripts/remove_duplicate_lessons.py` - Remove duplicate content (keeps oldest)

## Current Status

```bash
Database: out/learning/curriculum.sqlite
Lessons: 477 (all unique)
Duplicates Removed: 46
Last Backup: 2025-10-09 23:42:31
GCS Status: ✅ Synced
WAL Mode: ✅ Enabled
Content Deduplication: ✅ Active
Protection: 🔒 MAXIMUM
```

## Emergency Contacts

If you encounter issues:

1. Check `out/learning/curriculum_alerts.log`
2. Run `./scripts/monitor_curriculum.sh`
3. Restore from backup: `./scripts/restore_curriculum.sh`
4. Verify GCS backups: `gcloud storage ls gs://nerion-training-data/curriculum/`

**Your data is protected by 7 layers of redundancy!**
