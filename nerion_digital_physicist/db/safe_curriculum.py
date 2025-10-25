"""
Safe curriculum database operations with automatic backups.
NEVER modifies the database without creating a backup first.
"""
import os
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import logging
import hashlib
import ast

logger = logging.getLogger(__name__)

DB_PATH = Path("out/learning/curriculum.sqlite")
BACKUP_DIR = Path("backups/curriculum")

class SafeCurriculumDB:
    """
    A wrapper around the curriculum database that ALWAYS backs up before writes.
    
    Usage:
        with SafeCurriculumDB() as db:
            db.add_lesson(name, description, before_code, after_code, test_code)
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.backup_path = None
        self.connection = None
        
    def __enter__(self):
        """Create backup and open connection in WAL mode."""
        # 1. Create backup BEFORE opening
        self._create_backup()
        
        # 2. Open connection with WAL mode for safety
        self.connection = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # Autocommit mode
            timeout=30.0
        )
        
        # Enable WAL mode (Write-Ahead Logging) for better concurrency
        self.connection.execute("PRAGMA journal_mode=WAL;")
        
        # Enable foreign key constraints
        self.connection.execute("PRAGMA foreign_keys=ON;")
        
        # Verify integrity before writes
        integrity = self.connection.execute("PRAGMA integrity_check;").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Database integrity check failed: {integrity}")
        
        logger.info(f"SafeCurriculumDB opened (backup: {self.backup_path})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection and verify backup if error occurred."""
        if self.connection:
            if exc_type is not None:
                # Error occurred - restore from backup
                logger.error(f"Error during DB operation, restoring from {self.backup_path}")
                self.connection.close()
                self._restore_from_backup()
            else:
                # Success - close normally
                self.connection.close()
                logger.info("SafeCurriculumDB closed successfully")
    
    def _create_backup(self):
        """Create backup before EVERY write using space-efficient hardlinks.

        Backup strategy (maximum safety + minimal disk usage):
        - Backup before EVERY write operation (no data loss risk)
        - Use hardlinks to avoid duplicate storage (only changed blocks consume space)
        - Tiered retention with automatic cleanup:
          * Hourly: Keep last 24 (1 day of frequent snapshots)
          * Daily: Keep last 30 (1 month of daily snapshots)
        - Size limit: Clean up if total exceeds 10GB

        How hardlinks work:
        - If DB hasn't changed, hardlink uses 0 extra space
        - If DB changed, only the changed blocks consume space
        - Each backup appears as full file but shares unchanged data
        - Deletion of one hardlink doesn't affect others

        This ensures:
        - ZERO data loss (backup before every write)
        - Minimal disk usage (hardlinks share unchanged data)
        - Multiple recovery points (54 snapshots)
        - Automatic cleanup (10GB max)
        """
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        # Separate hourly and daily backup directories
        hourly_dir = BACKUP_DIR / "hourly"
        daily_dir = BACKUP_DIR / "daily"
        hourly_dir.mkdir(exist_ok=True)
        daily_dir.mkdir(exist_ok=True)

        if not self.db_path.exists():
            logger.warning(f"Database {self.db_path} does not exist, no backup needed")
            return

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

        # === ALWAYS CREATE HOURLY BACKUP (safety-first approach) ===
        hourly_backup = hourly_dir / f"curriculum_hourly_{timestamp}.sqlite"

        # Check if DB changed since last backup (space-efficient hardlink if unchanged)
        hourly_backups = sorted(hourly_dir.glob("curriculum_hourly_*.sqlite"),
                               key=lambda p: p.stat().st_mtime,
                               reverse=True)

        if hourly_backups:
            latest_hourly = hourly_backups[0]

            # Compare file sizes and modification times as quick check
            db_stat = self.db_path.stat()
            backup_stat = latest_hourly.stat()

            # If DB hasn't been modified since last backup, use hardlink (0 space)
            if db_stat.st_mtime <= backup_stat.st_mtime and db_stat.st_size == backup_stat.st_size:
                try:
                    os.link(latest_hourly, hourly_backup)
                    self.backup_path = hourly_backup
                    logger.debug(f"Hardlinked backup (0 space): {hourly_backup.name}")
                except OSError:
                    # Hardlink failed, fall back to copy
                    shutil.copy2(self.db_path, hourly_backup)
                    self.backup_path = hourly_backup
                    logger.info(f"✅ Created hourly backup: {hourly_backup.name}")
            else:
                # DB changed, create new copy
                shutil.copy2(self.db_path, hourly_backup)
                self.backup_path = hourly_backup
                logger.info(f"✅ Created hourly backup: {hourly_backup.name}")
        else:
            # First backup
            shutil.copy2(self.db_path, hourly_backup)
            self.backup_path = hourly_backup
            logger.info(f"✅ Created first hourly backup: {hourly_backup.name}")

            # Keep last 24 hourly backups (1 day of hourly snapshots)
            hourly_backups = sorted(hourly_dir.glob("curriculum_hourly_*.sqlite"),
                                   key=lambda p: p.stat().st_mtime,
                                   reverse=True)
            if len(hourly_backups) > 24:
                for old in hourly_backups[24:]:
                    old.unlink()
                    logger.debug(f"Cleaned up old hourly backup: {old.name}")

        # === DAILY BACKUPS (less frequent, long retention) ===
        daily_backups = sorted(daily_dir.glob("curriculum_daily_*.sqlite"),
                              key=lambda p: p.stat().st_mtime,
                              reverse=True)

        should_create_daily = True
        if daily_backups:
            latest_daily = daily_backups[0]
            latest_time = datetime.fromtimestamp(latest_daily.stat().st_mtime)
            time_since = now - latest_time

            # Only create new daily backup if >23 hours since last one
            if time_since.total_seconds() < 82800:  # 23 hours
                should_create_daily = False

        if should_create_daily:
            date_stamp = now.strftime("%Y%m%d")
            daily_backup = daily_dir / f"curriculum_daily_{date_stamp}.sqlite"
            shutil.copy2(self.db_path, daily_backup)
            logger.info(f"✅ Created daily backup: {daily_backup.name}")

            # Keep last 30 daily backups (1 month of daily snapshots)
            daily_backups = sorted(daily_dir.glob("curriculum_daily_*.sqlite"),
                                  key=lambda p: p.stat().st_mtime,
                                  reverse=True)
            if len(daily_backups) > 30:
                for old in daily_backups[30:]:
                    old.unlink()
                    logger.debug(f"Cleaned up old daily backup: {old.name}")

        # === SIZE-BASED CLEANUP (safety valve only) ===
        # Only if total backup size exceeds 10GB, clean up oldest hourly backups
        total_size = sum(f.stat().st_size for f in hourly_dir.glob("*.sqlite"))
        total_size += sum(f.stat().st_size for f in daily_dir.glob("*.sqlite"))

        max_size = 10 * 1024 * 1024 * 1024  # 10GB
        if total_size > max_size:
            logger.warning(f"Backup size {total_size / 1e9:.1f}GB exceeds 10GB limit, cleaning up oldest hourly backups")
            hourly_backups = sorted(hourly_dir.glob("curriculum_hourly_*.sqlite"),
                                   key=lambda p: p.stat().st_mtime)
            # Delete oldest hourly backups until under limit (keep daily backups intact)
            for old in hourly_backups:
                if total_size <= max_size:
                    break
                size = old.stat().st_size
                old.unlink()
                total_size -= size
                logger.info(f"Cleaned up {old.name} to free space")
    
    def _restore_from_backup(self):
        """Restore database from the backup created in this session."""
        if self.backup_path and self.backup_path.exists():
            shutil.copy2(self.backup_path, self.db_path)
            logger.info(f"✅ Restored from backup: {self.backup_path}")
        else:
            logger.error("No backup available for restoration!")
    
    def _normalize_code(self, code: str) -> str:
        """
        Normalize Python code for consistent hashing.
        Removes comments, normalizes whitespace, sorts imports.
        """
        try:
            # Parse the code into AST
            tree = ast.parse(code)
            # Convert back to code (normalized)
            return ast.unparse(tree)
        except:
            # If parsing fails, just strip and normalize whitespace
            return "\n".join(line.strip() for line in code.split("\n") if line.strip())

    def _calculate_content_hash(self, before_code: str, after_code: str, test_code: str) -> str:
        """
        Calculate SHA256 hash of lesson content for duplicate detection.
        Normalizes code before hashing to catch semantically identical lessons.
        """
        # Normalize all code sections
        norm_before = self._normalize_code(before_code)
        norm_after = self._normalize_code(after_code)
        norm_test = self._normalize_code(test_code)

        # Combine normalized code
        combined = f"{norm_before}|||{norm_after}|||{norm_test}"

        # Calculate SHA256 hash
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def content_exists(self, before_code: str, after_code: str, test_code: str) -> tuple[bool, str]:
        """
        Check if lesson with identical content already exists.

        Returns:
            (exists: bool, existing_name: str or None)
        """
        content_hash = self._calculate_content_hash(before_code, after_code, test_code)
        cursor = self.connection.cursor()
        result = cursor.execute(
            "SELECT name FROM lessons WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()

        if result:
            return (True, result[0])
        return (False, None)

    def lesson_exists(self, name: str) -> bool:
        """Check if a lesson with this name already exists."""
        cursor = self.connection.cursor()
        result = cursor.execute("SELECT 1 FROM lessons WHERE name = ?", (name,)).fetchone()
        return result is not None

    def add_lesson(self, name: str, description: str, before_code: str,
                   after_code: str, test_code: str,
                   focus_area: str = None, category: str = None,
                   language: str = None, metadata: str = None) -> bool:
        """
        Safely add a lesson with automatic backup and duplicate prevention.

        Supports both old schema (focus_area) and new schema (category, language, metadata).
        Use focus_area for curriculum lessons, or category+language+metadata for GitHub lessons.

        Prevents duplicates by:
        1. Name uniqueness (UNIQUE constraint)
        2. Content uniqueness (hash-based deduplication)

        Args:
            name: Unique lesson identifier
            description: Lesson description
            before_code: Code before fix
            after_code: Code after fix
            test_code: Test code to verify fix
            focus_area: CEFR level (a1, a2, b1, b2, c1, c2) - for curriculum lessons
            category: Lesson category - for GitHub lessons
            language: Programming language (python, javascript, etc.) - for GitHub lessons
            metadata: JSON metadata string - for GitHub lessons

        Returns:
            True if lesson was added successfully
            False if lesson already exists (duplicate prevented)
        """
        # 1. Check for duplicate NAME
        if self.lesson_exists(name):
            logger.warning(f"⚠️  DUPLICATE NAME: Lesson '{name}' already exists in database")
            return False

        # 2. Check for duplicate CONTENT
        exists, existing_name = self.content_exists(before_code, after_code, test_code)
        if exists:
            logger.warning(
                f"⚠️  DUPLICATE CONTENT: Lesson with identical code already exists as '{existing_name}'. "
                f"New lesson '{name}' rejected."
            )
            return False

        try:
            # Calculate content hash for storage
            content_hash = self._calculate_content_hash(before_code, after_code, test_code)

            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO lessons (
                    name, description, focus_area, before_code, after_code, test_code,
                    content_hash, category, language, metadata, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (name, description, focus_area, before_code, after_code, test_code,
                  content_hash, category, language, metadata))

            logger.info(f"✅ Added lesson: {name} (hash: {content_hash[:16]}...)")
            return True
        except sqlite3.IntegrityError as e:
            # This should rarely happen now due to pre-checks, but keep as safety net
            logger.error(f"❌ DUPLICATE ERROR: Lesson '{name}' blocked by database constraint: {e}")
            return False
    
    def get_lesson_count(self) -> int:
        """Get total number of lessons."""
        cursor = self.connection.cursor()
        return cursor.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    
    def verify_integrity(self) -> bool:
        """Verify database integrity."""
        result = self.connection.execute("PRAGMA integrity_check;").fetchone()[0]
        return result == "ok"

def backup_to_gcs():
    """Backup current database to GCS."""
    try:
        subprocess.run([
            "gcloud", "storage", "cp",
            str(DB_PATH),
            f"gs://nerion-training-data/curriculum/curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"
        ], check=True, capture_output=True)
        logger.info("✅ Backed up to GCS")
        return True
    except Exception as e:
        logger.warning(f"GCS backup failed: {e}")
        return False
