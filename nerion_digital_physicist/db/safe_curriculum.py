"""
Safe curriculum database operations with automatic backups.
NEVER modifies the database without creating a backup first.
"""
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
BACKUP_DIR = Path("backups/curriculum/before_write")

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
        """Create timestamped backup before ANY write operation."""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.backup_path = BACKUP_DIR / f"curriculum_before_write_{timestamp}.sqlite"
        
        if self.db_path.exists():
            shutil.copy2(self.db_path, self.backup_path)
            logger.info(f"✅ Created backup: {self.backup_path}")
        else:
            logger.warning(f"Database {self.db_path} does not exist, no backup needed")
    
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

    def add_lesson(self, name: str, description: str, focus_area: str,
                   before_code: str, after_code: str, test_code: str) -> bool:
        """
        Safely add a lesson with automatic backup and duplicate prevention.

        Prevents duplicates by:
        1. Name uniqueness (UNIQUE constraint)
        2. Content uniqueness (hash-based deduplication)

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
                INSERT INTO lessons (name, description, focus_area, before_code, after_code, test_code, content_hash, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (name, description, focus_area, before_code, after_code, test_code, content_hash))

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
