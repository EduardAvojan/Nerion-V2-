"""
This module provides a simple SQLite-backed store for the curriculum.

The CurriculumStore class handles all database operations for storing and retrieving
lessons, bug fixes, feature implementations, performance optimizations, and code
explanations. It provides context manager support for proper resource cleanup
and batch loading capabilities for memory-efficient operations.
"""
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

class CurriculumStore:
    """
    Handles all database operations for the curriculum.
    
    This class provides a SQLite-backed store for managing different types of
    lessons including regular lessons, bug fixes, feature implementations,
    performance optimizations, and code explanations.
    
    Attributes:
        _db_path (Path): Path to the SQLite database file
        _conn (sqlite3.Connection): Database connection
    
    Example:
        >>> with CurriculumStore(Path("curriculum.db")) as store:
        ...     lesson_data = {"name": "test", "description": "test lesson", ...}
        ...     store.add_lesson(lesson_data)
        ...     count = store.get_lesson_count()
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the CurriculumStore.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._bootstrap()

    def _bootstrap(self) -> None:
        """
        Create the necessary tables if they don't exist.
        
        This method sets up the database schema for all lesson types:
        - lessons: Regular refactoring lessons
        - bug_fixes: Bug fixing lessons
        - feature_implementations: Feature implementation lessons
        - performance_optimizations: Performance optimization lessons
        - code_explanations: Code explanation lessons
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                before_code TEXT NOT NULL,
                after_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bug_fixes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                before_code TEXT NOT NULL,
                after_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_implementations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                before_code TEXT NOT NULL,
                after_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                before_code TEXT NOT NULL,
                after_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                before_code TEXT NOT NULL,
                after_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def lesson_exists(self, lesson_name: str) -> bool:
        """Checks if a lesson with the given name already exists."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM lessons WHERE name = ?", (lesson_name,))
        return cursor.fetchone() is not None

    def add_lesson(self, lesson_data: Dict[str, Any]):
        """Adds a new lesson to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO lessons (name, description, focus_area, before_code, after_code, test_code, timestamp)
                VALUES (:name, :description, :focus_area, :before_code, :after_code, :test_code, :timestamp)
            """, lesson_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Lesson '{lesson_data['name']}' already exists in the database. Skipping.")

    def add_bug_fix(self, bug_fix_data: Dict[str, Any]):
        """Adds a new bug fix to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO bug_fixes (name, description, focus_area, before_code, after_code, test_code, timestamp)
                VALUES (:name, :description, :focus_area, :before_code, :after_code, :test_code, :timestamp)
            """, bug_fix_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Bug fix '{bug_fix_data['name']}' already exists in the database. Skipping.")

    def add_feature_implementation(self, feature_data: Dict[str, Any]):
        """Adds a new feature implementation to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO feature_implementations (name, description, focus_area, before_code, after_code, test_code, timestamp)
                VALUES (:name, :description, :focus_area, :before_code, :after_code, :test_code, :timestamp)
            """, feature_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Feature implementation '{feature_data['name']}' already exists in the database. Skipping.")

    def add_performance_optimization(self, performance_data: Dict[str, Any]):
        """Adds a new performance optimization to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO performance_optimizations (name, description, focus_area, before_code, after_code, test_code, timestamp)
                VALUES (:name, :description, :focus_area, :before_code, :after_code, :test_code, :timestamp)
            """, performance_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Performance optimization '{performance_data['name']}' already exists in the database. Skipping.")

    def add_code_explanation(self, explanation_data: Dict[str, Any]):
        """Adds a new code explanation to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO code_explanations (name, description, focus_area, before_code, after_code, test_code, timestamp)
                VALUES (:name, :description, :focus_area, :before_code, :after_code, :test_code, :timestamp)
            """, explanation_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Code explanation '{explanation_data['name']}' already exists in the database. Skipping.")

    def close(self) -> None:
        """
        Close the database connection.
        
        This method properly closes the SQLite connection and sets it to None
        to prevent resource leaks.
        """
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "CurriculumStore":
        """
        Context manager entry.
        
        Returns:
            Self for use in 'with' statements
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.close()
    
    def get_lesson_count(self) -> int:
        """
        Get total number of lessons across all tables.
        
        Returns:
            Total count of all lessons in the database
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM lessons")
            count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM bug_fixes")
            count += cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM feature_implementations")
            count += cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM performance_optimizations")
            count += cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM code_explanations")
            count += cursor.fetchone()[0]
            return count
        finally:
            cursor.close()
    
    def get_lessons_batch(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get lessons in batches to avoid memory issues.
        
        Args:
            limit: Maximum number of lessons to return
            offset: Number of lessons to skip
            
        Returns:
            List of lesson dictionaries with metadata
        """
        cursor = self._conn.cursor()
        try:
            # Use UNION ALL to combine all lesson types efficiently
            cursor.execute("""
                SELECT name, description, focus_area, before_code, after_code, test_code, timestamp, 'lesson' as type
                FROM lessons
                UNION ALL
                SELECT name, description, focus_area, before_code, after_code, test_code, timestamp, 'bug_fix' as type
                FROM bug_fixes
                UNION ALL
                SELECT name, description, focus_area, before_code, after_code, test_code, timestamp, 'feature' as type
                FROM feature_implementations
                UNION ALL
                SELECT name, description, focus_area, before_code, after_code, test_code, timestamp, 'performance' as type
                FROM performance_optimizations
                UNION ALL
                SELECT name, description, focus_area, before_code, after_code, test_code, timestamp, 'explanation' as type
                FROM code_explanations
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_all_bug_fixes(self) -> List[Dict[str, Any]]:
        """
        Get all bug fixes from the database.

        Returns:
            List of bug fix dictionaries
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                SELECT name, description, focus_area, before_code, after_code, test_code, timestamp
                FROM bug_fixes
                ORDER BY timestamp DESC
            """)

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()
