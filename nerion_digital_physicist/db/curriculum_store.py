"""
This module provides a simple SQLite-backed store for the curriculum.
"""
import sqlite3
from pathlib import Path
from typing import Dict, Any

class CurriculumStore:
    """Handles all database operations for the curriculum."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._bootstrap()

    def _bootstrap(self):
        """Creates the necessary tables if they don't exist."""
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
                buggy_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                fixed_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_implementations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                initial_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                final_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                inefficient_code TEXT NOT NULL,
                test_code TEXT NOT NULL,
                optimized_code TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                focus_area TEXT,
                code_snippet TEXT NOT NULL,
                explanation TEXT NOT NULL,
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
                INSERT INTO bug_fixes (name, description, focus_area, buggy_code, test_code, fixed_code, timestamp)
                VALUES (:name, :description, :focus_area, :buggy_code, :test_code, :fixed_code, :timestamp)
            """, bug_fix_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Bug fix '{bug_fix_data['name']}' already exists in the database. Skipping.")

    def add_feature_implementation(self, feature_data: Dict[str, Any]):
        """Adds a new feature implementation to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO feature_implementations (name, description, focus_area, initial_code, test_code, final_code, timestamp)
                VALUES (:name, :description, :focus_area, :initial_code, :test_code, :final_code, :timestamp)
            """, feature_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Feature implementation '{feature_data['name']}' already exists in the database. Skipping.")

    def add_performance_optimization(self, performance_data: Dict[str, Any]):
        """Adds a new performance optimization to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO performance_optimizations (name, description, focus_area, inefficient_code, test_code, optimized_code, timestamp)
                VALUES (:name, :description, :focus_area, :inefficient_code, :test_code, :optimized_code, :timestamp)
            """, performance_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Performance optimization '{performance_data['name']}' already exists in the database. Skipping.")

    def add_code_explanation(self, explanation_data: Dict[str, Any]):
        """Adds a new code explanation to the database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO code_explanations (name, description, focus_area, code_snippet, explanation, timestamp)
                VALUES (:name, :description, :focus_area, :code_snippet, :explanation, :timestamp)
            """, explanation_data)
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"  - WARNING: Code explanation '{explanation_data['name']}' already exists in the database. Skipping.")

    def close(self):
        """Closes the database connection."""
        self._conn.close()
