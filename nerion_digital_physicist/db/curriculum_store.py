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

    def close(self):
        """Closes the database connection."""
        self._conn.close()
