"""
User Authentication & Account Management Module.
Manages user accounts, SQLite database persistence, and secure SHA-256 password hashing.
"""

import os
import sqlite3
import hashlib
import secrets
from typing import Tuple, Optional


class AuthManager:
    """Handles user registration, credential verification, and account persistence."""

    def __init__(self, db_path: str = "users.db"):
        self.db_path = os.path.abspath(db_path)
        self._init_db()
        self._seed_default_user()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes SQLite users table."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """Computes salted SHA-256 hash."""
        return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

    def _seed_default_user(self):
        """Seeds default project user if database is empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            if count == 0:
                salt = secrets.token_hex(8)
                pwd_hash = self._hash_password("admin123", salt)
                cursor.execute(
                    """
                    INSERT INTO users (username, password_hash, salt, full_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("admin", pwd_hash, salt, "Project Administrator")
                )
                conn.commit()

    def register_user(
        self,
        username: str,
        password: str,
        confirm_password: str,
        full_name: str
    ) -> Tuple[bool, str]:
        """
        Registers a new user account.
        Returns (success: bool, message: str)
        """
        username = username.strip().lower()
        full_name = full_name.strip()

        if not username or not password:
            return False, "Username and password are required."

        if len(username) < 3:
            return False, "Username must be at least 3 characters long."

        if len(password) < 4:
            return False, "Password must be at least 4 characters long."

        if password != confirm_password:
            return False, "Passwords do not match."

        if not full_name:
            full_name = username.capitalize()

        salt = secrets.token_hex(8)
        pwd_hash = self._hash_password(password, salt)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (username, password_hash, salt, full_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, pwd_hash, salt, full_name)
                )
                conn.commit()
                return True, f"Account created successfully for {username}! You can now sign in."
        except sqlite3.IntegrityError:
            return False, "Username already exists. Please choose a different username."
        except Exception as e:
            return False, f"Registration failed: {str(e)}"

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """
        Verifies login credentials.
        Returns (success: bool, message: str, full_name: Optional[str])
        """
        username = username.strip().lower()
        if not username or not password:
            return False, "Please enter both username and password.", None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password_hash, salt, full_name FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()

            if not row:
                return False, "Invalid username or password.", None

            stored_hash, salt, full_name = row
            computed_hash = self._hash_password(password, salt)

            if computed_hash == stored_hash:
                return True, f"Welcome back, {full_name}!", full_name
            else:
                return False, "Invalid username or password.", None


# Global authentication manager instance
auth_db = AuthManager()
