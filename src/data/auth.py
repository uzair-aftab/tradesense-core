"""
TradeSense User Authentication & Session Security Engine
=========================================================
Institutional-grade security using standard Python cryptographic libraries:
  - FIPS-compliant PBKDF2-HMAC-SHA256 password hashing (100,000 iterations)
  - 16-byte cryptographically secure random per-user salt
  - 32-byte secure session tokens with 30-day expiry
  - Constant-time HMAC comparison to prevent timing attacks
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from src.data.database import DatabaseManager, db_manager
from src.utils.logger import get_logger

logger = get_logger("hfp.auth")


class AuthManager:
    """Manages user registration, authentication, and persistent web sessions."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    @staticmethod
    def hash_password(password: str, salt: bytes) -> str:
        """Hash password with PBKDF2-HMAC-SHA256 (100,000 iterations)."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations=100_000,
        ).hex()

    def register_user(
        self,
        username: str,
        password: str,
        starting_capital: float = 10000.0,
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Register a new user account.

        Returns (success: bool, message: str, user_id: int | None).
        """
        username = username.strip()
        if len(username) < 3 or len(username) > 25:
            return False, "Username must be between 3 and 25 characters", None

        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            return False, "Username can only contain alphanumeric characters and underscores", None

        if len(password) < 6:
            return False, "Password must be at least 6 characters long", None

        if self.db.get_user_by_username(username):
            return False, f"Username '{username}' is already taken", None

        salt_bytes = secrets.token_bytes(16)
        salt_hex = salt_bytes.hex()
        pwd_hash = self.hash_password(password, salt_bytes)

        try:
            user_id = self.db.create_user(
                username=username,
                password_hash=pwd_hash,
                salt=salt_hex,
                starting_capital=starting_capital,
            )
            logger.info("New user registered: %s (id=%d, capital=$%.2f)", username, user_id, starting_capital)
            return True, f"Account '{username}' created successfully! Please log in.", user_id
        except Exception as e:
            logger.error("Failed to register user %s: %s", username, e, exc_info=True)
            return False, f"Registration failed: {e}", None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials.
        On success, creates a 30-day persistent session token in SQLite.
        """
        user = self.db.get_user_by_username(username)
        if not user:
            return None

        salt = bytes.fromhex(user["salt"])
        expected_hash = self.hash_password(password, salt)

        if not hmac.compare_digest(user["password_hash"], expected_hash):
            logger.warning("Failed login attempt for user: %s", username)
            return None

        # Generate 30-day session token for F5 refresh immunity
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        self.db.create_session(user["id"], session_token, expires_at)
        self.db.update_user_last_login(user["id"])

        user_dict = dict(user)
        user_dict["session_token"] = session_token
        logger.info("User authenticated: %s (token=%s...)", username, session_token[:8])
        return user_dict

    def get_user_by_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user profile if valid."""
        if not session_token:
            return None
        return self.db.get_session(session_token)

    def logout_user(self, session_token: str) -> None:
        """Invalidate session token."""
        if session_token:
            self.db.delete_session(session_token)
            logger.info("Session invalidated: %s...", session_token[:8])


# Global singleton instance
auth_manager = AuthManager()
