"""
TradeSense Passwordless Cookie Session Manager
===============================================
Issues and verifies HMAC-SHA256 signed session tokens stored in HttpOnly cookies:
  - Cookie name: tradesense_session
  - 90-day TTL for frictionless continuous access
  - Zero URL parameter clutter (?user=uzair / ?user=waleed eliminated)
  - Persistent secret stored in data/.session_secret
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Optional

from src.config import config
from src.data.profiles import UserProfile, profile_manager

SESSION_COOKIE_NAME = "tradesense_session"
SESSION_TTL_SECONDS = 90 * 24 * 3600  # 90 days
SECRET_KEY_PATH = config.DATA_DIR / ".session_secret"


def _get_secret_key() -> bytes:
    """Return persistent HMAC secret key, creating one if not present."""
    SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SECRET_KEY_PATH.exists():
        try:
            raw = SECRET_KEY_PATH.read_text(encoding="utf-8").strip()
            if len(raw) >= 32:
                return raw.encode("utf-8")
        except Exception:
            pass
    secret = secrets.token_hex(32)
    SECRET_KEY_PATH.write_text(secret, encoding="utf-8")
    return secret.encode("utf-8")


def create_session_token(profile_id: str) -> str:
    """Create an HMAC-SHA256 signed session token for a profile."""
    payload_dict = {
        "pid": str(profile_id).lower(),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
        "nonce": secrets.token_hex(8),
    }
    payload_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    sig = hmac.new(_get_secret_key(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_session_token(token: Optional[str]) -> Optional[str]:
    """
    Verify an HMAC-SHA256 signed session token.
    Returns profile_id if valid and unexpired, None otherwise.
    """
    if not token or "." not in token:
        return None
    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(_get_secret_key(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        pad = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + pad).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        pid = payload.get("pid")
        if not pid:
            return None
        return str(pid).lower()
    except Exception:
        return None


def get_authenticated_profile(token: Optional[str]) -> Optional[UserProfile]:
    """Convenience helper: resolve valid session token directly to UserProfile."""
    pid = verify_session_token(token)
    if not pid:
        return None
    return profile_manager.get_profile(pid)
