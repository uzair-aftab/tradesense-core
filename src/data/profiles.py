"""
TradeSense Passwordless Multi-Account Profile Engine
====================================================
Manages friendly, passwordless trading profiles (Uzair, Waleed, and on-demand added accounts):
  - 1-Click Profile Cards on login screen
  - Cookie-based session token persistence (zero ?user= query param clutter)
  - Isolated SQLite WAL portfolios, virtual ledgers, and real Wealthsimple logs
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.data.database import DatabaseManager, db_manager
from src.utils.logger import get_logger

logger = get_logger("hfp.profiles")

COLOR_PALETTE = [
    "#38bdf8",  # Ice Cyan (Uzair)
    "#34d399",  # Mint Emerald (Waleed)
    "#a78bfa",  # Soft Violet
    "#fbbf24",  # Warm Amber
    "#f472b6",  # Rose Quartz
    "#22d3ee",  # Electric Sky
]


@dataclass
class UserProfile:
    """Represents a passwordless user identity in TradeSense."""
    profile_id: str
    display_name: str
    avatar_initials: str
    accent_color: str = "#38bdf8"
    ws_tier: str = "Core"
    usd_account_active: bool = False
    active_symbol: str = "BTC-CAD"
    active_timeframe: str = "1h"
    horizon_bars: int = 5
    model_preset: str = "AUTO_DNA"
    selected_models: List[str] = field(default_factory=list)
    blending_mode: str = "WALK_FORWARD"
    use_proxy_route: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["selected_models_json"] = json.dumps(self.selected_models)
        d["use_proxy_route"] = 1 if self.use_proxy_route else 0
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> UserProfile:
        models = []
        raw_m = row.get("selected_models_json", "[]")
        if raw_m:
            try:
                models = json.loads(raw_m)
            except Exception:
                models = []
        return cls(
            profile_id=row["profile_id"],
            display_name=row["display_name"],
            avatar_initials=row.get("avatar_initials", row["display_name"][:2].upper()),
            accent_color=row.get("accent_color", "#38bdf8"),
            ws_tier=row.get("ws_tier", "Core"),
            usd_account_active=bool(row.get("usd_account_active", 0)),
            active_symbol=row.get("active_symbol", "BTC-CAD"),
            active_timeframe=row.get("active_timeframe", "1h"),
            horizon_bars=int(row.get("horizon_bars", 5)),
            model_preset=row.get("model_preset", "AUTO_DNA"),
            selected_models=models,
            blending_mode=row.get("blending_mode", "WALK_FORWARD"),
            use_proxy_route=bool(row.get("use_proxy_route", 1)),
        )


class ProfileManager:
    """Coordinates multi-account lifecycle and preferences."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    def ensure_default_profiles(self) -> None:
        """Guarantees default Uzair and Waleed profiles exist on startup."""
        defaults = [
            ("uzair", "Uzair", "UZ", "#38bdf8", "Core"),
            ("waleed", "Waleed", "WA", "#34d399", "Core"),
        ]
        for pid, name, initials, color, tier in defaults:
            existing = self.db.get_user_profile(pid)
            if not existing:
                self.db.save_user_profile({
                    "profile_id": pid,
                    "display_name": name,
                    "avatar_initials": initials,
                    "accent_color": color,
                    "ws_tier": tier,
                    "usd_account_active": 0,
                    "active_symbol": "BTC-CAD",
                    "active_timeframe": "1h",
                    "horizon_bars": 5,
                    "model_preset": "AUTO_DNA",
                    "selected_models_json": "[]",
                    "blending_mode": "WALK_FORWARD",
                    "use_proxy_route": 1,
                })
                # Ensure isolated virtual trading wallet
                self.db.get_or_create_virtual_account(pid, starting_cash=10000.0, ws_tier=tier)
                logger.info("Created default profile: %s", name)

    def list_profiles(self) -> List[UserProfile]:
        """List all available profiles."""
        rows = self.db.list_user_profiles()
        return [UserProfile.from_row(r) for r in rows]

    def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Retrieve profile by slug."""
        if not profile_id:
            return None
        row = self.db.get_user_profile(str(profile_id).lower())
        return UserProfile.from_row(row) if row else None

    def save_profile(self, profile: UserProfile) -> UserProfile:
        """Persist a UserProfile instance directly."""
        self.db.save_user_profile(profile.to_dict())
        return profile

    def create_profile(
        self,
        display_name: str,
        starting_cad: float = 10000.0,
        ws_tier: str = "Core",
        accent_color: Optional[str] = None,
    ) -> UserProfile:
        """Create a new profile with isolated wallet and preferences."""
        clean_name = display_name.strip()
        slug = re.sub(r"\s+", "_", clean_name.lower())
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        if not slug:
            slug = f"user_{len(self.list_profiles()) + 1}"

        # Guarantee unique slug
        base_slug = slug
        counter = 2
        while self.db.get_user_profile(slug):
            slug = f"{base_slug}_{counter}"
            counter += 1

        # Pick accent color
        existing_count = len(self.list_profiles())
        color = accent_color or COLOR_PALETTE[existing_count % len(COLOR_PALETTE)]

        # Initials
        parts = clean_name.split()
        if len(parts) >= 2:
            initials = (parts[0][0] + parts[1][0]).upper()
        else:
            initials = clean_name[:2].upper()

        profile = UserProfile(
            profile_id=slug,
            display_name=clean_name,
            avatar_initials=initials,
            accent_color=color,
            ws_tier=ws_tier,
            usd_account_active=False,
            active_symbol="BTC-CAD",
            active_timeframe="1h",
            horizon_bars=5,
            model_preset="AUTO_DNA",
            selected_models=[],
            blending_mode="WALK_FORWARD",
            use_proxy_route=True,
        )

        self.db.save_user_profile(profile.to_dict())
        self.db.get_or_create_virtual_account(slug, starting_cash=starting_cad, ws_tier=ws_tier)
        logger.info("Registered new profile: %s (%s)", clean_name, slug)
        return profile

    def update_profile(self, profile_id: str, **kwargs) -> Optional[UserProfile]:
        """Update profile preferences in database."""
        curr = self.get_profile(profile_id)
        if not curr:
            return None
        d = curr.to_dict()
        for k, v in kwargs.items():
            if k == "selected_models":
                d["selected_models_json"] = json.dumps(v if isinstance(v, list) else [])
                d["selected_models"] = v
            elif k == "use_proxy_route":
                d["use_proxy_route"] = 1 if v else 0
            elif k in d:
                d[k] = v
        self.db.save_user_profile(d)
        return self.get_profile(profile_id)


# Global singleton instance
profile_manager = ProfileManager()
