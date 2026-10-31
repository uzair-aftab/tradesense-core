"""
TradeSense v7 — Data Layer
==========================
Data ingestion, caching, Wealthsimple catalog & fee schedule, user profiles & database.
"""

from src.data.data_feed import DataLake
from src.data.cache import DataCache, cache
from src.data.ws_catalog import (
    WealthsimpleCatalog,
    WealthsimpleAsset,
    FeeEstimate,
    WealthsimpleFeeSchedule,
    CURATED_WS_ASSETS,
)
from src.data.database import DatabaseManager, db_manager, get_database
from src.data.profiles import ProfileManager, profile_manager, UserProfile
from src.data.market_intel import MarketIntelligenceEngine, market_intel

__all__ = [
    "DataLake",
    "DataCache",
    "cache",
    "WealthsimpleCatalog",
    "WealthsimpleAsset",
    "FeeEstimate",
    "WealthsimpleFeeSchedule",
    "CURATED_WS_ASSETS",
    "DatabaseManager",
    "db_manager",
    "get_database",
    "ProfileManager",
    "profile_manager",
    "UserProfile",
    "MarketIntelligenceEngine",
    "market_intel",
]
