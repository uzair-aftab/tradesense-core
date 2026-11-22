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
from src.data.finbert_engine import FinBERTEngine, FinBERTResult, finbert_engine
from src.data.finnhub_client import FinnhubClient, finnhub_client
from src.data.prediction_feed import KalshiPredictionFeed, PredictionMarket, prediction_feed

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
    "FinBERTEngine",
    "FinBERTResult",
    "finbert_engine",
    "FinnhubClient",
    "finnhub_client",
    "KalshiPredictionFeed",
    "PredictionMarket",
    "prediction_feed",
]
