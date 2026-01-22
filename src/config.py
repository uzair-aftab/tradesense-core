"""
TradeSense Configuration Module

Handles all configuration loading from environment variables
and provides sensible defaults for the trading signal scanner.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Central configuration class for TradeSense.
    
    All configuration values are loaded from environment variables
    with sensible defaults. Use .env file for local development.
    """
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # API Keys
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    MARKETAUX_API_KEY: str = os.getenv("MARKETAUX_API_KEY", "")
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "TradeSense/1.0")
    
    # Scanner Settings
    SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
    BATCH_MODE: bool = os.getenv("BATCH_MODE", "false").lower() == "true"
    
    # Signal Thresholds (defaults from user requirements)
    RSI_OVERSOLD: int = int(os.getenv("RSI_OVERSOLD", "20"))
    RSI_OVERBOUGHT: int = int(os.getenv("RSI_OVERBOUGHT", "80"))
    PUT_CALL_RATIO_THRESHOLD: float = float(os.getenv("PUT_CALL_RATIO_THRESHOLD", "0.7"))
    VOLUME_MULTIPLIER: float = float(os.getenv("VOLUME_MULTIPLIER", "1.0"))
    
    # Database
    CACHE_DB_PATH: str = os.getenv("CACHE_DB_PATH", str(DATA_DIR / "cache.db"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    NEWS_CACHE_TTL: int = int(os.getenv("NEWS_CACHE_TTL", "1800"))  # 30 minutes
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", str(LOGS_DIR / "tradesense.log"))
    
    # Notifications
    ENABLE_DESKTOP_NOTIFICATIONS: bool = os.getenv(
        "ENABLE_DESKTOP_NOTIFICATIONS", "true"
    ).lower() == "true"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def has_finnhub_key(cls) -> bool:
        """Check if Finnhub API key is configured."""
        return bool(cls.FINNHUB_API_KEY)
    
    @classmethod
    def has_reddit_credentials(cls) -> bool:
        """Check if Reddit API credentials are configured."""
        return bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET)


# Create singleton instance
config = Config()
config.ensure_directories()
