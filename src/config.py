"""
HFP Configuration Module

Central configuration for the High-Frequency Prediction system.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration for HFP.

    All values are loaded from environment variables with sensible defaults.
    Use .env file for local development.
    """

    # ── Paths ──────────────────────────────────────────────────────────
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models" / "hfp"

    # ── API Keys ───────────────────────────────────────────────────────
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    MARKETAUX_API_KEY: str = os.getenv("MARKETAUX_API_KEY", "")
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "HFP/2.0")

    # ── Engine Defaults ────────────────────────────────────────────────
    DEFAULT_TIMEFRAME: str = os.getenv("DEFAULT_TIMEFRAME", "1d")
    DEFAULT_LOOKBACK: str = os.getenv("DEFAULT_LOOKBACK", "1y")
    LSTM_EPOCHS: int = int(os.getenv("LSTM_EPOCHS", "50"))
    LSTM_HIDDEN: int = int(os.getenv("LSTM_HIDDEN", "128"))
    LSTM_SEQ_LEN: int = int(os.getenv("LSTM_SEQ_LEN", "30"))
    XGB_ESTIMATORS: int = int(os.getenv("XGB_ESTIMATORS", "500"))
    SENTIMENT_BACKEND: str = os.getenv("SENTIMENT_BACKEND", "vader")

    # ── Paper Trading ──────────────────────────────────────────────────
    STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "10000"))
    PAPER_PORTFOLIO_PATH: str = os.getenv(
        "PAPER_PORTFOLIO_PATH", str(DATA_DIR / "paper_portfolio.json")
    )

    # ── Cache ──────────────────────────────────────────────────────────
    CACHE_DB_PATH: str = os.getenv("CACHE_DB_PATH", str(DATA_DIR / "cache.db"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # ── UI ─────────────────────────────────────────────────────────────
    AUTO_REFRESH_INTERVAL: int = int(os.getenv("AUTO_REFRESH_INTERVAL", "10"))

    # ── Logging ────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", str(LOGS_DIR / "tradesense.log"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def has_finnhub_key(cls) -> bool:
        return bool(cls.FINNHUB_API_KEY)

    @classmethod
    def has_reddit_credentials(cls) -> bool:
        return bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET)


# Singleton
config = Config()
config.ensure_directories()
