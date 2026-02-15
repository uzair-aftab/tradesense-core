"""
TradeSense Data Cache

SQLite-based caching layer for market data with TTL support.
Prevents redundant API calls and stores historical data locally.
"""

import sqlite3
import json
import pickle
from typing import Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCache:
    """
    SQLite-based data cache with TTL support.
    
    Caches various data types including DataFrames, dicts, and
    primitive values with configurable time-to-live.
    
    Example:
        >>> cache = DataCache()
        >>> cache.set("AAPL_price", df, ttl=3600)
        >>> df = cache.get("AAPL_price")
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the cache.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.CACHE_DB_PATH
        self._ensure_db_exists()
        self._init_db()
    
    def _ensure_db_exists(self) -> None:
        """Ensure the database directory exists."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    data_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Price data table (optimized for time series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    ticker TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON cache(expires_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_ticker 
                ON price_data(ticker)
            """)
            
            conn.commit()
            
        logger.debug(f"Initialized cache database at {self.db_path}")

    def _json_default(self, obj: Any) -> Any:
        """JSON serializer for numpy/pandas objects."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return str(obj)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store (DataFrame, dict, or primitive)
            ttl: Time-to-live in seconds (uses default if None)
            metadata: Optional metadata dict
            
        Returns:
            True if stored successfully
        """
        try:
            ttl = ttl or config.CACHE_TTL_SECONDS
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Serialize value based on type
            if isinstance(value, pd.DataFrame):
                serialized = pickle.dumps(value)
                data_type = "dataframe"
            elif isinstance(value, dict):
                serialized = json.dumps(value, default=self._json_default).encode()
                data_type = "dict"
            elif isinstance(value, (list, tuple)):
                serialized = json.dumps(value, default=self._json_default).encode()
                data_type = "list"
            else:
                serialized = pickle.dumps(value)
                data_type = "pickle"
            
            metadata_str = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO cache 
                    (key, value, data_type, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, serialized, data_type, expires_at, metadata_str))
                conn.commit()
            
            logger.debug(f"Cached {key} (type={data_type}, ttl={ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching {key}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached value or default
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value, data_type, expires_at 
                    FROM cache 
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                
                if row is None:
                    return default
                
                value, data_type, expires_at = row
                
                # Check expiration
                if expires_at:
                    exp_time = datetime.fromisoformat(expires_at)
                    if exp_time < datetime.now():
                        # Expired, delete and return default
                        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                        conn.commit()
                        logger.debug(f"Cache expired for {key}")
                        return default
                
                # Deserialize based on type
                if data_type == "dataframe":
                    return pickle.loads(value)
                elif data_type == "dict":
                    return json.loads(value.decode())
                elif data_type == "list":
                    return json.loads(value.decode())
                else:
                    return pickle.loads(value)
                    
        except Exception as e:
            logger.error(f"Error retrieving {key} from cache: {e}")
            return default
    
    def has(self, key: str) -> bool:
        """
        Check if key exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is valid
        """
        return self.get(key) is not None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting {key}: {e}")
            return False
    
    def clear(self) -> int:
        """
        Clear all cached data.
        
        Returns:
            Number of items deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                cursor.execute("DELETE FROM cache")
                conn.commit()
                logger.info(f"Cleared {count} items from cache")
                return count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of items removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE expires_at < ?
                """, (datetime.now().isoformat(),))
                count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleaned up {count} expired cache entries")
                return count
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def save_price_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save price data to optimized table.
        
        Args:
            ticker: Stock/crypto symbol
            df: DataFrame with OHLCV data
            
        Returns:
            True if saved successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                for idx, row in df.iterrows():
                    date_val = idx.date() if hasattr(idx, 'date') else idx
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (ticker, date, open, high, low, close, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker,
                        date_val,
                        row.get("Open"),
                        row.get("High"),
                        row.get("Low"),
                        row.get("Close"),
                        row.get("Volume"),
                        row.get("Adj Close"),
                    ))
                conn.commit()
                
            logger.debug(f"Saved {len(df)} price rows for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving price data for {ticker}: {e}")
            return False
    
    def get_price_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve price data from optimized table.
        
        Args:
            ticker: Stock/crypto symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            query = "SELECT * FROM price_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df.drop(columns=["ticker", "updated_at"], inplace=True, errors="ignore")
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "adj_close": "Adj Close",
                })
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM cache")
                cache_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at < ?", 
                             (datetime.now().isoformat(),))
                expired_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT ticker) FROM price_data")
                ticker_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM price_data")
                price_rows = cursor.fetchone()[0]
                
                return {
                    "cache_entries": cache_count,
                    "expired_entries": expired_count,
                    "tickers_cached": ticker_count,
                    "price_data_rows": price_rows,
                    "db_path": self.db_path,
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


# Singleton instance
cache = DataCache()
