"""
Tests for TradeSense Data Layer

Run with: pytest tests/test_data_layer.py -v
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.data import (
    watchlist, WatchlistManager, AssetCategory,
    stock_fetcher, StockFetcher,
    crypto_fetcher, CryptoFetcher,
    fundamentals_fetcher, FundamentalsFetcher,
    options_fetcher, OptionsFetcher,
    cache, DataCache,
)


class TestWatchlistManager:
    """Tests for WatchlistManager."""
    
    def test_default_watchlists(self):
        """Test default watchlists are populated."""
        wm = WatchlistManager()
        
        assert len(wm.get_tickers(AssetCategory.CANADIAN_STOCKS)) == 5
        assert len(wm.get_tickers(AssetCategory.US_STOCKS)) == 5
        assert len(wm.get_tickers(AssetCategory.ETFS)) == 5
        assert len(wm.get_tickers(AssetCategory.CRYPTO)) == 5
    
    def test_get_all_tickers(self):
        """Test getting all tickers."""
        wm = WatchlistManager()
        all_tickers = wm.get_all_tickers()
        
        assert len(all_tickers) == 20
        assert "AAPL" in all_tickers
        assert "RY.TO" in all_tickers
        assert "BTC-CAD" in all_tickers
    
    def test_add_ticker(self):
        """Test adding a ticker."""
        wm = WatchlistManager()
        
        assert wm.add_ticker(AssetCategory.US_STOCKS, "TSLA")
        assert "TSLA" in wm.get_tickers(AssetCategory.US_STOCKS)
        
        # Duplicate should return False
        assert not wm.add_ticker(AssetCategory.US_STOCKS, "TSLA")
    
    def test_remove_ticker(self):
        """Test removing a ticker."""
        wm = WatchlistManager()
        
        assert wm.remove_ticker(AssetCategory.US_STOCKS, "AAPL")
        assert "AAPL" not in wm.get_tickers(AssetCategory.US_STOCKS)
        
        # Non-existent should return False
        assert not wm.remove_ticker(AssetCategory.US_STOCKS, "FAKE")
    
    def test_is_crypto(self):
        """Test crypto detection."""
        wm = WatchlistManager()
        
        assert wm.is_crypto("BTC-CAD")
        assert not wm.is_crypto("AAPL")
    
    def test_is_canadian(self):
        """Test Canadian stock detection."""
        wm = WatchlistManager()
        
        assert wm.is_canadian("RY.TO")
        assert not wm.is_canadian("AAPL")


class TestStockFetcher:
    """Tests for StockFetcher."""
    
    def test_get_historical(self):
        """Test fetching historical data."""
        df = stock_fetcher.get_historical("AAPL", period="5d")
        
        assert not df.empty
        assert "Open" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
    
    def test_get_quote(self):
        """Test fetching quote."""
        quote = stock_fetcher.get_quote("AAPL")
        
        assert quote["ticker"] == "AAPL"
        assert "price" in quote
        assert "error" not in quote or quote.get("price") is not None


class TestCryptoFetcher:
    """Tests for CryptoFetcher."""
    
    def test_normalize_ticker(self):
        """Test ticker normalization."""
        fetcher = CryptoFetcher()
        
        assert fetcher._normalize_ticker("BTC") == "BTC-CAD"
        assert fetcher._normalize_ticker("BTC-USD") == "BTC-USD"
        assert fetcher._normalize_ticker("btc-cad") == "BTC-CAD"
    
    def test_get_historical(self):
        """Test fetching crypto historical data."""
        df = crypto_fetcher.get_historical("BTC-CAD", period="5d")
        
        assert not df.empty
        assert "Close" in df.columns


class TestDataCache:
    """Tests for DataCache."""
    
    def test_set_and_get_dict(self):
        """Test caching dictionaries."""
        test_data = {"key": "value", "number": 42}
        
        cache.set("test_dict", test_data, ttl=60)
        result = cache.get("test_dict")
        
        assert result == test_data
    
    def test_set_and_get_dataframe(self):
        """Test caching DataFrames."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        cache.set("test_df", df, ttl=60)
        result = cache.get("test_df")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_expiration(self):
        """Test cache expiration."""
        cache.set("test_expire", "value", ttl=1)
        
        # Should exist immediately
        assert cache.get("test_expire") == "value"
        
        # After TTL, should return default
        import time
        time.sleep(2)
        assert cache.get("test_expire") is None
    
    def test_delete(self):
        """Test cache deletion."""
        cache.set("test_delete", "value", ttl=60)
        assert cache.get("test_delete") == "value"
        
        cache.delete("test_delete")
        assert cache.get("test_delete") is None
    
    def test_stats(self):
        """Test cache statistics."""
        stats = cache.get_stats()
        
        assert "cache_entries" in stats
        assert "db_path" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
