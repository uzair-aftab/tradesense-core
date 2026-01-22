"""
TradeSense Watchlist Manager

Manages watchlists for stocks, ETFs, and crypto with
add/remove/expand functionality. All assets are legal
for trading in Ontario, Canada.
"""

from typing import List, Dict, Optional
from enum import Enum


class AssetCategory(Enum):
    """Asset category enumeration."""
    CANADIAN_STOCKS = "canadian_stocks"
    US_STOCKS = "us_stocks"
    ETFS = "etfs"
    CRYPTO = "crypto"


class WatchlistManager:
    """
    Manages trading watchlists with add/remove/expand functionality.
    
    Default watchlists contain 5 tickers per category, easily expandable.
    All assets are legal for trading in Ontario, Canada.
    
    Example:
        >>> wm = WatchlistManager()
        >>> wm.add_ticker(AssetCategory.CANADIAN_STOCKS, "BCE.TO")
        >>> all_tickers = wm.get_all_tickers()
    """
    
    # Default watchlists - 5 per category (easily expandable)
    _DEFAULTS = {
        AssetCategory.CANADIAN_STOCKS: [
            "RY.TO",    # Royal Bank of Canada
            "TD.TO",    # TD Bank
            "SHOP.TO",  # Shopify
            "ENB.TO",   # Enbridge
            "CNR.TO",   # CN Rail
        ],
        AssetCategory.US_STOCKS: [
            "AAPL",     # Apple
            "MSFT",     # Microsoft
            "NVDA",     # NVIDIA
            "GOOGL",    # Alphabet
            "AMZN",     # Amazon
        ],
        AssetCategory.ETFS: [
            "SPY",      # S&P 500
            "QQQ",      # Nasdaq 100
            "XIU.TO",   # iShares TSX 60
            "VFV.TO",   # Vanguard S&P 500 (CAD)
            "ARKK",     # ARK Innovation
        ],
        AssetCategory.CRYPTO: [
            "BTC-CAD",  # Bitcoin
            "ETH-CAD",  # Ethereum
            "SOL-CAD",  # Solana
            "ADA-CAD",  # Cardano
            "AVAX-CAD", # Avalanche
        ],
    }
    
    def __init__(self):
        """Initialize watchlist with default tickers."""
        self._watchlists: Dict[AssetCategory, List[str]] = {
            category: list(tickers) 
            for category, tickers in self._DEFAULTS.items()
        }
    
    def get_tickers(self, category: AssetCategory) -> List[str]:
        """
        Get tickers for a specific category.
        
        Args:
            category: Asset category to retrieve
            
        Returns:
            List of ticker symbols
        """
        return list(self._watchlists.get(category, []))
    
    def get_all_tickers(self) -> List[str]:
        """
        Get all tickers across all categories.
        
        Returns:
            Flattened list of all ticker symbols
        """
        all_tickers = []
        for tickers in self._watchlists.values():
            all_tickers.extend(tickers)
        return all_tickers
    
    def get_stocks(self) -> List[str]:
        """Get all stock tickers (Canadian + US)."""
        return (
            self.get_tickers(AssetCategory.CANADIAN_STOCKS) +
            self.get_tickers(AssetCategory.US_STOCKS)
        )
    
    def get_etfs(self) -> List[str]:
        """Get all ETF tickers."""
        return self.get_tickers(AssetCategory.ETFS)
    
    def get_crypto(self) -> List[str]:
        """Get all crypto tickers."""
        return self.get_tickers(AssetCategory.CRYPTO)
    
    def add_ticker(self, category: AssetCategory, ticker: str) -> bool:
        """
        Add a ticker to a category.
        
        Args:
            category: Target category
            ticker: Ticker symbol to add
            
        Returns:
            True if added, False if already exists
        """
        ticker = ticker.upper()
        if ticker not in self._watchlists[category]:
            self._watchlists[category].append(ticker)
            return True
        return False
    
    def remove_ticker(self, category: AssetCategory, ticker: str) -> bool:
        """
        Remove a ticker from a category.
        
        Args:
            category: Target category
            ticker: Ticker symbol to remove
            
        Returns:
            True if removed, False if not found
        """
        ticker = ticker.upper()
        if ticker in self._watchlists[category]:
            self._watchlists[category].remove(ticker)
            return True
        return False
    
    def get_category_for_ticker(self, ticker: str) -> Optional[AssetCategory]:
        """
        Find which category a ticker belongs to.
        
        Args:
            ticker: Ticker symbol to find
            
        Returns:
            AssetCategory or None if not found
        """
        ticker = ticker.upper()
        for category, tickers in self._watchlists.items():
            if ticker in tickers:
                return category
        return None
    
    def is_crypto(self, ticker: str) -> bool:
        """Check if ticker is a cryptocurrency."""
        return ticker.upper() in self._watchlists[AssetCategory.CRYPTO]
    
    def is_canadian(self, ticker: str) -> bool:
        """Check if ticker is a Canadian stock (ends with .TO)."""
        return ticker.upper().endswith(".TO")
    
    def reset_to_defaults(self) -> None:
        """Reset all watchlists to default values."""
        self._watchlists = {
            category: list(tickers) 
            for category, tickers in self._DEFAULTS.items()
        }
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get summary of watchlist counts.
        
        Returns:
            Dict with category names and ticker counts
        """
        return {
            category.value: len(tickers)
            for category, tickers in self._watchlists.items()
        }


# Singleton instance for easy import
watchlist = WatchlistManager()
