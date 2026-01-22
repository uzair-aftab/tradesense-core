"""
TradeSense Data Layer

Data fetching, caching, and watchlist management.
"""

from src.data.watchlists import WatchlistManager, AssetCategory, watchlist
from src.data.fetcher_stocks import StockFetcher, stock_fetcher
from src.data.fetcher_crypto import CryptoFetcher, crypto_fetcher
from src.data.fetcher_fundamentals import FundamentalsFetcher, fundamentals_fetcher
from src.data.fetcher_options import OptionsFetcher, options_fetcher
from src.data.cache import DataCache, cache

__all__ = [
    # Watchlists
    "WatchlistManager",
    "AssetCategory",
    "watchlist",
    # Stocks
    "StockFetcher",
    "stock_fetcher",
    # Crypto
    "CryptoFetcher",
    "crypto_fetcher",
    # Fundamentals
    "FundamentalsFetcher",
    "fundamentals_fetcher",
    # Options
    "OptionsFetcher",
    "options_fetcher",
    # Cache
    "DataCache",
    "cache",
]
