"""
TradeSense Stock Data Fetcher

Fetches stock price data using yFinance as primary source.
Supports historical OHLCV data and real-time quotes.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockFetcher:
    """
    Fetches stock and ETF price data from yFinance.
    
    Provides historical OHLCV data and current quotes for
    stocks and ETFs traded on US and Canadian exchanges.
    
    Example:
        >>> fetcher = StockFetcher()
        >>> df = fetcher.get_historical("AAPL", period="3mo")
        >>> quote = fetcher.get_quote("AAPL")
    """
    
    def __init__(self):
        """Initialize the stock fetcher."""
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_historical(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'RY.TO')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Start date string (YYYY-MM-DD), overrides period
            end: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            
        Raises:
            ValueError: If ticker not found or no data available
        """
        try:
            logger.debug(f"Fetching historical data for {ticker}")
            
            stock = yf.Ticker(ticker)
            
            if start and end:
                df = stock.history(start=start, end=end, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            
            # Clean up the dataframe
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # Remove timezone info
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise
    
    def get_historical_batch(
        self,
        tickers: List[str],
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dict mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.get_historical(
                    ticker, period=period, interval=interval
                )
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        
        return results
    
    def get_quote(self, ticker: str) -> Dict:
        """
        Get current quote/price info for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with price info:
                - price: Current price
                - change: Price change
                - change_percent: Percent change
                - volume: Current volume
                - market_cap: Market capitalization
                - pe_ratio: P/E ratio
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get fast info for current price
            fast_info = stock.fast_info
            
            quote = {
                "ticker": ticker,
                "price": fast_info.get("lastPrice", info.get("currentPrice")),
                "previous_close": fast_info.get("previousClose", info.get("previousClose")),
                "open": fast_info.get("open", info.get("open")),
                "day_high": fast_info.get("dayHigh", info.get("dayHigh")),
                "day_low": fast_info.get("dayLow", info.get("dayLow")),
                "volume": fast_info.get("lastVolume", info.get("volume")),
                "market_cap": fast_info.get("marketCap", info.get("marketCap")),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "name": info.get("shortName", ticker),
            }
            
            # Calculate change
            if quote["price"] and quote["previous_close"]:
                quote["change"] = quote["price"] - quote["previous_close"]
                quote["change_percent"] = (quote["change"] / quote["previous_close"]) * 100
            else:
                quote["change"] = None
                quote["change_percent"] = None
            
            return quote
            
        except Exception as e:
            logger.error(f"Error getting quote for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_quotes_batch(self, tickers: List[str]) -> List[Dict]:
        """
        Get quotes for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            List of quote dicts
        """
        return [self.get_quote(ticker) for ticker in tickers]
    
    def get_intraday(
        self,
        ticker: str,
        interval: str = "5m",
        period: str = "1d"
    ) -> pd.DataFrame:
        """
        Get intraday price data.
        
        Args:
            ticker: Stock symbol
            interval: Time interval (1m, 2m, 5m, 15m, 30m, 60m, 90m)
            period: Period to fetch (1d, 5d)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        return self.get_historical(ticker, period=period, interval=interval)


# Singleton instance
stock_fetcher = StockFetcher()
