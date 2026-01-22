"""
TradeSense Crypto Data Fetcher

Fetches cryptocurrency price data using yFinance.
Supports CAD pairs for Canadian users.
"""

from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoFetcher:
    """
    Fetches cryptocurrency price data from yFinance.
    
    Uses CAD pairs by default for Canadian users.
    Supports major cryptocurrencies available on regulated Canadian platforms.
    
    Example:
        >>> fetcher = CryptoFetcher()
        >>> df = fetcher.get_historical("BTC-CAD", period="1mo")
        >>> df = fetcher.get_historical("ETH-CAD", period="3mo")
    """
    
    # Valid crypto pairs with CAD
    SUPPORTED_CRYPTOS = {
        "BTC": "BTC-CAD",
        "ETH": "ETH-CAD",
        "SOL": "SOL-CAD",
        "ADA": "ADA-CAD",
        "AVAX": "AVAX-CAD",
        "DOT": "DOT-CAD",
        "LINK": "LINK-CAD",
        "MATIC": "MATIC-CAD",
        "XRP": "XRP-CAD",
        "DOGE": "DOGE-CAD",
    }
    
    def __init__(self, default_currency: str = "CAD"):
        """
        Initialize crypto fetcher.
        
        Args:
            default_currency: Default quote currency (CAD, USD)
        """
        self.default_currency = default_currency
    
    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker to include currency pair.
        
        Args:
            ticker: Crypto ticker (e.g., 'BTC', 'BTC-CAD', 'BTC-USD')
            
        Returns:
            Normalized ticker with currency pair
        """
        ticker = ticker.upper()
        
        # Already has currency pair
        if "-" in ticker:
            return ticker
        
        # Add default currency
        return f"{ticker}-{self.default_currency}"
    
    def get_historical(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a cryptocurrency.
        
        Args:
            ticker: Crypto symbol (e.g., 'BTC', 'BTC-CAD', 'ETH-CAD')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)
            interval: Data interval (1d, 1wk, 1mo for crypto)
            start: Start date string (YYYY-MM-DD)
            end: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        ticker = self._normalize_ticker(ticker)
        
        try:
            logger.debug(f"Fetching crypto data for {ticker}")
            
            crypto = yf.Ticker(ticker)
            
            if start and end:
                df = crypto.history(start=start, end=end, interval=interval)
            else:
                df = crypto.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for crypto: {ticker}")
            
            # Clean up the dataframe
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto {ticker}: {e}")
            raise
    
    def get_historical_batch(
        self,
        tickers: List[str],
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple cryptocurrencies.
        
        Args:
            tickers: List of crypto symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dict mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            normalized = self._normalize_ticker(ticker)
            try:
                results[normalized] = self.get_historical(
                    ticker, period=period, interval=interval
                )
            except Exception as e:
                logger.warning(f"Failed to fetch {normalized}: {e}")
                results[normalized] = pd.DataFrame()
        
        return results
    
    def get_quote(self, ticker: str) -> Dict:
        """
        Get current price info for a cryptocurrency.
        
        Args:
            ticker: Crypto symbol
            
        Returns:
            Dict with price info
        """
        ticker = self._normalize_ticker(ticker)
        
        try:
            crypto = yf.Ticker(ticker)
            info = crypto.info
            fast_info = crypto.fast_info
            
            quote = {
                "ticker": ticker,
                "price": fast_info.get("lastPrice"),
                "previous_close": fast_info.get("previousClose"),
                "open": fast_info.get("open"),
                "day_high": fast_info.get("dayHigh"),
                "day_low": fast_info.get("dayLow"),
                "volume": fast_info.get("lastVolume"),
                "market_cap": info.get("marketCap"),
                "currency": ticker.split("-")[-1] if "-" in ticker else "CAD",
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
            logger.error(f"Error getting crypto quote for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_quotes_batch(self, tickers: List[str]) -> List[Dict]:
        """
        Get quotes for multiple cryptocurrencies.
        
        Args:
            tickers: List of crypto symbols
            
        Returns:
            List of quote dicts
        """
        return [self.get_quote(ticker) for ticker in tickers]


# Singleton instance
crypto_fetcher = CryptoFetcher()
