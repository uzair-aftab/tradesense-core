"""
TradeSense Options Data Fetcher

Fetches options chain data including put/call ratios.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionsFetcher:
    """
    Fetches options chain data for stocks.
    
    Provides options data including put/call ratios, open interest,
    and volume analysis for signal generation.
    
    Example:
        >>> fetcher = OptionsFetcher()
        >>> chain = fetcher.get_options_chain("AAPL")
        >>> pc_ratio = fetcher.get_put_call_ratio("AAPL")
    """
    
    def get_expiration_dates(self, ticker: str) -> Tuple[str, ...]:
        """
        Get available options expiration dates.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Tuple of expiration date strings
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.options
        except Exception as e:
            logger.error(f"Error getting expiration dates for {ticker}: {e}")
            return ()
    
    def get_options_chain(
        self,
        ticker: str,
        expiration: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get options chain data for a ticker.
        
        Args:
            ticker: Stock symbol
            expiration: Expiration date string (YYYY-MM-DD), uses nearest if None
            
        Returns:
            Dict with 'calls' and 'puts' DataFrames
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expirations
            expirations = stock.options
            if not expirations:
                logger.warning(f"No options available for {ticker}")
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
            
            # Use specified or nearest expiration
            if expiration is None or expiration not in expirations:
                expiration = expirations[0]  # Nearest expiration
            
            # Get the options chain
            chain = stock.option_chain(expiration)
            
            logger.info(f"Fetched options chain for {ticker} exp={expiration}")
            
            return {
                "calls": chain.calls,
                "puts": chain.puts,
                "expiration": expiration,
            }
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {ticker}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "error": str(e)}
    
    def get_put_call_ratio(
        self,
        ticker: str,
        expiration: Optional[str] = None,
        method: str = "volume"
    ) -> Dict[str, Any]:
        """
        Calculate put/call ratio for a ticker.
        
        Args:
            ticker: Stock symbol
            expiration: Expiration date, uses nearest if None
            method: Calculation method - "volume" or "open_interest"
            
        Returns:
            Dict with put/call ratio data:
                - ratio: Put/Call ratio
                - signal: True if ratio < 0.7 (bullish)
                - put_volume: Total put volume
                - call_volume: Total call volume
        """
        try:
            chain = self.get_options_chain(ticker, expiration)
            
            if chain["calls"].empty or chain["puts"].empty:
                return {
                    "ticker": ticker,
                    "ratio": None,
                    "signal": None,
                    "error": "No options data available"
                }
            
            calls = chain["calls"]
            puts = chain["puts"]
            
            if method == "volume":
                call_total = calls["volume"].sum() if "volume" in calls.columns else 0
                put_total = puts["volume"].sum() if "volume" in puts.columns else 0
            else:  # open_interest
                call_total = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
                put_total = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            
            # Handle NaN values
            call_total = 0 if pd.isna(call_total) else call_total
            put_total = 0 if pd.isna(put_total) else put_total
            
            # Calculate ratio
            if call_total > 0:
                ratio = put_total / call_total
            else:
                ratio = None
            
            result = {
                "ticker": ticker,
                "expiration": chain.get("expiration"),
                "method": method,
                "ratio": ratio,
                "put_total": put_total,
                "call_total": call_total,
                "signal": ratio < 0.7 if ratio is not None else None,  # Bullish if < 0.7
                "sentiment": self._interpret_ratio(ratio) if ratio else None,
            }
            
            logger.info(f"Put/Call ratio for {ticker}: {ratio:.2f}" if ratio else f"No P/C ratio for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating put/call ratio for {ticker}: {e}")
            return {"ticker": ticker, "ratio": None, "error": str(e)}
    
    def _interpret_ratio(self, ratio: float) -> str:
        """
        Interpret put/call ratio sentiment.
        
        Args:
            ratio: Put/call ratio value
            
        Returns:
            Sentiment string
        """
        if ratio < 0.5:
            return "Very Bullish"
        elif ratio < 0.7:
            return "Bullish"
        elif ratio < 1.0:
            return "Neutral"
        elif ratio < 1.3:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def get_unusual_activity(
        self,
        ticker: str,
        volume_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect unusual options activity.
        
        Args:
            ticker: Stock symbol
            volume_threshold: Volume multiplier to consider "unusual"
            
        Returns:
            Dict with unusual activity data
        """
        try:
            chain = self.get_options_chain(ticker)
            
            if chain["calls"].empty and chain["puts"].empty:
                return {"ticker": ticker, "unusual_activity": []}
            
            unusual = []
            
            for option_type, df in [("call", chain["calls"]), ("put", chain["puts"])]:
                if df.empty:
                    continue
                    
                # Find high volume contracts relative to open interest
                if "volume" in df.columns and "openInterest" in df.columns:
                    df = df.copy()
                    df["vol_oi_ratio"] = df["volume"] / df["openInterest"].replace(0, 1)
                    
                    high_activity = df[df["vol_oi_ratio"] > volume_threshold]
                    
                    for _, row in high_activity.iterrows():
                        unusual.append({
                            "type": option_type,
                            "strike": row.get("strike"),
                            "volume": row.get("volume"),
                            "open_interest": row.get("openInterest"),
                            "vol_oi_ratio": row.get("vol_oi_ratio"),
                            "implied_volatility": row.get("impliedVolatility"),
                        })
            
            result = {
                "ticker": ticker,
                "expiration": chain.get("expiration"),
                "unusual_activity": unusual,
                "unusual_count": len(unusual),
            }
            
            logger.info(f"Found {len(unusual)} unusual options for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting unusual activity for {ticker}: {e}")
            return {"ticker": ticker, "unusual_activity": [], "error": str(e)}


# Singleton instance
options_fetcher = OptionsFetcher()
