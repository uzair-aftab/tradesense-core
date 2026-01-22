"""
TradeSense Backtesting Engine

Powered by vectorbt for fast, vectorized backtesting.
Validates trading strategies against historical data.
"""

from typing import Dict, List, Optional, Any
import vectorbt as vbt
import pandas as pd
import numpy as np
import pandas_ta as ta

from src.utils.logger import get_logger
from src.data import stock_fetcher, crypto_fetcher
from src.config import config

logger = get_logger(__name__)


class BacktestEngine:
    """
    Backtesting engine using vectorbt.
    
    Example:
        >>> engine = BacktestEngine()
        >>> stats = engine.run("AAPL", "1y")
    """
    
    def __init__(self):
        """Initialize backtest engine."""
        self.fees = 0.001  # 0.1% fee
        self.slippage = 0.001  # 0.1% slippage
    
    def fetch_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        # Use stock fetcher (vectorbt expects OHLCV)
        df = stock_fetcher.get_historical(ticker, period=period, interval=interval)
        return df
    
    def run_technical_strategy(
        self,
        ticker: str,
        period: str = "1y",
        rsi_threshold: int = 30,
        ma_fast: int = 12,
        ma_slow: int = 26
    ) -> Dict[str, Any]:
        """
        Run a combined technical strategy backtest.
        
        Strategy:
        - Buy when RSI < threshold AND MACD bullish crossover
        - Sell when RSI > 70 OR MACD bearish crossover
        
        Args:
            ticker: Asset symbol
            period: Backtest period
            rsi_threshold: RSI buy threshold
            ma_fast: MACD fast period
            ma_slow: MACD slow period
            
        Returns:
            Dict with performance metrics
        """
        logger.info(f"Running backtest for {ticker} over {period}")
        
        # 1. Get Data
        df = self.fetch_data(ticker, period)
        if df.empty:
            return {"error": "No data found"}
        
        close = df["Close"]
        
        # 2. Calculate Indicators (Vectorized)
        rsi = vbt.RSI.run(close, window=14)
        macd = vbt.MACD.run(close, fast_window=ma_fast, slow_window=ma_slow)
        
        # 3. Define Entry/Exit Logic
        # Buy: RSI oversold AND MACD histogram positive (momentum up)
        entries = (rsi.rsi < rsi_threshold) & (macd.hist > 0)
        
        # Sell: RSI overbought OR MACD histogram negative (momentum down)
        exits = (rsi.rsi > 70) | (macd.hist < 0)
        
        # Clean signals (remove repeated entries)
        clean_entries, clean_exits = entries.vbt.signals.clean(exits)
        
        # 4. Run Portfolio
        portfolio = vbt.Portfolio.from_signals(
            close,
            clean_entries,
            clean_exits,
            fees=self.fees,
            slippage=self.slippage,
            init_cash=10000,
            freq="1D"
        )
        
        # 5. Calculate Stats
        stats = portfolio.stats()
        
        return {
            "ticker": ticker,
            "total_return": stats["Total Return [%]"],
            "benchmark_return": stats["Benchmark Return [%]"],
            "win_rate": stats["Win Rate [%]"],
            "max_drawdown": stats["Max Drawdown [%]"],
            "sharpe_ratio": stats["Sharpe Ratio"],
            "trades": stats["Total Trades"],
            "portfolio": portfolio  # Return portfolio object for plotting
        }
    
    def optimize_rsi(
        self,
        ticker: str,
        period: str = "1y",
        thresholds: List[int] = None
    ) -> pd.DataFrame:
        """
        Optimize RSI threshold.
        
        Args:
            ticker: Asset symbol
            period: Time period
            thresholds: List of thresholds to test
            
        Returns:
            DataFrame with results
        """
        if thresholds is None:
            thresholds = range(10, 45, 5)
            
        df = self.fetch_data(ticker, period)
        close = df["Close"]
        
        # Vectorized optimization
        rsi = vbt.RSI.run(close, window=14)
        
        # Test all thresholds at once
        entries = rsi.rsi_below(thresholds)
        exits = rsi.rsi_above(70)
        
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            fees=self.fees,
            freq="1D"
        )
        
        return portfolio.total_return() * 100


# Singleton instance
backtest_engine = BacktestEngine()
