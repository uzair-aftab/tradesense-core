"""
TradeSense Technical Analysis Engine

Provides technical indicators using pandas-ta.
Includes RSI, MACD, Bollinger Bands, Volume analysis.
"""

from typing import Dict, Optional, Tuple, Any
import pandas as pd
import pandas_ta as ta
import numpy as np

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class RSIIndicator:
    """
    Relative Strength Index (RSI) indicator.
    
    Generates buy signals when RSI is oversold.
    Default oversold threshold is 20 (configurable).
    
    Example:
        >>> rsi = RSIIndicator()
        >>> df = rsi.calculate(price_df)
        >>> signal = rsi.get_signal(df)
    """
    
    def __init__(
        self,
        period: int = 14,
        oversold: int = None,
        overbought: int = None
    ):
        """
        Initialize RSI indicator.
        
        Args:
            period: RSI calculation period
            oversold: Oversold threshold (default from config)
            overbought: Overbought threshold (default from config)
        """
        self.period = period
        self.oversold = oversold or config.RSI_OVERSOLD
        self.overbought = overbought or config.RSI_OVERBOUGHT
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for price data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            DataFrame with 'RSI' column added
        """
        df = df.copy()
        df["RSI"] = ta.rsi(df["Close"], length=self.period)
        return df
    
    def get_signal(self, df: pd.DataFrame, oversold: int = None, overbought: int = None) -> Dict[str, Any]:
        """
        Get RSI signal for trading with optional dynamic thresholds.
        
        Args:
            df: DataFrame with RSI column
            oversold: Override oversold threshold
            overbought: Override overbought threshold
            
        Returns:
            Dict with signal details
        """
        if "RSI" not in df.columns:
            df = self.calculate(df)
        
        current_rsi = df["RSI"].iloc[-1]
        
        if pd.isna(current_rsi):
            return {"signal": None, "rsi": None, "condition": "Insufficient data"}
            
        # Use provided thresholds or defaults
        thresh_os = oversold if oversold is not None else self.oversold
        thresh_ob = overbought if overbought is not None else self.overbought
        
        if current_rsi <= thresh_os:
            return {
                "signal": True,
                "rsi": float(current_rsi),
                "condition": f"Oversold (RSI={current_rsi:.1f} <= {thresh_os})"
            }
        elif current_rsi >= thresh_ob:
            return {
                "signal": False,
                "rsi": float(current_rsi),
                "condition": f"Overbought (RSI={current_rsi:.1f} >= {thresh_ob})"
            }
        else:
            return {
                "signal": None,
                "rsi": float(current_rsi),
                "condition": f"Neutral (RSI={current_rsi:.1f})"
            }


class MACDIndicator:
    """
    Moving Average Convergence Divergence (MACD) indicator.
    
    Generates buy signals on bullish crossovers.
    
    Example:
        >>> macd = MACDIndicator()
        >>> df = macd.calculate(price_df)
        >>> signal = macd.get_signal(df)
    """
    
    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9
    ):
        """
        Initialize MACD indicator.
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal_period: Signal line period
        """
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD for price data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            DataFrame with MACD columns added
        """
        df = df.copy()
        macd = ta.macd(df["Close"], fast=self.fast, slow=self.slow, signal=self.signal_period)
        df = pd.concat([df, macd], axis=1)
        return df
    
    def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get MACD signal for trading.
        
        Args:
            df: DataFrame with MACD columns
            
        Returns:
            Dict with:
                - signal: True (bullish crossover), False (bearish), None (hold)
                - macd_value: MACD line value
                - histogram: MACD histogram value
                - condition: Description
        """
        macd_col = f"MACD_{self.fast}_{self.slow}_{self.signal_period}"
        signal_col = f"MACDs_{self.fast}_{self.slow}_{self.signal_period}"
        hist_col = f"MACDh_{self.fast}_{self.slow}_{self.signal_period}"
        
        if macd_col not in df.columns:
            df = self.calculate(df)
        
        if len(df) < 2:
            return {"signal": None, "condition": "Insufficient data"}
        
        current_macd = df[macd_col].iloc[-1]
        current_signal = df[signal_col].iloc[-1]
        current_hist = df[hist_col].iloc[-1]
        prev_hist = df[hist_col].iloc[-2]
        
        if pd.isna(current_macd) or pd.isna(prev_hist):
            return {"signal": None, "condition": "Insufficient data"}
        
        # Bullish crossover: histogram crosses from negative to positive
        if prev_hist < 0 and current_hist >= 0:
            return {
                "signal": True,
                "macd_value": float(current_macd),
                "histogram": float(current_hist),
                "condition": "Bullish crossover"
            }
        # Bearish crossover
        elif prev_hist > 0 and current_hist <= 0:
            return {
                "signal": False,
                "macd_value": float(current_macd),
                "histogram": float(current_hist),
                "condition": "Bearish crossover"
            }
        # Momentum building
        elif current_hist > prev_hist and current_hist > 0:
            return {
                "signal": True,
                "macd_value": float(current_macd),
                "histogram": float(current_hist),
                "condition": "Bullish momentum"
            }
        elif current_hist < prev_hist and current_hist < 0:
            return {
                "signal": False,
                "macd_value": float(current_macd),
                "histogram": float(current_hist),
                "condition": "Bearish momentum"
            }
        else:
            return {
                "signal": None,
                "macd_value": float(current_macd),
                "histogram": float(current_hist),
                "condition": "Neutral"
            }


class BollingerBands:
    """
    Bollinger Bands indicator.
    
    Generates buy signals near the lower band.
    
    Example:
        >>> bb = BollingerBands()
        >>> df = bb.calculate(price_df)
        >>> signal = bb.get_signal(df)
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
        """
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            DataFrame with BB columns added
        """
        df = df.copy()
        bbands = ta.bbands(df["Close"], length=self.period, std=self.std_dev)
        df = pd.concat([df, bbands], axis=1)
        return df
    
    def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get Bollinger Bands signal.
        
        Args:
            df: DataFrame with BB columns
            
        Returns:
            Dict with signal and band values
        """
        # Find BB columns dynamically (pandas-ta naming varies)
        bb_cols = [c for c in df.columns if c.startswith("BB")]
        if not bb_cols:
            df = self.calculate(df)
            bb_cols = [c for c in df.columns if c.startswith("BB")]
        
        # Find the specific columns
        lower_col = next((c for c in bb_cols if c.startswith("BBL")), None)
        mid_col = next((c for c in bb_cols if c.startswith("BBM")), None)
        upper_col = next((c for c in bb_cols if c.startswith("BBU")), None)
        
        if not all([lower_col, mid_col, upper_col]):
            return {"signal": None, "condition": "BB columns not found"}
        
        current_price = df["Close"].iloc[-1]
        lower_band = df[lower_col].iloc[-1]
        mid_band = df[mid_col].iloc[-1]
        upper_band = df[upper_col].iloc[-1]
        
        if pd.isna(lower_band):
            return {"signal": None, "condition": "Insufficient data"}
        
        # Calculate %B (position within bands)
        bandwidth = upper_band - lower_band
        if bandwidth > 0:
            percent_b = (current_price - lower_band) / bandwidth
        else:
            percent_b = 0.5
        
        # Near lower band = potential buy
        if percent_b <= 0.1:
            return {
                "signal": True,
                "percent_b": float(percent_b),
                "lower_band": float(lower_band),
                "upper_band": float(upper_band),
                "condition": f"Near lower band (%B={percent_b:.2f})"
            }
        # Near upper band = potential sell
        elif percent_b >= 0.9:
            return {
                "signal": False,
                "percent_b": float(percent_b),
                "lower_band": float(lower_band),
                "upper_band": float(upper_band),
                "condition": f"Near upper band (%B={percent_b:.2f})"
            }
        else:
            return {
                "signal": None,
                "percent_b": float(percent_b),
                "lower_band": float(lower_band),
                "upper_band": float(upper_band),
                "condition": f"Within bands (%B={percent_b:.2f})"
            }


class VolumeAnalyzer:
    """
    Volume analysis indicator.
    
    Compares current volume to moving average.
    
    Example:
        >>> vol = VolumeAnalyzer()
        >>> df = vol.calculate(price_df)
        >>> signal = vol.get_signal(df)
    """
    
    def __init__(self, period: int = 20, threshold: float = None):
        """
        Initialize volume analyzer.
        
        Args:
            period: Volume MA period
            threshold: Volume multiplier threshold (default from config)
        """
        self.period = period
        self.threshold = threshold or config.VOLUME_MULTIPLIER
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume metrics.
        
        Args:
            df: DataFrame with 'Volume' column
            
        Returns:
            DataFrame with volume metrics added
        """
        df = df.copy()
        df["Volume_MA"] = ta.sma(df["Volume"], length=self.period)
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
        return df
    
    def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get volume signal.
        
        Args:
            df: DataFrame with volume metrics
            
        Returns:
            Dict with:
                - high_volume: True if above threshold
                - volume_ratio: Current volume / MA
                - confirms_trend: True if volume confirms price direction
        """
        if "Volume_Ratio" not in df.columns:
            df = self.calculate(df)
        
        current_ratio = df["Volume_Ratio"].iloc[-1]
        current_volume = df["Volume"].iloc[-1]
        vol_ma = df["Volume_MA"].iloc[-1]
        
        if pd.isna(current_ratio):
            return {"high_volume": False, "volume_ratio": None}
        
        # Check price direction
        if len(df) >= 2:
            price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
            price_up = price_change > 0
        else:
            price_up = None
        
        high_volume = current_ratio >= (1 + self.threshold)
        
        return {
            "high_volume": high_volume,
            "volume_ratio": float(current_ratio),
            "current_volume": float(current_volume),
            "volume_ma": float(vol_ma),
            "price_direction_up": price_up,
            "confirms_trend": high_volume,  # High volume confirms the current move
            "condition": f"Volume {current_ratio:.1f}x average"
        }


class TechnicalSignalGenerator:
    """
    Combines all technical indicators into unified signals.
    
    Example:
        >>> generator = TechnicalSignalGenerator()
        >>> signals = generator.analyze(price_df)
    """
    
    def __init__(self):
        """Initialize all technical indicators."""
        self.rsi = RSIIndicator()
        self.macd = MACDIndicator()
        self.bb = BollingerBands()
        self.volume = VolumeAnalyzer()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators
        """
        df = self.rsi.calculate(df)
        df = self.macd.calculate(df)
        df = self.bb.calculate(df)
        df = self.volume.calculate(df)
        return df
    
    def analyze(self, df: pd.DataFrame, thresholds: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze price data and generate signals.
        
        Args:
            df: DataFrame with OHLCV data
            thresholds: Optional dynamic thresholds (e.g. {'rsi_oversold': 20})
            
        Returns:
            Dict with signals and scores
        """
        if len(df) < 30:
            return {
                "error": "Insufficient data (need at least 30 rows)",
                "composite_signal": None,
                "technical_score": 0
            }
        
        # Calculate all indicators
        df = self.calculate_all(df)
        
        # Get individual signals with optional overrides
        thresholds = thresholds or {}
        rsi_signal = self.rsi.get_signal(
            df, 
            oversold=thresholds.get("rsi_oversold"),
            overbought=thresholds.get("rsi_overbought")
        )
        macd_signal = self.macd.get_signal(df)
        bb_signal = self.bb.get_signal(df)
        volume_signal = self.volume.get_signal(df)
        
        # Calculate composite signal
        signals = [
            rsi_signal.get("signal"),
            macd_signal.get("signal"),
            bb_signal.get("signal"),
        ]
        
        # Count buy/sell signals
        buy_count = sum(1 for s in signals if s is True)
        sell_count = sum(1 for s in signals if s is False)
        
        # Composite signal logic
        if buy_count >= 2:
            composite = True
            score = buy_count / 3.0
        elif sell_count >= 2:
            composite = False
            score = -sell_count / 3.0
        else:
            composite = None
            score = (buy_count - sell_count) / 3.0
        
        # Volume confirmation bonus
        if volume_signal.get("confirms_trend") and composite is not None:
            score = min(1.0, abs(score) + 0.15) * (1 if score >= 0 else -1)
        
        return {
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "bb_signal": bb_signal,
            "volume_signal": volume_signal,
            "composite_signal": composite,
            "technical_score": float((score + 1) / 2),  # Normalize to 0-1
            "buy_signals": buy_count,
            "sell_signals": sell_count,
        }


# Singleton instances
rsi_indicator = RSIIndicator()
macd_indicator = MACDIndicator()
bollinger_bands = BollingerBands()
volume_analyzer = VolumeAnalyzer()
technical_signal_generator = TechnicalSignalGenerator()
