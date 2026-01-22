"""
TradeSense Adaptive Thresholds Engine

Adjusts technical indicators and signal parameters based on 
market volatility and asset class.
"""

from typing import Dict, Any, Optional
import pandas as pd
import pandas_ta as ta
import numpy as np

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)

class AdaptiveThresholds:
    """
    Manages dynamic thresholds for technical analysis.
    
    Uses ATR (Average True Range) to detect volatility regimes
    and adjust parameters like RSI limits and Stop Loss distances.
    """
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        
        # Base config values
        self.base_rsi_oversold = config.RSI_OVERSOLD
        self.base_rsi_overbought = config.RSI_OVERBOUGHT
        
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect volatility regime based on ATR.
        
        Returns:
            Dict with:
                - regime: "LOW", "NORMAL", "HIGH"
                - atr: Current ATR value
                - atr_percent: ATR as percentage of price
        """
        if len(df) < self.atr_period + 1:
            return {"regime": "NORMAL", "atr": 0.0, "atr_percent": 0.0}
            
        # Calculate ATR
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_period)
        current_atr = atr_series.iloc[-1]
        current_price = df["Close"].iloc[-1]
        
        if pd.isna(current_atr):
            return {"regime": "NORMAL", "atr": 0.0, "atr_percent": 0.0}
            
        atr_percent = (current_atr / current_price) * 100
        
        # Define regimes based on ATR %
        # Low: < 1% (Stable, large caps)
        # Normal: 1-3% (Typical stocks)
        # High: > 3% (Crypto, volatile stocks)
        
        if atr_percent < 1.0:
            regime = "LOW"
        elif atr_percent > 3.0:
            regime = "HIGH"
        else:
            regime = "NORMAL"
            
        return {
            "regime": regime,
            "atr": float(current_atr),
            "atr_percent": float(atr_percent)
        }
        
    def get_rsi_thresholds(self, regime: str) -> Dict[str, int]:
        """
        Get adjusted RSI thresholds based on volatility.
        
        High volatility -> Wider bands (e.g. 20/80) to avoid false signals.
        Low volatility -> Tighter bands (e.g. 35/65) to catch smaller moves.
        """
        if regime == "HIGH":
            return {"oversold": 20, "overbought": 80}
        elif regime == "LOW":
            return {"oversold": 35, "overbought": 65}
        else:
            return {
                "oversold": self.base_rsi_oversold,
                "overbought": self.base_rsi_overbought
            }

    def get_stop_loss_multiplier(self, regime: str) -> float:
        """
        Get ATR multiplier for stop loss.
        
        High volatility -> Wider stops (e.g. 2.5x ATR).
        Low volatility -> Tighter stops (e.g. 1.5x ATR).
        """
        if regime == "HIGH":
            return 2.5
        elif regime == "LOW":
            return 1.5
        else:
            return 2.0

# Singleton instance
adaptive_thresholds = AdaptiveThresholds()
