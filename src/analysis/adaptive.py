"""
TradeSense Adaptive Regime Detection

Classifies market state (BULL, BEAR, SIDEWAYS, CRISIS) and 
provides dynamic trading thresholds.
"""

from typing import Dict, Any, Optional
from enum import Enum
import pandas as pd
import pandas_ta as ta
import numpy as np

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"         # Strong uptrend, buy dips
    BEAR = "BEAR"         # Strong downtrend, short rallies
    SIDEWAYS = "SIDEWAYS" # Range-bound, mean reversion
    CRISIS = "CRISIS"     # Extreme volatility, defensive
    UNKNOWN = "UNKNOWN"


class AdaptiveThresholds:
    """
    Manages dynamic thresholds based on Market Regime.
    """
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self.base_rsi_oversold = config.RSI_OVERSOLD
        self.base_rsi_overbought = config.RSI_OVERBOUGHT
        
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regime using Trend, Momentum, and Volatility.
        
        Logic:
        - BULL: SMA50 > SMA200, or ADX > 25 + RSI > 50 + Price > SMA50
        - BEAR: SMA50 < SMA200, or ADX > 25 + RSI < 50 + Price < SMA50
        - CRISIS: ATR spiked > 2x average, or recent crash
        - SIDEWAYS: Low ADX (< 20) or price chopping around SMA
        """
        if len(df) < 200:
            return {
                "regime": MarketRegime.UNKNOWN.value, 
                "volatility": "NORMAL",
                "details": "Insufficient data (need 200 bars)"
            }
            
        # --- 1. Calculate Indicators ---
        close = df["Close"]
        
        # Trend
        sma50 = ta.sma(close, length=50)
        sma200 = ta.sma(close, length=200)
        
        # Momentum/Strength
        adx = ta.adx(df["High"], df["Low"], close, length=14)
        if adx is not None and not adx.empty:
            adx_val = adx["ADX_14"].iloc[-1]
        else:
            adx_val = 0
            
        rsi = ta.rsi(close, length=14).iloc[-1]
        
        # Volatility
        atr = ta.atr(df["High"], df["Low"], close, length=14)
        current_atr = atr.iloc[-1]
        atr_pct = (current_atr / close.iloc[-1]) * 100
        atr_avg = atr.rolling(30).mean().iloc[-1]
        
        # Volatility Regime
        if atr_pct > 3.0 or current_atr > (atr_avg * 1.5):
            vol_regime = "HIGH"
        elif atr_pct < 1.0:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"
            
        # --- 2. Classify Market Regime ---
        current_price = close.iloc[-1]
        last_sma50 = sma50.iloc[-1]
        last_sma200 = sma200.iloc[-1]
        
        regime = MarketRegime.SIDEWAYS # Default
        
        # Crisis Check First
        if vol_regime == "HIGH" and current_price < last_sma50:
            regime = MarketRegime.CRISIS
            
        # Trend Checks
        elif adx_val > 25:
            # Strong Trend
            if current_price > last_sma50 and rsi > 50:
                regime = MarketRegime.BULL
            elif current_price < last_sma50 and rsi < 50:
                regime = MarketRegime.BEAR
        
        # Moving Average Flows
        elif last_sma50 > last_sma200 and current_price > last_sma50:
            regime = MarketRegime.BULL
        elif last_sma50 < last_sma200 and current_price < last_sma50:
            regime = MarketRegime.BEAR
            
        return {
            "regime": regime.value,
            "volatility": vol_regime,
            "atr_percent": round(atr_pct, 2),
            "adx": round(adx_val, 2),
            "details": f"Price vs SMA50: {'Above' if current_price > last_sma50 else 'Below'}"
        }

    def get_rsi_thresholds(self, regime_str: str) -> Dict[str, int]:
        """Adjust RSI limits based on regime."""
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            regime = MarketRegime.UNKNOWN

        if regime == MarketRegime.BULL:
            # Bull market: RSI stays high, oversold is rare/shallow
            return {"oversold": 40, "overbought": 80}
            
        elif regime == MarketRegime.BEAR:
            # Bear market: RSI stays low, overbought is rare/shallow
            return {"oversold": 20, "overbought": 60}
            
        elif regime == MarketRegime.CRISIS:
            # volatile - wide bands
            return {"oversold": 15, "overbought": 85}
            
        else: # SIDEWAYS / UNKNOWN
            return {"oversold": 30, "overbought": 70}

    def get_stop_loss_multiplier(self, regime_str: str) -> float:
        """Adjust Stop Loss ATR multiplier."""
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            regime = MarketRegime.UNKNOWN

        if regime == MarketRegime.BULL:
            return 2.0  # Standard
        elif regime == MarketRegime.BEAR:
            return 1.5  # Tighten stops in downtrend (if long) or standard if short
        elif regime == MarketRegime.CRISIS:
            return 3.0  # Wide stops for volatility (or cash)
        elif regime == MarketRegime.SIDEWAYS:
            return 1.5  # Tight stops in range
        return 2.0

# Singleton
adaptive_thresholds = AdaptiveThresholds()
