"""
Take Profit / Stop Loss Calculator

Calculates optimal take-profit and stop-loss targets using ATR-based volatility,
support/resistance levels, and regime-aware multipliers.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TPSLCalculator:
    """
    Calculate take-profit and stop-loss targets for trading signals.
    
    Uses a combination of technical factors:
    - Support/resistance levels for logical exits
    - ATR-based volatility for dynamic stops
    - Regime-aware multipliers for adaptive risk management
    
    Example:
        >>> calculator = TPSLCalculator()
        >>> tp, sl = calculator.calculate(
        ...     entry_price=150.0,
        ...     direction="long",
        ...     df=price_data,
        ...     sr_levels={"nearest_resistance": 155.0, "nearest_support": 145.0},
        ...     sl_multiplier=1.5
        ... )
    """
    
    def __init__(
        self,
        atr_length: int = 14,
        default_tp_atr_multiple: float = 2.0,
        default_sl_atr_multiple: float = 1.5
    ):
        """
        Initialize TP/SL calculator.
        
        Args:
            atr_length: ATR calculation period
            default_tp_atr_multiple: ATR multiple for take-profit
            default_sl_atr_multiple: Default ATR multiple for stop-loss
        """
        self.atr_length = atr_length
        self.default_tp_atr_multiple = default_tp_atr_multiple
        self.default_sl_atr_multiple = default_sl_atr_multiple
    
    def calculate(
        self,
        entry_price: float,
        direction: str,
        df: pd.DataFrame,
        sr_levels: Optional[Dict] = None,
        sl_multiplier: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate take-profit and stop-loss targets.
        
        Args:
            entry_price: Entry price for the trade
            direction: Trade direction ("long" or "short")
            df: Price data with OHLC columns
            sr_levels: Support/resistance levels dict
            sl_multiplier: Stop-loss ATR multiplier (overrides default)
            
        Returns:
            Tuple of (take_profit, stop_loss) prices
        """
        try:
            if sr_levels is None:
                sr_levels = {}
            
            if sl_multiplier is None:
                sl_multiplier = self.default_sl_atr_multiple
            
            # Calculate ATR for volatility-based targets
            atr = self._calculate_atr(df)
            
            if direction.lower() in ["long", "buy"]:
                tp, sl = self._calculate_long_targets(
                    entry_price, atr, sr_levels, sl_multiplier
                )
            elif direction.lower() in ["short", "sell"]:
                tp, sl = self._calculate_short_targets(
                    entry_price, atr, sr_levels, sl_multiplier
                )
            else:
                logger.warning(f"Unknown direction '{direction}', returning None targets")
                return None, None
            
            return tp, sl
        
        except Exception as e:
            logger.error(f"TP/SL calculation failed: {e}")
            return None, None
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """
        Calculate Average True Range for volatility measurement.
        
        Args:
            df: Price data with High, Low, Close
            
        Returns:
            ATR value
        """
        try:
            atr_series = ta.atr(
                df["High"],
                df["Low"],
                df["Close"],
                length=self.atr_length
            )
            
            if atr_series is not None and not atr_series.empty:
                atr = atr_series.iloc[-1]
                if not np.isnan(atr):
                    return atr
            
            # Fallback: simple high-low range average
            logger.warning("ATR calculation failed, using fallback range average")
            atr = (df["High"] - df["Low"]).rolling(window=self.atr_length).mean().iloc[-1]
            return atr if not np.isnan(atr) else 0.0
        
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0
    
    def _calculate_long_targets(
        self,
        entry_price: float,
        atr: float,
        sr_levels: Dict,
        sl_multiplier: float
    ) -> Tuple[float, float]:
        """
        Calculate targets for long positions.
        
        Take-profit: Resistance level or entry + 2*ATR
        Stop-loss: Support level or entry - sl_multiplier*ATR
        """
        # Take profit at resistance or ATR-based target
        resistance = sr_levels.get("nearest_resistance")
        if resistance and resistance > entry_price:
            take_profit = resistance
        else:
            take_profit = entry_price + (atr * self.default_tp_atr_multiple)
        
        # Stop loss at support or ATR-based stop
        support = sr_levels.get("nearest_support")
        if support and support < entry_price:
            stop_loss = support
        else:
            stop_loss = entry_price - (atr * sl_multiplier)
        
        # Ensure TP > entry > SL
        take_profit = max(take_profit, entry_price * 1.01)  # At least 1% profit
        stop_loss = min(stop_loss, entry_price * 0.99)  # At least 1% stop
        
        return take_profit, stop_loss
    
    def _calculate_short_targets(
        self,
        entry_price: float,
        atr: float,
        sr_levels: Dict,
        sl_multiplier: float
    ) -> Tuple[float, float]:
        """
        Calculate targets for short positions.
        
        Take-profit: Support level or entry - 2*ATR
        Stop-loss: Resistance level or entry + sl_multiplier*ATR
        """
        # Take profit at support or ATR-based target
        support = sr_levels.get("nearest_support")
        if support and support < entry_price:
            take_profit = support
        else:
            take_profit = entry_price - (atr * self.default_tp_atr_multiple)
        
        # Stop loss at resistance or ATR-based stop
        resistance = sr_levels.get("nearest_resistance")
        if resistance and resistance > entry_price:
            stop_loss = resistance
        else:
            stop_loss = entry_price + (atr * sl_multiplier)
        
        # Ensure SL > entry > TP
        take_profit = min(take_profit, entry_price * 0.99)  # At least 1% profit
        stop_loss = max(stop_loss, entry_price * 1.01)  # At least 1% stop
        
        return take_profit, stop_loss
    
    def calculate_risk_reward(
        self,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float]
    ) -> Optional[float]:
        """
        Calculate risk/reward ratio for the trade.
        
        Args:
            entry_price: Entry price
            take_profit: Take-profit target
            stop_loss: Stop-loss target
            
        Returns:
            Risk/reward ratio (reward / risk) or None if invalid
        """
        if take_profit is None or stop_loss is None:
            return None
        
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk == 0:
                return None
            
            return reward / risk
        except Exception as e:
            logger.error(f"Risk/reward calculation failed: {e}")
            return None


# Singleton instance
tp_sl_calculator = TPSLCalculator()
