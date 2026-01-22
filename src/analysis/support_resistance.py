"""
TradeSense Support and Resistance Detection

Identifies key support and resistance levels using various methods.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupportResistance:
    """
    Support and Resistance level detector.
    
    Uses pivot points, local minima/maxima, and volume profile
    to identify key price levels.
    
    Example:
        >>> sr = SupportResistance()
        >>> levels = sr.find_levels(price_df)
    """
    
    def __init__(
        self,
        lookback: int = 20,
        touch_tolerance: float = 0.02
    ):
        """
        Initialize S/R detector.
        
        Args:
            lookback: How many bars to look back
            touch_tolerance: % tolerance for level touches
        """
        self.lookback = lookback
        self.touch_tolerance = touch_tolerance
    
    def find_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate traditional pivot points.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with pivot point levels:
                - pivot: Central pivot
                - r1, r2, r3: Resistance levels
                - s1, s2, s3: Support levels
        """
        # Use previous day's data
        high = df["High"].iloc[-2] if len(df) > 1 else df["High"].iloc[-1]
        low = df["Low"].iloc[-2] if len(df) > 1 else df["Low"].iloc[-1]
        close = df["Close"].iloc[-2] if len(df) > 1 else df["Close"].iloc[-1]
        
        # Standard pivot point calculation
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            "pivot": float(pivot),
            "r1": float(r1),
            "r2": float(r2),
            "r3": float(r3),
            "s1": float(s1),
            "s2": float(s2),
            "s3": float(s3),
        }
    
    def find_local_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Find local support and resistance from price action.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with 'support' and 'resistance' level lists
        """
        lookback = min(self.lookback, len(df))
        recent = df.tail(lookback)
        
        supports = []
        resistances = []
        
        # Find local minima (support) and maxima (resistance)
        for i in range(2, len(recent) - 2):
            # Local minima
            if (recent["Low"].iloc[i] <= recent["Low"].iloc[i-1] and
                recent["Low"].iloc[i] <= recent["Low"].iloc[i-2] and
                recent["Low"].iloc[i] <= recent["Low"].iloc[i+1] and
                recent["Low"].iloc[i] <= recent["Low"].iloc[i+2]):
                supports.append(float(recent["Low"].iloc[i]))
            
            # Local maxima
            if (recent["High"].iloc[i] >= recent["High"].iloc[i-1] and
                recent["High"].iloc[i] >= recent["High"].iloc[i-2] and
                recent["High"].iloc[i] >= recent["High"].iloc[i+1] and
                recent["High"].iloc[i] >= recent["High"].iloc[i+2]):
                resistances.append(float(recent["High"].iloc[i]))
        
        # Add recent swing points
        if len(recent) > 0:
            supports.append(float(recent["Low"].min()))
            resistances.append(float(recent["High"].max()))
        
        # Remove duplicates within tolerance
        supports = self._merge_nearby_levels(supports)
        resistances = self._merge_nearby_levels(resistances)
        
        return {
            "support": sorted(supports),
            "resistance": sorted(resistances, reverse=True),
        }
    
    def _merge_nearby_levels(self, levels: List[float]) -> List[float]:
        """Merge levels that are within tolerance of each other."""
        if not levels:
            return []
        
        levels = sorted(levels)
        merged = [levels[0]]
        
        for level in levels[1:]:
            # Check if within tolerance of the last merged level
            if abs(level - merged[-1]) / merged[-1] <= self.touch_tolerance:
                # Average the levels
                merged[-1] = (merged[-1] + level) / 2
            else:
                merged.append(level)
        
        return merged
    
    def find_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Find all support and resistance levels.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with all level information
        """
        current_price = df["Close"].iloc[-1]
        
        # Get pivot points
        pivots = self.find_pivot_points(df)
        
        # Get local levels
        local = self.find_local_levels(df)
        
        # Find nearest levels
        all_supports = local["support"] + [pivots["s1"], pivots["s2"], pivots["s3"]]
        all_resistances = local["resistance"] + [pivots["r1"], pivots["r2"], pivots["r3"]]
        
        # Below current price = support
        supports = [s for s in all_supports if s < current_price]
        # Above current price = resistance
        resistances = [r for r in all_resistances if r > current_price]
        
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        return {
            "current_price": float(current_price),
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "pivot_points": pivots,
            "local_supports": local["support"],
            "local_resistances": local["resistance"],
            "support_distance_percent": (
                (current_price - nearest_support) / current_price * 100
                if nearest_support else None
            ),
            "resistance_distance_percent": (
                (nearest_resistance - current_price) / current_price * 100
                if nearest_resistance else None
            ),
        }
    
    def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading signal based on S/R levels.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with signal and level info
        """
        levels = self.find_levels(df)
        current = levels["current_price"]
        support = levels["nearest_support"]
        resistance = levels["nearest_resistance"]
        
        signal = None
        condition = "Neutral"
        
        # Near support = potential buy
        if support and levels["support_distance_percent"] <= 2:
            signal = True
            condition = f"Near support ({support:.2f})"
        # Near resistance = potential sell
        elif resistance and levels["resistance_distance_percent"] <= 2:
            signal = False
            condition = f"Near resistance ({resistance:.2f})"
        
        return {
            "signal": signal,
            "condition": condition,
            "nearest_support": support,
            "nearest_resistance": resistance,
            "current_price": current,
        }


# Singleton instance
support_resistance = SupportResistance()
