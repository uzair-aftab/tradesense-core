"""
Multi-Timeframe Engine

Supports multiple trading timeframes from scalping to position trading.
Each timeframe has optimized features and model settings.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from src.data import stock_fetcher
from src.ml import FeatureEngineer, signal_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Timeframe(Enum):
    """Trading timeframes with metadata."""
    M5 = ("5m", "5 Minutes", "scalping", 6, "5d")       # 6 periods = 30 min ahead
    M15 = ("15m", "15 Minutes", "day_trading", 4, "5d")  # 4 periods = 1 hour ahead
    H1 = ("1h", "1 Hour", "intraday", 4, "60d")          # 4 periods = 4 hours ahead
    H4 = ("4h", "4 Hours", "swing", 6, "60d")            # 6 periods = 24 hours ahead
    D1 = ("1d", "Daily", "position", 3, "2y")            # 3 periods = 3 days ahead
    
    def __init__(self, interval: str, label: str, use_case: str, horizon: int, period: str):
        self.interval = interval
        self.label = label
        self.use_case = use_case
        self.horizon = horizon
        self.period = period


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: Timeframe
    feature_lookback: int  # Number of periods for feature calculation
    volatility_window: int  # Window for volatility calculations
    profit_target: float  # Target return for this timeframe
    
    @classmethod
    def get_config(cls, tf: Timeframe) -> 'TimeframeConfig':
        """Get default config for a timeframe."""
        configs = {
            Timeframe.M5: cls(tf, 50, 10, 0.002),    # 0.2% target
            Timeframe.M15: cls(tf, 50, 15, 0.003),   # 0.3% target
            Timeframe.H1: cls(tf, 40, 20, 0.005),    # 0.5% target
            Timeframe.H4: cls(tf, 30, 15, 0.008),    # 0.8% target
            Timeframe.D1: cls(tf, 50, 20, 0.01),     # 1.0% target
        }
        return configs.get(tf, configs[Timeframe.D1])


class MultiTimeframeEngine:
    """
    Engine for multi-timeframe analysis and predictions.
    
    Supports:
    - 5m: Scalping (30-min predictions)
    - 15m: Day trading (1-hour predictions)
    - 1h: Intraday (4-hour predictions)
    - 4h: Swing trading (24-hour predictions)
    - 1d: Position trading (3-day predictions)
    
    Example:
        >>> engine = MultiTimeframeEngine()
        >>> signal = engine.get_signal("AAPL", Timeframe.H1)
        >>> signals = engine.get_all_timeframe_signals("AAPL")
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self._data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    def get_data(
        self, 
        ticker: str, 
        timeframe: Timeframe,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for a specific timeframe.
        
        Args:
            ticker: Stock symbol
            timeframe: Target timeframe
            use_cache: Use cached data if available
            
        Returns:
            OHLCV DataFrame for the timeframe
        """
        cache_key = f"{ticker}_{timeframe.interval}"
        
        if use_cache and ticker in self._data_cache:
            if cache_key in self._data_cache[ticker]:
                return self._data_cache[ticker][cache_key]
        
        try:
            if timeframe in [Timeframe.M5, Timeframe.M15]:
                # Intraday data
                df = stock_fetcher.get_intraday(
                    ticker, 
                    interval=timeframe.interval, 
                    period=timeframe.period
                )
            else:
                # Daily/hourly data
                df = stock_fetcher.get_historical(
                    ticker, 
                    period=timeframe.period, 
                    interval=timeframe.interval
                )
            
            # Cache the data
            if ticker not in self._data_cache:
                self._data_cache[ticker] = {}
            self._data_cache[ticker][cache_key] = df
            
            logger.info(f"Fetched {len(df)} rows for {ticker} @ {timeframe.label}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker} @ {timeframe.interval}: {e}")
            return pd.DataFrame()
    
    def get_features(
        self, 
        ticker: str, 
        timeframe: Timeframe
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Get features for a ticker at a specific timeframe.
        
        Args:
            ticker: Stock symbol
            timeframe: Target timeframe
            
        Returns:
            Tuple of (features DataFrame, target Series if available)
        """
        df = self.get_data(ticker, timeframe)
        
        if df.empty:
            return pd.DataFrame(), None
        
        config = TimeframeConfig.get_config(timeframe)
        
        # Calculate features with timeframe-specific horizon
        features_df = self.feature_engineer.calculate_features(
            df,
            include_targets=False,
            prediction_horizon=timeframe.horizon,
            profit_target=config.profit_target
        )
        
        return features_df, None
    
    def get_signal(
        self, 
        ticker: str, 
        timeframe: Timeframe
    ) -> Dict:
        """
        Get ML signal for a ticker at a specific timeframe.
        
        Args:
            ticker: Stock symbol
            timeframe: Target timeframe
            
        Returns:
            Signal dict with prediction and confidence
        """
        features_df, _ = self.get_features(ticker, timeframe)
        
        if features_df.empty or not signal_model.is_trained:
            return {
                "ticker": ticker,
                "timeframe": timeframe.label,
                "signal": "HOLD",
                "confidence": 0.0,
                "error": "No data or model not trained"
            }
        
        try:
            # Get latest row for prediction
            X = features_df[self.feature_engineer.feature_columns].iloc[-1:]
            prob = signal_model.predict_proba(X)[0]
            
            # Determine signal
            if prob > 0.6:
                signal_type = "BUY"
                confidence = prob
            elif prob < 0.4:
                signal_type = "SELL"
                confidence = 1 - prob
            else:
                signal_type = "HOLD"
                confidence = 0.5
            
            # Get current price
            current_price = features_df['Close'].iloc[-1] if 'Close' in features_df.columns else 0
            
            return {
                "ticker": ticker,
                "timeframe": timeframe.label,
                "interval": timeframe.interval,
                "use_case": timeframe.use_case,
                "signal": signal_type,
                "probability": prob,
                "confidence": confidence,
                "horizon": f"{timeframe.horizon} periods",
                "current_price": current_price,
            }
            
        except Exception as e:
            logger.error(f"Signal error for {ticker} @ {timeframe.label}: {e}")
            return {
                "ticker": ticker,
                "timeframe": timeframe.label,
                "signal": "HOLD",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_all_timeframe_signals(self, ticker: str) -> List[Dict]:
        """
        Get signals for all timeframes.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            List of signals for each timeframe
        """
        signals = []
        for tf in Timeframe:
            signal = self.get_signal(ticker, tf)
            signals.append(signal)
        return signals
    
    def get_consensus_signal(self, ticker: str) -> Dict:
        """
        Get weighted consensus across all timeframes.
        
        Higher timeframes get more weight for trend direction.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Consensus signal dict
        """
        signals = self.get_all_timeframe_signals(ticker)
        
        # Weights: Higher timeframes = more weight
        weights = {
            "5m": 0.10,
            "15m": 0.15,
            "1h": 0.20,
            "4h": 0.25,
            "1d": 0.30
        }
        
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        for sig in signals:
            weight = weights.get(sig.get("interval", "1d"), 0.2)
            prob = sig.get("probability", 0.5)
            
            if sig["signal"] == "BUY":
                buy_score += prob * weight
            elif sig["signal"] == "SELL":
                sell_score += (1 - prob) * weight
            
            total_weight += weight
        
        # Normalize
        buy_score /= total_weight
        sell_score /= total_weight
        
        # Determine consensus
        if buy_score > 0.55:
            consensus = "BUY"
            confidence = buy_score
        elif sell_score > 0.55:
            consensus = "SELL"
            confidence = sell_score
        else:
            consensus = "HOLD"
            confidence = 0.5
        
        return {
            "ticker": ticker,
            "consensus": consensus,
            "confidence": confidence,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "timeframe_signals": signals
        }
    
    def clear_cache(self, ticker: Optional[str] = None):
        """Clear data cache."""
        if ticker:
            self._data_cache.pop(ticker, None)
        else:
            self._data_cache.clear()


# Singleton instance
mtf_engine = MultiTimeframeEngine()
