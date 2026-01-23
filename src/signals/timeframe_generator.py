"""
Timeframe-Aware Signal Generator

Generates actionable trade setups optimized for specific timeframes.
Different models, thresholds, and targets per timeframe.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.ml.multi_timeframe import Timeframe, mtf_engine
from src.ml import ensemble_model, FeatureEngineer
from src.analysis import technical_signal_generator, support_resistance, adaptive_thresholds
from src.sentiment import sentiment_aggregator
from src.data.fetcher_fundamentals import fundamentals_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeSetup:
    """
    Complete actionable trade setup with entry/exit levels.
    This is what a trader needs to execute a trade.
    """
    ticker: str
    direction: str  # "LONG", "SHORT", "HOLD"
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float  # 0-1
    timeframe: str
    duration: str  # Expected trade duration
    
    # Component scores
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    ml_score: float = 0.0
    
    # Reasoning
    reasons: List[str] = field(default_factory=list)
    
    # Calculated on demand
    _position_size: Optional[int] = None
    _portfolio_value: float = 10000.0
    _risk_per_trade: float = 0.02
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_price - self.entry_price)
        return reward / risk if risk > 0 else 0
    
    @property
    def position_size(self) -> int:
        """Calculate position size based on 2% risk rule."""
        if self._position_size is not None:
            return self._position_size
        
        risk_amount = self._portfolio_value * self._risk_per_trade
        stop_distance = abs(self.entry_price - self.stop_loss)
        if stop_distance == 0:
            return 0
        return int(risk_amount / stop_distance)
    
    @property
    def risk_amount(self) -> float:
        """Dollar risk for this trade."""
        return abs(self.entry_price - self.stop_loss) * self.position_size
    
    @property
    def reward_amount(self) -> float:
        """Dollar reward for this trade."""
        return abs(self.target_price - self.entry_price) * self.position_size
    
    @property
    def target_pct(self) -> float:
        """Target as percentage from entry."""
        return ((self.target_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def stop_pct(self) -> float:
        """Stop as percentage from entry."""
        return ((self.stop_loss - self.entry_price) / self.entry_price) * 100
    
    @property
    def is_actionable(self) -> bool:
        """Check if this setup is worth trading."""
        return (
            self.confidence >= 0.55 and 
            self.risk_reward_ratio >= 1.5 and 
            self.direction != "HOLD"
        )
    
    def set_portfolio_context(self, portfolio_value: float, risk_pct: float = 0.02):
        """Set portfolio context for position sizing."""
        self._portfolio_value = portfolio_value
        self._risk_per_trade = risk_pct
        self._position_size = None  # Reset to recalculate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry": self.entry_price,
            "target": self.target_price,
            "stop": self.stop_loss,
            "confidence": self.confidence,
            "rr_ratio": self.risk_reward_ratio,
            "position_size": self.position_size,
            "timeframe": self.timeframe,
            "duration": self.duration,
            "actionable": self.is_actionable,
            "reasons": self.reasons,
        }


# Timeframe-specific configuration
TIMEFRAME_CONFIGS = {
    Timeframe.M5: {
        "min_confidence": 0.60,
        "profit_target_pct": 0.003,  # 0.3%
        "stop_loss_pct": 0.0015,     # 0.15%
        "duration": "30 minutes",
        "rsi_period": 7,
        "sentiment_weight": 0.05,   # Less important for scalping
        "technical_weight": 0.55,
        "ml_weight": 0.40,
        "fundamental_weight": 0.0,  # Ignore for scalping
    },
    Timeframe.M15: {
        "min_confidence": 0.58,
        "profit_target_pct": 0.005,  # 0.5%
        "stop_loss_pct": 0.0025,     # 0.25%
        "duration": "1 hour",
        "rsi_period": 9,
        "sentiment_weight": 0.10,
        "technical_weight": 0.50,
        "ml_weight": 0.35,
        "fundamental_weight": 0.05,
    },
    Timeframe.H1: {
        "min_confidence": 0.55,
        "profit_target_pct": 0.008,  # 0.8%
        "stop_loss_pct": 0.004,      # 0.4%
        "duration": "4 hours",
        "rsi_period": 14,
        "sentiment_weight": 0.15,
        "technical_weight": 0.45,
        "ml_weight": 0.30,
        "fundamental_weight": 0.10,
    },
    Timeframe.H4: {
        "min_confidence": 0.55,
        "profit_target_pct": 0.015,  # 1.5%
        "stop_loss_pct": 0.0075,     # 0.75%
        "duration": "1 day",
        "rsi_period": 14,
        "sentiment_weight": 0.20,
        "technical_weight": 0.40,
        "ml_weight": 0.25,
        "fundamental_weight": 0.15,
    },
    Timeframe.D1: {
        "min_confidence": 0.55,
        "profit_target_pct": 0.03,   # 3%
        "stop_loss_pct": 0.015,      # 1.5%
        "duration": "3 days",
        "rsi_period": 14,
        "sentiment_weight": 0.25,
        "technical_weight": 0.35,
        "ml_weight": 0.25,
        "fundamental_weight": 0.15,
    },
}


class TimeframeSignalGenerator:
    """
    Generates signals optimized for specific timeframes.
    Uses different weights, thresholds, and targets per timeframe.
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
    
    def generate(self, ticker: str, timeframe: Timeframe) -> TradeSetup:
        """
        Generate a complete trade setup for ticker at given timeframe.
        
        Args:
            ticker: Stock symbol
            timeframe: Trading timeframe
            
        Returns:
            TradeSetup with entry, target, stop, and confidence
        """
        config = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS[Timeframe.D1])
        
        logger.info(f"Generating {timeframe.label} signal for {ticker}")
        
        # 1. Get timeframe-appropriate data
        df = mtf_engine.get_data(ticker, timeframe, use_cache=False)
        
        if df.empty or len(df) < 30:
            return self._create_hold_setup(ticker, 0.0, timeframe, config, ["Insufficient data"])
        
        current_price = df["Close"].iloc[-1]
        reasons = []
        
        # 2. Technical Analysis with timeframe-specific thresholds
        try:
            # Detect market regime
            regime_info = adaptive_thresholds.detect_regime(df)
            regime = regime_info.get("regime", "UNKNOWN")
            
            # Get RSI thresholds for regime
            rsi_thresh = adaptive_thresholds.get_rsi_thresholds(regime)
            
            tech_result = technical_signal_generator.analyze(df, thresholds={
                "rsi_oversold": rsi_thresh["oversold"],
                "rsi_overbought": rsi_thresh["overbought"]
            })
            technical_score = tech_result.get("technical_score", 0.5)
            
            if tech_result.get("composite_signal") is True:
                reasons.append(f"Technical: {tech_result.get('buy_signals', 0)}/3 bullish")
            elif tech_result.get("composite_signal") is False:
                reasons.append(f"Technical: {tech_result.get('sell_signals', 0)}/3 bearish")
                
            if regime != "UNKNOWN":
                reasons.append(f"Regime: {regime}")
                
        except Exception as e:
            logger.warning(f"Technical analysis failed: {e}")
            technical_score = 0.5
        
        # 3. Sentiment Analysis (weighted by timeframe)
        sentiment_score = 0.5
        if config["sentiment_weight"] > 0:
            try:
                sent_result = sentiment_aggregator.get_sentiment(ticker)
                raw_score = sent_result.get("overall_score", 0)
                sentiment_score = (raw_score + 1) / 2  # Convert -1,1 to 0,1
                
                if sent_result.get("signal") is True:
                    reasons.append("Sentiment: Bullish")
                elif sent_result.get("signal") is False:
                    reasons.append("Sentiment: Bearish")
            except Exception as e:
                logger.warning(f"Sentiment failed: {e}")
        
        # 4. Fundamental Analysis (for longer timeframes)
        fundamental_score = 0.5
        if config["fundamental_weight"] > 0:
            try:
                fund_signals = fundamentals_fetcher.get_fundamental_signals(ticker)
                fundamental_score = fund_signals.get("fundamental_score", 0.5)
            except Exception as e:
                logger.warning(f"Fundamentals failed: {e}")
        
        # 5. ML Ensemble Prediction (GPU accelerated)
        ml_score = 0.5
        try:
            ensemble_result = ensemble_model.predict(df)
            ml_score = ensemble_result.get("probability", 0.5)
            components = ensemble_result.get("components", {})
            active = [k for k, v in components.items() if v is not None]
            if active:
                reasons.append(f"ML ({', '.join(active)}): {ml_score:.0%}")
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
        
        # 6. Calculate weighted composite score
        composite = (
            technical_score * config["technical_weight"] +
            sentiment_score * config["sentiment_weight"] +
            fundamental_score * config["fundamental_weight"] +
            ml_score * config["ml_weight"]
        )
        
        # 7. Determine direction
        if composite >= 0.55:
            direction = "LONG"
            confidence = composite
        elif composite <= 0.45:
            direction = "SHORT"
            confidence = 1 - composite
        else:
            direction = "HOLD"
            confidence = 1 - abs(composite - 0.5) * 2
        
        # 8. Calculate targets using ATR and config
        target_price, stop_loss = self._calculate_levels(
            current_price, direction, df, config
        )
        
        return TradeSetup(
            ticker=ticker,
            direction=direction,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            timeframe=timeframe.label,
            duration=config["duration"],
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            ml_score=ml_score,
            reasons=reasons,
        )
    
    def _calculate_levels(
        self, 
        price: float, 
        direction: str, 
        df, 
        config: Dict
    ) -> tuple:
        """Calculate target and stop loss based on ATR and config."""
        
        # Calculate ATR for volatility-based levels
        high_low = df["High"] - df["Low"]
        atr = high_low.rolling(window=14).mean().iloc[-1]
        
        # Use config percentages as minimum, ATR as maximum
        pct_target = config["profit_target_pct"]
        pct_stop = config["stop_loss_pct"]
        
        if direction == "LONG":
            target = price * (1 + max(pct_target, atr / price * 1.5))
            stop = price * (1 - max(pct_stop, atr / price * 0.75))
        elif direction == "SHORT":
            target = price * (1 - max(pct_target, atr / price * 1.5))
            stop = price * (1 + max(pct_stop, atr / price * 0.75))
        else:
            target = price
            stop = price
        
        return round(target, 2), round(stop, 2)
    
    def _create_hold_setup(
        self, 
        ticker: str, 
        price: float, 
        timeframe: Timeframe,
        config: Dict,
        reasons: List[str]
    ) -> TradeSetup:
        """Create a HOLD setup when no signal."""
        return TradeSetup(
            ticker=ticker,
            direction="HOLD",
            entry_price=price,
            target_price=price,
            stop_loss=price,
            confidence=0.5,
            timeframe=timeframe.label,
            duration=config["duration"],
            reasons=reasons,
        )


# Singleton instance
timeframe_signal_generator = TimeframeSignalGenerator()
