"""
TradeSense Unified Signal Generator

Combines technical, fundamental, and sentiment signals
into actionable trading signals with targets.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import numpy as np
import pandas_ta as ta

from src.utils.logger import get_logger
from src.data import stock_fetcher, crypto_fetcher, watchlist
from src.analysis import (
    technical_signal_generator, 
    support_resistance,
    adaptive_thresholds
)
from src.analysis.adaptive import AdaptiveThresholds
from src.analysis.fundamental import fundamental_analyzer
from src.analysis.options import options_analyzer
from src.data.fetcher_fundamentals import fundamentals_fetcher
from src.data.fetcher_options import options_fetcher
from src.signals.tp_sl_calculator import tp_sl_calculator
from src.sentiment import sentiment_aggregator
from src.ml import FeatureEngineer, signal_model, ensemble_model, unified_predictor
from src.config import config

logger = get_logger(__name__)


class SignalType(Enum):
    """Signal type enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """
    Represents a complete trading signal.
    
    Contains all information needed to execute a trade,
    including entry, targets, and stop loss.
    """
    ticker: str
    signal_type: SignalType
    confidence: float  # 0-1
    entry_price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Component scores
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    options_score: float = 0.0
    ml_score: float = 0.5  # ML model prediction (neutral default)
    
    # Supporting details
    reasons: List[str] = field(default_factory=list)
    technical_details: Dict = field(default_factory=dict)
    fundamental_details: Dict = field(default_factory=dict)
    sentiment_details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "signal": self.signal_type.value,
            "confidence": round(self.confidence, 2),
            "entry_price": round(self.entry_price, 2),
            "take_profit": round(self.take_profit, 2) if self.take_profit else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "technical_score": round(self.technical_score, 2),
            "fundamental_score": round(self.fundamental_score, 2),
            "sentiment_score": round(self.sentiment_score, 2),
            "options_score": round(self.options_score, 2),
            "reasons": self.reasons,
        }


@dataclass
class SignalWeights:
    """Weights for combining different signal sources."""
    technical: float = 0.30
    fundamental: float = 0.20
    sentiment: float = 0.20
    options: float = 0.10
    ml: float = 0.20  # ML model weight


class SignalGenerator:
    """
    Unified signal generator combining all analysis sources.
    
    Generates buy/sell signals with take profit and stop loss targets
    based on technical, fundamental, sentiment, and options analysis.
    
    Example:
        >>> generator = SignalGenerator()
        >>> signal = generator.generate_signal("AAPL")
        >>> print(signal.signal_type, signal.confidence)
    """
    
    def __init__(
        self,
        weights: Optional[SignalWeights] = None,
        min_confidence: float = 0.6
    ):
        """
        Initialize signal generator.
        
        Args:
            weights: Custom weights for signal sources
            min_confidence: Minimum confidence to generate a signal
        """
        self.weights = weights or SignalWeights()
        self.min_confidence = min_confidence

    def _combine_scores(self, scores: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, float]]:
        weight_map = {
            "technical": self.weights.technical,
            "fundamental": self.weights.fundamental,
            "sentiment": self.weights.sentiment,
            "options": self.weights.options,
            "ml": self.weights.ml,
        }

        active = {k: v for k, v in scores.items() if v is not None}
        if not active:
            return 0.5, {}

        total_weight = sum(weight_map[k] for k in active.keys())
        if total_weight <= 0:
            return 0.5, {}

        normalized = {k: weight_map[k] / total_weight for k in active.keys()}
        composite = sum(active[k] * normalized[k] for k in active.keys())
        return composite, normalized

    def _calculate_confidence(
        self,
        composite_score: float,
        signal_type: SignalType,
        active_sources: int
    ) -> float:
        decision_strength = min(1.0, abs(composite_score - 0.5) * 2)
        coverage = min(1.0, active_sources / 5.0)

        if signal_type == SignalType.HOLD:
            base = 1 - decision_strength
        else:
            base = decision_strength

        return max(0.0, min(1.0, base * (0.6 + 0.4 * coverage)))
    
    def generate_signal(
        self,
        ticker: str,
        include_fundamentals: bool = True,
        include_sentiment: bool = True,
        include_options: bool = True
    ) -> TradingSignal:
        """
        Generate a complete trading signal for a ticker.
        
        Args:
            ticker: Stock/crypto symbol
            include_fundamentals: Include fundamental analysis
            include_sentiment: Include sentiment analysis
            include_options: Include options analysis
            
        Returns:
            TradingSignal object with all details
        """
        logger.info(f"Generating signal for {ticker}")
        
        # Get price data
        is_crypto = watchlist.is_crypto(ticker)
        if is_crypto:
            df = crypto_fetcher.get_historical(ticker, period="3mo")
        else:
            df = stock_fetcher.get_historical(ticker, period="3mo")
        
        if df.empty:
            return self._create_hold_signal(ticker, 0.0, ["No price data available"])
        
        current_price = df["Close"].iloc[-1]
        reasons = []
        
        # Adaptive Thresholds - Detect Regime
        regime_info = adaptive_thresholds.detect_regime(df)
        regime = regime_info.get("regime", "UNKNOWN")
        volatility = regime_info.get("volatility", "NORMAL")
        logger.info(f"Market Regime for {ticker}: {regime} (Vol: {volatility}, ATR={regime_info.get('atr_percent', 'N/A')}%)")
        
        # Get dynamic thresholds
        rsi_thresh = adaptive_thresholds.get_rsi_thresholds(regime)
        thresholds_overrides = {
            "rsi_oversold": rsi_thresh["oversold"],
            "rsi_overbought": rsi_thresh["overbought"]
        }
        
        # Technical Analysis with adaptive thresholds
        tech_result = technical_signal_generator.analyze(df, thresholds=thresholds_overrides)
        technical_score = tech_result.get("technical_score")
        
        if tech_result.get("composite_signal") is True:
            reasons.append(f"Technical: {tech_result.get('buy_signals', 0)}/3 buy signals")
        elif tech_result.get("composite_signal") is False:
            reasons.append(f"Technical: {tech_result.get('sell_signals', 0)}/3 sell signals")
            
        if regime != "UNKNOWN":
            reasons.append(f"Regime: {regime} ({volatility} Vol)")
        
        # Support/Resistance
        sr_result = support_resistance.find_levels(df)
        
        # Fundamental Analysis
        fundamental_score = None
        fundamental_details = {}
        if include_fundamentals and not is_crypto:
            try:
                fund_result = fundamental_analyzer.analyze(ticker)
                fundamental_score = fund_result.get("fundamental_score")
                fundamental_details = fund_result
                
                # Add key reasons from comprehensive analysis
                fund_reasons = fund_result.get("reasons", [])
                for reason in fund_reasons[:2]:  # Top 2 reasons
                    reasons.append(f"Fundamental: {reason}")
            except Exception as e:
                logger.warning(f"Fundamental analysis failed for {ticker}: {e}")
        
        # Sentiment Analysis
        sentiment_score = None
        sentiment_details = {}
        if include_sentiment:
            try:
                sent_result = sentiment_aggregator.get_sentiment(
                    ticker, include_finbert=True
                )
                # Convert -1 to +1 scale to 0-1
                raw_score = sent_result.get("overall_score", 0)
                sentiment_score = (raw_score + 1) / 2
                sentiment_details = sent_result
                
                if sent_result.get("signal") is True:
                    reasons.append("Sentiment: Bullish market sentiment")
                elif sent_result.get("signal") is False:
                    reasons.append("Sentiment: Bearish market sentiment")
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for {ticker}: {e}")
        
        # Options Analysis
        options_score = None
        if include_options and not is_crypto:
            try:
                opt_result = options_analyzer.analyze(ticker, include_unusual=True)
                options_score = opt_result.get("options_score")
                
                # Add key options insights
                opt_reasons = opt_result.get("reasons", [])
                for reason in opt_reasons[:2]:  # Top 2 reasons
                    reasons.append(f"Options: {reason}")
            except Exception as e:
                logger.warning(f"Options analysis failed for {ticker}: {e}")
        
        # ML Model Prediction (unified interface with fallback)
        ml_score = None
        try:
            # Use unified predictor with automatic fallback
            ml_result = unified_predictor.predict_with_fallback(
                df,
                preferred_models=["ensemble", "xgboost", "lstm"]
            )
            
            if ml_result:
                ml_score = ml_result.probability
                reasons.append(
                    f"ML {ml_result.model_type.upper()}: "
                    f"{ml_score:.0%} bullish (conf: {ml_result.confidence:.0%})"
                )
        except Exception as e:
            logger.warning(f"ML prediction failed for {ticker}: {e}")
        
        scores = {
            "technical": technical_score,
            "fundamental": fundamental_score,
            "sentiment": sentiment_score,
            "options": options_score,
            "ml": ml_score,
        }

        composite_score, normalized_weights = self._combine_scores(scores)
        
        # Determine signal type
        if composite_score >= 0.6:
            signal_type = SignalType.BUY
        elif composite_score <= 0.4:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        active_sources = len([v for v in scores.values() if v is not None])
        confidence = self._calculate_confidence(composite_score, signal_type, active_sources)

        if signal_type != SignalType.HOLD and confidence < self.min_confidence:
            signal_type = SignalType.HOLD
            reasons.append("Confidence below threshold")
            confidence = self._calculate_confidence(composite_score, signal_type, active_sources)
        
        # Calculate targets with adaptive multiplier
        sl_multiplier = adaptive_thresholds.get_stop_loss_multiplier(regime)
        
        # Use dedicated TP/SL calculator
        direction = "long" if signal_type == SignalType.BUY else "short" if signal_type == SignalType.SELL else None
        if direction:
            take_profit, stop_loss = tp_sl_calculator.calculate(
                entry_price=current_price,
                direction=direction,
                df=df,
                sr_levels=sr_result,
                sl_multiplier=sl_multiplier
            )
        else:
            take_profit, stop_loss = None, None
        
        return TradingSignal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=current_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            options_score=options_score,
            ml_score=ml_score,
            reasons=reasons,
            technical_details=tech_result,
            fundamental_details=fundamental_details,
            sentiment_details=sentiment_details,
        )
    
    def _calculate_targets(
        self,
        entry_price: float,
        signal_type: SignalType,
        sr_levels: Dict,
        df,
        sl_multiplier: float = 1.5
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate take profit and stop loss targets.
        
        Uses support/resistance levels and ATR-based volatility.
        """
        if signal_type == SignalType.HOLD:
            return None, None
        
        # Calculate ATR for volatility-based stops
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr = atr_series.iloc[-1] if atr_series is not None and not atr_series.empty else None
        if atr is None or np.isnan(atr):
            atr = (df["High"] - df["Low"]).rolling(window=14).mean().iloc[-1]
        
        if signal_type == SignalType.BUY:
            # Take profit at resistance or 2x ATR
            resistance = sr_levels.get("nearest_resistance")
            if resistance and resistance > entry_price:
                take_profit = resistance
            else:
                take_profit = entry_price + (atr * 2)
            
            # Stop loss at support or dynamic ATR multiple
            support = sr_levels.get("nearest_support")
            if support and support < entry_price:
                stop_loss = support
            else:
                stop_loss = entry_price - (atr * sl_multiplier)
        else:  # SELL
            # Take profit at support or 2x ATR down
            support = sr_levels.get("nearest_support")
            if support and support < entry_price:
                take_profit = support
            else:
                take_profit = entry_price - (atr * 2)
            
            # Stop loss at resistance or dynamic ATR multiple
            resistance = sr_levels.get("nearest_resistance")
            if resistance and resistance > entry_price:
                stop_loss = resistance
            else:
                stop_loss = entry_price + (atr * sl_multiplier)
        
        return take_profit, stop_loss
    
    def _create_hold_signal(
        self,
        ticker: str,
        price: float,
        reasons: List[str]
    ) -> TradingSignal:
        """Create a HOLD signal with given reasons."""
        return TradingSignal(
            ticker=ticker,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            entry_price=price,
            reasons=reasons,
        )
    
    def scan_watchlist(
        self,
        category: str = "all"
    ) -> List[TradingSignal]:
        """
        Scan entire watchlist and generate signals.
        
        Args:
            category: Asset category ('stocks', 'etfs', 'crypto', 'all')
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if category in ("stocks", "all"):
            for ticker in watchlist.get_stocks():
                try:
                    signal = self.generate_signal(ticker)
                    signals.append(signal)
                except Exception as e:
                    logger.error(f"Error scanning {ticker}: {e}")
        
        if category in ("etfs", "all"):
            for ticker in watchlist.get_etfs():
                try:
                    signal = self.generate_signal(ticker)
                    signals.append(signal)
                except Exception as e:
                    logger.error(f"Error scanning {ticker}: {e}")
        
        if category in ("crypto", "all"):
            for ticker in watchlist.get_crypto():
                try:
                    # Crypto: skip fundamentals and options
                    signal = self.generate_signal(
                        ticker,
                        include_fundamentals=False,
                        include_options=False
                    )
                    signals.append(signal)
                except Exception as e:
                    logger.error(f"Error scanning {ticker}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals
    
    def get_top_signals(
        self,
        n: int = 5,
        signal_type: Optional[SignalType] = None
    ) -> List[TradingSignal]:
        """
        Get top N signals from watchlist scan.
        
        Args:
            n: Number of signals to return
            signal_type: Filter by signal type (BUY/SELL)
            
        Returns:
            List of top signals by confidence
        """
        signals = self.scan_watchlist()
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        return signals[:n]


# Singleton instance
signal_generator = SignalGenerator()
