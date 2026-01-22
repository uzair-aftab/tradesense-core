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

from src.utils.logger import get_logger
from src.data import stock_fetcher, crypto_fetcher, watchlist
from src.analysis import (
    technical_signal_generator, 
    support_resistance,
    adaptive_thresholds
)
from src.analysis.adaptive import AdaptiveThresholds
from src.data.fetcher_fundamentals import fundamentals_fetcher
from src.data.fetcher_options import options_fetcher
from src.sentiment import sentiment_aggregator
from src.ml import FeatureEngineer, signal_model
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
        regime = regime_info["regime"]
        logger.info(f"Market Regime for {ticker}: {regime} (ATR={regime_info['atr_percent']:.2f}%)")
        
        # Get dynamic thresholds
        rsi_thresh = adaptive_thresholds.get_rsi_thresholds(regime)
        thresholds_overrides = {
            "rsi_oversold": rsi_thresh["oversold"],
            "rsi_overbought": rsi_thresh["overbought"]
        }
        
        # Technical Analysis with adaptive thresholds
        tech_result = technical_signal_generator.analyze(df, thresholds=thresholds_overrides)
        technical_score = tech_result.get("technical_score", 0.5)
        
        if tech_result.get("composite_signal") is True:
            reasons.append(f"Technical: {tech_result.get('buy_signals', 0)}/3 buy signals")
        elif tech_result.get("composite_signal") is False:
            reasons.append(f"Technical: {tech_result.get('sell_signals', 0)}/3 sell signals")
            
        reasons.append(f"Regime: {regime} Volatility")
        
        # Support/Resistance
        sr_result = support_resistance.find_levels(df)
        
        # Fundamental Analysis
        fundamental_score = 0.5
        fundamental_details = {}
        if include_fundamentals and not is_crypto:
            try:
                fund_signals = fundamentals_fetcher.get_fundamental_signals(ticker)
                fundamental_score = fund_signals.get("fundamental_score", 0.5)
                fundamental_details = fund_signals.get("details", {})
                
                if fund_signals.get("pe_signal"):
                    reasons.append("Fundamental: P/E below sector average")
                if fund_signals.get("eps_beat_signal"):
                    reasons.append("Fundamental: Recent EPS beat")
            except Exception as e:
                logger.warning(f"Fundamental analysis failed for {ticker}: {e}")
        
        # Sentiment Analysis
        sentiment_score = 0.5
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
        options_score = 0.5
        if include_options and not is_crypto:
            try:
                pc_result = options_fetcher.get_put_call_ratio(ticker)
                if pc_result.get("signal") is True:
                    options_score = 0.7
                    reasons.append(f"Options: Bullish P/C ratio ({pc_result.get('ratio', 0):.2f})")
                elif pc_result.get("ratio") and pc_result.get("ratio") > 1.0:
                    options_score = 0.3
                    reasons.append(f"Options: Bearish P/C ratio ({pc_result.get('ratio', 0):.2f})")
            except Exception as e:
                logger.warning(f"Options analysis failed for {ticker}: {e}")
        
        # ML Model Prediction
        ml_score = 0.5  # Neutral default
        try:
            if signal_model.is_trained or os.path.exists(signal_model.model_path):
                # Generate features for current state
                feature_engineer = FeatureEngineer()
                features_df = feature_engineer.calculate_features(df, include_targets=False)
                if not features_df.empty:
                    # Get latest row for prediction
                    X_latest = features_df[feature_engineer.feature_columns].iloc[[-1]]
                    ml_score = signal_model.predict_proba(X_latest)[0]
                    reasons.append(f"ML Model: {ml_score:.0%} bullish probability")
        except Exception as e:
            logger.warning(f"ML prediction failed for {ticker}: {e}")
        
        # Calculate weighted composite score
        composite_score = (
            technical_score * self.weights.technical +
            fundamental_score * self.weights.fundamental +
            sentiment_score * self.weights.sentiment +
            options_score * self.weights.options +
            ml_score * self.weights.ml
        )
        
        # Determine signal type
        if composite_score >= 0.6:
            signal_type = SignalType.BUY
        elif composite_score <= 0.4:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Calculate confidence
        # Higher confidence when all signals agree
        scores = [technical_score, fundamental_score, sentiment_score, options_score]
        score_variance = np.var(scores)
        agreement_factor = 1 - min(score_variance * 2, 0.5)  # Lower variance = higher agreement
        confidence = abs(composite_score - 0.5) * 2 * agreement_factor
        
        # Calculate targets with adaptive multiplier
        sl_multiplier = adaptive_thresholds.get_stop_loss_multiplier(regime)
        
        take_profit, stop_loss = self._calculate_targets(
            current_price, signal_type, sr_result, df, sl_multiplier
        )
        
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
        high_low = df["High"] - df["Low"]
        atr = high_low.rolling(window=14).mean().iloc[-1]
        
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
