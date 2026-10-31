"""
TradeSense Institutional Multi-Strategy Engine & Tournament Arena
==================================================================

Implements top-tier hedge-fund quantitative trading strategies:
1. DeepQuantileSniper — ML Quantile Asymmetry + Whale Confluence
2. InstitutionalWhaleFlow — Large-Block Order Flow + VWAP Anchoring
3. VolatilityBreakout — Donchian Channel Expansion + ADX Trend Acceleration
4. StatisticalMeanReversion — Bollinger Band 2.5-Sigma Extremes + RSI Exhaustion
5. EnsembleAlpha — Meta-Consensus Voting Algorithm

Provides the StrategyArena tournament runner for parallel strategy evaluation.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.models.ai_engine import AIEngine, QuantilePrediction
from src.models.trader import Gauntlet, Side, TradeOrder
from src.utils.logger import get_logger

logger = get_logger("hfp.strategies")


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    strategy_name: str
    ticker: str
    signal_type: SignalType
    strength: float           # 0.0 to 1.0 confidence/strength
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def ensure_strategy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that all technical & microstructure columns required by the 6
    institutional strategies exist on a copy of *df*, without mutating the
    original DataFrame passed to AIEngine.
    """
    if df is None or df.empty:
        return df

    needed = {
        "RSI_14", "EMA_9", "EMA_20", "EMA_50", "ATR_14",
        "BB_Mid", "BB_Upper", "BB_Lower", "VWAP", "Whale_Z", "ADX_14", "Sentiment",
    }
    if needed.issubset(df.columns):
        return df

    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float) if "High" in out.columns else close
    low = out["Low"].astype(float) if "Low" in out.columns else close
    volume = out["Volume"].astype(float) if "Volume" in out.columns else pd.Series(1.0, index=out.index)

    if "EMA_9" not in out.columns:
        out["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    if "EMA_20" not in out.columns:
        out["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    if "EMA_50" not in out.columns:
        out["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    if "RSI_14" not in out.columns:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        out["RSI_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    if "ATR_14" not in out.columns:
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        out["ATR_14"] = tr.rolling(14, min_periods=1).mean().bfill().fillna(close * 0.015)

    if "BB_Mid" not in out.columns or "BB_Upper" not in out.columns or "BB_Lower" not in out.columns:
        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std().fillna(close * 0.01)
        out["BB_Mid"] = bb_mid
        out["BB_Upper"] = bb_mid + 2.0 * bb_std
        out["BB_Lower"] = bb_mid - 2.0 * bb_std

    if "VWAP" not in out.columns:
        typical = (high + low + close) / 3.0
        vol_safe = volume.replace(0, np.nan).ffill().fillna(1.0)
        out["VWAP"] = (
            (typical * vol_safe).rolling(20, min_periods=1).sum()
            / vol_safe.rolling(20, min_periods=1).sum()
        ).fillna(close)

    if "Whale_Z" not in out.columns:
        vol_mean = volume.rolling(20, min_periods=2).mean()
        vol_std = volume.rolling(20, min_periods=2).std().replace(0, np.nan)
        out["Whale_Z"] = ((volume - vol_mean) / vol_std).fillna(0.0).clip(-5.0, 8.0)

    if "ADX_14" not in out.columns:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_safe = out["ATR_14"].replace(0, np.nan).bfill().fillna(1.0)
        plus_di = 100.0 * pd.Series(plus_dm, index=out.index).ewm(span=14, adjust=False).mean() / atr_safe
        minus_di = 100.0 * pd.Series(minus_dm, index=out.index).ewm(span=14, adjust=False).mean() / atr_safe
        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(20.0)
        out["ADX_14"] = dx.ewm(span=14, adjust=False).mean().fillna(20.0)

    if "Sentiment" not in out.columns:
        out["Sentiment"] = 0.0

    return out


class BaseStrategy(ABC):
    """Abstract base class for all institutional trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        """Evaluate market data and generate a trading signal."""
        pass

    def get_decision_attribution(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> Dict[str, Any]:
        """Return rich quantitative attribution explaining this strategy's live decision."""
        if df is not None and not df.empty and "RSI_14" not in df.columns:
            df = ensure_strategy_indicators(df)

        sig = self.generate_signal(df, current_idx=current_idx, prediction=prediction)
        price = float(df["Close"].iloc[current_idx]) if not df.empty else 100.0
        bar = df.iloc[current_idx] if not df.empty else {}

        target = sig.target_price or (price * 1.03 if sig.signal_type == SignalType.BUY else price * 0.97)
        stop = sig.stop_price or (price * 0.98 if sig.signal_type == SignalType.BUY else price * 1.02)
        upside = abs(target - price) / price * 100.0 if price > 0 else 0.0
        downside = abs(price - stop) / price * 100.0 if price > 0 else 0.0
        rrr = upside / max(0.01, downside)

        whale_z = float(bar.get("Whale_Z", 0.0)) if hasattr(bar, "get") else 0.0
        vwap = float(bar.get("VWAP", price)) if hasattr(bar, "get") else price
        rsi = float(bar.get("RSI_14", 50.0)) if hasattr(bar, "get") else 50.0
        sentiment = float(bar.get("Sentiment", 0.0)) if hasattr(bar, "get") else 0.0

        return {
            "strategy_name": self.name,
            "signal_type": sig.signal_type.value,
            "strength": sig.strength,
            "reason": sig.reason,
            "target_price": target,
            "stop_price": stop,
            "expected_return_pct": upside,
            "rrr": rrr,
            "hurdle_pass": upside >= 1.5,
            "factors": [
                {
                    "name": "Whale Flow Z-Score",
                    "value": f"{whale_z:.2f}σ",
                    "status": "ACCUMULATION" if whale_z >= 2.0 else ("DISTRIBUTION" if whale_z <= -2.0 else "NORMAL"),
                    "color": "#10b981" if whale_z >= 2.0 else ("#ef4444" if whale_z <= -2.0 else "#94a3b8"),
                },
                {
                    "name": "VWAP Distance",
                    "value": f"{((price - vwap) / vwap) * 100:+.2f}%",
                    "status": "ABOVE VWAP" if price >= vwap else "BELOW VWAP",
                    "color": "#10b981" if price >= vwap else "#ef4444",
                },
                {
                    "name": "RSI 14 Momentum",
                    "value": f"{rsi:.1f}",
                    "status": "OVERSOLD (<35)" if rsi < 35 else ("OVERBOUGHT (>65)" if rsi > 65 else "NEUTRAL"),
                    "color": "#10b981" if rsi < 35 else ("#ef4444" if rsi > 65 else "#94a3b8"),
                },
                {
                    "name": "News & Catalyst Pulse",
                    "value": f"{sentiment:+.2f}",
                    "status": "BULLISH CATALYST" if sentiment >= 0.15 else ("BEARISH HEADWIND" if sentiment <= -0.15 else "NEUTRAL WIRE"),
                    "color": "#10b981" if sentiment >= 0.15 else ("#ef4444" if sentiment <= -0.15 else "#94a3b8"),
                },
                {
                    "name": "Wealthsimple Spread Hurdle",
                    "value": f"{upside:.2f}% (min 1.5%)",
                    "status": "PASS" if upside >= 1.5 else "BELOW 1.5%",
                    "color": "#10b981" if upside >= 1.5 else "#f59e0b",
                },
            ],
            "sub_signals": [],
        }


class DeepQuantileSniper(BaseStrategy):
    """
    Strategy 1: Deep Quantile Sniper.
    Exploits asymmetric probabilistic forecast distributions from LSTM + XGBoost.
    Requires RRR >= 1.5x and model confidence >= 68%.
    """

    def __init__(self, min_conf: float = 0.68, min_rrr: float = 1.5):
        super().__init__("Deep Quantile Sniper")
        self.min_conf = min_conf
        self.min_rrr = min_rrr

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if df is None or df.empty:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="No market data available")

        price = float(df["Close"].iloc[current_idx])
        bar = df.iloc[current_idx]

        # Check if prediction is live for the current bar's price domain (within 3%)
        use_live_pred = (
            prediction is not None
            and prediction.current_price > 0
            and abs(prediction.current_price - price) / max(price, 1e-6) <= 0.03
        )

        if use_live_pred and prediction is not None:
            if prediction.direction == "BULLISH" and prediction.confidence >= self.min_conf:
                stop = float(min(prediction.p10, price * 0.988))
                target = float(max(prediction.p90, price * 1.022))
                upside = target - price
                downside = price - stop
                rrr = upside / max(0.01, downside)
                if rrr >= self.min_rrr:
                    return StrategySignal(
                        self.name,
                        ticker,
                        SignalType.BUY,
                        strength=prediction.confidence,
                        target_price=target,
                        stop_price=stop,
                        reason=f"Quantile Asymmetry RRR={rrr:.2f}x | Conf={prediction.confidence:.0%}",
                    )
            elif prediction.direction == "BEARISH" and prediction.confidence >= self.min_conf:
                stop = float(max(prediction.p90, price * 1.015))
                target = float(min(prediction.p10, price * 0.975))
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.SELL,
                    strength=prediction.confidence,
                    target_price=target,
                    stop_price=stop,
                    reason=f"Bearish Quantile Skew P50=${prediction.p50:,.2f} | Conf={prediction.confidence:.0%}",
                )

        # Bar-local quantile approximation when backtesting historical bars in Tournament Arena
        if len(df) >= 20:
            ema9 = float(bar.get("EMA_9", price))
            ema20 = float(bar.get("EMA_20", price))
            atr = float(bar.get("ATR_14", price * 0.018))
            rsi = float(bar.get("RSI_14", 50.0))
            whale_z = float(bar.get("Whale_Z", 0.0))

            if ema9 >= ema20 and 42.0 <= rsi <= 67.0 and whale_z >= 0.35:
                stop = price - 1.35 * atr
                target = price + 2.85 * atr
                rrr = (target - price) / max(0.01, price - stop)
                conf = float(np.clip(0.70 + 0.05 * whale_z, 0.68, 0.92))
                if rrr >= self.min_rrr and conf >= self.min_conf:
                    return StrategySignal(
                        self.name,
                        ticker,
                        SignalType.BUY,
                        strength=conf,
                        target_price=target,
                        stop_price=stop,
                        reason=f"Bar Quantile Skew RRR={rrr:.2f}x | Conf={conf:.0%}",
                    )
            elif ema9 < ema20 * 0.992 and (rsi >= 66.0 or whale_z <= -1.0):
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.SELL,
                    strength=0.74,
                    target_price=price - 2.2 * atr,
                    stop_price=price + 1.4 * atr,
                    reason=f"Downside Quantile Breakdown | RSI={rsi:.1f}",
                )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Conditions not met")


class InstitutionalWhaleFlow(BaseStrategy):
    """
    Strategy 2: Institutional Whale Flow Tracker.
    Capitalizes on smart-money block order accumulation when Volume Z-score >= 2.0
    and price trades above VWAP with positive volume momentum.
    """

    def __init__(self, z_threshold: float = 2.0):
        super().__init__("Institutional Whale Flow")
        self.z_threshold = z_threshold

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if df.empty or len(df) < 5:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        bar = df.iloc[current_idx]
        price = float(bar["Close"])
        whale_z = float(bar.get("Whale_Z", 0.0))
        vwap = float(bar.get("VWAP", price))
        atr = float(bar.get("ATR_14", price * 0.02))

        # Whale accumulation check: strong volume spike trading above VWAP
        if whale_z >= self.z_threshold and price >= vwap:
            stop = price - 1.5 * atr
            target = price + 3.0 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=min(1.0, max(0.72, whale_z / 3.5)),
                target_price=target,
                stop_price=stop,
                reason=f"Whale volume surge Z={whale_z:.2f}σ above VWAP",
            )
        elif whale_z >= self.z_threshold and price < vwap:
            # Whale distribution / dump
            stop = price + 1.5 * atr
            target = price - 3.0 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=min(1.0, max(0.72, whale_z / 3.5)),
                target_price=target,
                stop_price=stop,
                reason=f"Whale liquidation Z={whale_z:.2f}σ below VWAP",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Normal volume")


class VolatilityBreakout(BaseStrategy):
    """
    Strategy 3: Donchian Volatility Breakout.
    Enters trend continuation on 20-bar channel high break with ADX > 20 confirmation.
    """

    def __init__(self, lookback: int = 20, adx_threshold: float = 20.0):
        super().__init__("Donchian Volatility Breakout")
        self.lookback = lookback
        self.adx_threshold = adx_threshold

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if len(df) < self.lookback + 5:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        recent = df.iloc[: current_idx + 1 if current_idx != -1 else len(df)]
        bar = recent.iloc[-1]
        price = float(bar["Close"])
        highs = recent["High"].iloc[-self.lookback - 1 : -1]
        lows = recent["Low"].iloc[-min(10, self.lookback) - 1 : -1]
        highest_high = float(highs.max()) if not highs.empty else price
        lowest_low = float(lows.min()) if not lows.empty else price
        adx = float(bar.get("ADX_14", 20.0))
        atr = float(bar.get("ATR_14", price * 0.02))
        bb_u = float(bar.get("BB_Upper", price * 1.02))
        bb_l = float(bar.get("BB_Lower", price * 0.98))
        bb_b = (price - bb_l) / max(0.01, (bb_u - bb_l))

        if price > highest_high:
            if adx < self.adx_threshold:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.HOLD,
                    0.0,
                    reason=f"Breakout suppressed in chop (ADX={adx:.1f} < {self.adx_threshold:.1f})",
                )
            if bb_b > 0.92 and adx < 26.0:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.HOLD,
                    0.0,
                    reason=f"Breakout suppressed at ceiling (BB%B={bb_b*100:.1f}%, ADX={adx:.1f})",
                )

            stop = price - 1.8 * atr
            target = price + 3.6 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=min(1.0, max(0.70, adx / 35.0)),
                target_price=target,
                stop_price=stop,
                reason=f"Breakout above {self.lookback}-bar high (${highest_high:.2f}) | ADX={adx:.1f}",
            )
        elif price < lowest_low and adx >= self.adx_threshold:
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=0.75,
                target_price=price - 3.0 * atr,
                stop_price=price + 1.8 * atr,
                reason=f"Channel breakdown below ${lowest_low:.2f} | ADX={adx:.1f}",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Within channel")


class StatisticalMeanReversion(BaseStrategy):
    """
    Strategy 4: Statistical Mean Reversion.
    Fades Bollinger Band extensions when RSI indicates exhaustion.
    """

    def __init__(self):
        super().__init__("Statistical Mean Reversion")

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if len(df) < 20:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        if "RSI_14" not in df.columns:
            df = ensure_strategy_indicators(df)

        bar = df.iloc[current_idx]
        price = float(bar["Close"])
        lower_bb = float(bar.get("BB_Lower", price * 0.98))
        upper_bb = float(bar.get("BB_Upper", price * 1.02))
        mid_bb = float(bar.get("BB_Mid", price))
        rsi = float(bar.get("RSI_14", 50.0))
        atr = float(bar.get("ATR_14", price * 0.015))

        # Oversold bounce: Price at or near lower Bollinger Band AND RSI oversold (<= 40)
        if price <= lower_bb * 1.005 and rsi <= 40.0:
            target = max(mid_bb, price + 2.2 * atr)
            stop = price - 1.4 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=0.82,
                target_price=target,
                stop_price=stop,
                reason=f"Mean reversion floor bounce (${lower_bb:.2f}) | RSI={rsi:.1f}",
            )
        elif price >= upper_bb * 0.995 and rsi >= 62.0:
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=0.78,
                target_price=min(mid_bb, price - 2.0 * atr),
                stop_price=price + 1.4 * atr,
                reason=f"Mean reversion ceiling fade (${upper_bb:.2f}) | RSI={rsi:.1f}",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Price inside bands")


class NewsCatalystMomentum(BaseStrategy):
    """
    Strategy 5: News & Catalyst Momentum.
    Combines live financial news/catalyst sentiment with institutional volume flow
    (Whale_Z) and VWAP trend structure to capture high-conviction catalyst breakouts.
    """

    def __init__(self, sentiment_threshold: float = 0.14):
        super().__init__("News & Catalyst Momentum")
        self.sentiment_threshold = sentiment_threshold

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if df is None or df.empty or len(df) < 10:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        if "RSI_14" not in df.columns:
            df = ensure_strategy_indicators(df)

        recent = df.iloc[: current_idx + 1 if current_idx != -1 else len(df)]
        bar = recent.iloc[-1]
        price = float(bar["Close"])
        sentiment = float(bar.get("Sentiment", 0.0))
        whale_z = float(bar.get("Whale_Z", 0.0))
        vwap = float(bar.get("VWAP", price))
        rsi = float(bar.get("RSI_14", 50.0))
        atr = float(bar.get("ATR_14", price * 0.018))

        # When historical bar has mild/static recorded sentiment, blend with 3-bar price+volume catalyst shock
        effective_sentiment = sentiment
        if abs(effective_sentiment) < self.sentiment_threshold and len(recent) >= 5:
            ret_3 = (price - float(recent["Close"].iloc[-4])) / max(float(recent["Close"].iloc[-4]), 1e-6)
            if ret_3 >= 0.014 and whale_z >= 0.85:
                effective_sentiment = min(0.65, max(self.sentiment_threshold + 0.05, ret_3 * 10.0))
            elif ret_3 <= -0.016 and whale_z >= 0.85:
                effective_sentiment = max(-0.65, min(-self.sentiment_threshold - 0.05, ret_3 * 10.0))

        if effective_sentiment >= self.sentiment_threshold and price >= vwap * 0.997 and rsi <= 72.0:
            stop = price - 1.45 * atr
            target = price + 3.10 * atr
            strength = float(np.clip(0.68 + abs(effective_sentiment) * 0.35 + max(0.0, whale_z) * 0.05, 0.68, 0.96))
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=strength,
                target_price=target,
                stop_price=stop,
                reason=f"Bullish Catalyst Pulse (Sent={effective_sentiment:+.2f}, Z={whale_z:.1f}σ) above VWAP",
            )
        elif effective_sentiment <= -self.sentiment_threshold and price < vwap:
            stop = price + 1.45 * atr
            target = price - 2.90 * atr
            strength = float(np.clip(0.68 + abs(effective_sentiment) * 0.35, 0.68, 0.94))
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=strength,
                target_price=target,
                stop_price=stop,
                reason=f"Bearish Catalyst Headwind (Sent={effective_sentiment:+.2f}) below VWAP",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="No active catalyst impulse")


class EnsembleAlpha(BaseStrategy):
    """
    Strategy 6: Ensemble Multi-Strategy Alpha.
    Aggregates consensus votes across all 5 sub-strategies with dynamic market regime gating.
    """

    def __init__(self, strategies: Optional[List[BaseStrategy]] = None):
        super().__init__("Ensemble Multi-Strategy Alpha")
        self.strategies = strategies or [
            DeepQuantileSniper(),
            InstitutionalWhaleFlow(),
            VolatilityBreakout(),
            StatisticalMeanReversion(),
            NewsCatalystMomentum(),
        ]
        from src.models.regime import RegimeClassifier
        self.regime_classifier = RegimeClassifier()
        self.last_regime_context = None

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if df is not None and not df.empty and "RSI_14" not in df.columns:
            df = ensure_strategy_indicators(df)

        price = float(df["Close"].iloc[current_idx]) if not df.empty else 100.0

        # 1. Evaluate Active Market Regime
        regime_ctx = self.regime_classifier.classify(df, current_idx)
        self.last_regime_context = regime_ctx

        # 2. Hard Cash Preservation Guard in Bear Trends
        if regime_ctx.cash_preservation_active:
            self.last_sub_signals = []
            self.last_buy_votes = 0.0
            self.last_sell_votes = 1.0
            return StrategySignal(
                self.name,
                ticker,
                SignalType.HOLD,
                0.0,
                reason=f"CASH PRESERVATION LOCK: {regime_ctx.reason}",
            )

        # 3. Query Sub-Strategies
        signals = [s.generate_signal(df, current_idx, prediction) for s in self.strategies]
        self.last_sub_signals = signals

        # 4. Regime-Adaptive Conviction Weighting
        buy_votes = 0.0
        sell_votes = 0.0
        from src.models.regime import MarketRegime

        for s in signals:
            strat_name = s.strategy_name
            weight = 1.0

            if regime_ctx.regime == MarketRegime.BULL_TREND:
                # In Bull Trend: amplify breakout, catalyst & whale flow; mute mean reversion fades
                if strat_name in ("Volatility Breakout", "Donchian Volatility Breakout"):
                    weight = 1.3
                elif strat_name in ("Institutional Whale Flow", "News & Catalyst Momentum"):
                    weight = 1.25
                elif strat_name == "Statistical Mean Reversion":
                    weight = 0.3
            elif regime_ctx.regime == MarketRegime.RANGING_CHOP:
                # In Range Chop: suppress breakouts; amplify mean reversion
                if strat_name in ("Volatility Breakout", "Donchian Volatility Breakout"):
                    weight = 0.0  # Zero weight to breakout buys in chop!
                elif strat_name == "Statistical Mean Reversion":
                    weight = 1.5
                elif strat_name == "Institutional Whale Flow":
                    weight = 0.6

            if s.signal_type == SignalType.BUY:
                buy_votes += s.strength * weight
            elif s.signal_type == SignalType.SELL:
                sell_votes += s.strength * weight

        self.last_buy_votes = buy_votes
        self.last_sell_votes = sell_votes

        min_threshold = 0.65 if prediction is None else 0.85

        # In Ranging Chop, never allow BUY if price is at upper half of Bollinger Band
        if regime_ctx.regime == MarketRegime.RANGING_CHOP and buy_votes >= min_threshold:
            if not regime_ctx.allow_mean_reversion_buys:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.HOLD,
                    0.0,
                    reason=f"BUY suppressed in range chop (BB%B={regime_ctx.bb_pct_b*100:.1f}% > 20%)",
                )

        if buy_votes > sell_votes and buy_votes >= min_threshold:
            reasons = [f"{s.strategy_name}: {s.reason}" for s in signals if s.signal_type == SignalType.BUY]
            targets = [s.target_price for s in signals if s.target_price and s.signal_type == SignalType.BUY]
            stops = [s.stop_price for s in signals if s.stop_price and s.signal_type == SignalType.BUY]
            strength = min(1.0, max(0.70, buy_votes / len(self.strategies) * 1.5))
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=strength,
                target_price=float(np.mean(targets)) if targets else price * 1.03,
                stop_price=float(np.mean(stops)) if stops else price * 0.98,
                reason=f"[{regime_ctx.regime.value}] " + " | ".join(reasons[:2]),
            )
        elif sell_votes > buy_votes and sell_votes >= min_threshold:
            reasons = [f"{s.strategy_name}: {s.reason}" for s in signals if s.signal_type == SignalType.SELL]
            strength = min(1.0, max(0.70, sell_votes / len(self.strategies) * 1.5))
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=strength,
                target_price=price * 0.97,
                stop_price=price * 1.02,
                reason=f"[{regime_ctx.regime.value}] " + " | ".join(reasons[:2]),
            )

        return StrategySignal(
            self.name,
            ticker,
            SignalType.HOLD,
            0.0,
            reason=f"Consensus below threshold ({buy_votes:.2f} / {min_threshold:.2f}) [{regime_ctx.regime.value}]",
        )

    def get_decision_attribution(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> Dict[str, Any]:
        """Aggregate attribution across all quantitative sub-strategies with consensus breakdown."""
        if df is not None and not df.empty and "RSI_14" not in df.columns:
            df = ensure_strategy_indicators(df)

        base_attr = super().get_decision_attribution(df, current_idx=current_idx, prediction=prediction)
        sub_list = []
        buy_cnt = 0
        sell_cnt = 0

        for s in self.strategies:
            s_sig = s.generate_signal(df, current_idx=current_idx, prediction=prediction)
            if s_sig.signal_type == SignalType.BUY:
                buy_cnt += 1
            elif s_sig.signal_type == SignalType.SELL:
                sell_cnt += 1
            sub_list.append({
                "strategy_name": s.name,
                "signal_type": s_sig.signal_type.value,
                "strength": s_sig.strength,
                "reason": s_sig.reason,
            })

        base_attr["sub_signals"] = sub_list
        base_attr["buy_count"] = buy_cnt
        base_attr["sell_count"] = sell_cnt
        base_attr["total_strategies"] = len(self.strategies)
        base_attr["consensus_pct"] = round((max(buy_cnt, sell_cnt) / len(self.strategies)) * 100.0, 1)
        return base_attr


@dataclass
class TournamentResult:
    """Performance results for a single strategy in the tournament arena."""
    strategy_name: str
    total_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_equity: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Strategy": self.strategy_name,
            "Return %": f"{self.total_return_pct:+0.2f}%",
            "Alpha %": f"{self.alpha_pct:+0.2f}%",
            "Sharpe": f"{self.sharpe_ratio:.2f}",
            "Win Rate %": f"{self.win_rate:.1f}%",
            "Max DD %": f"{self.max_drawdown_pct:.2f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Total Trades": self.total_trades,
            "Equity ($)": f"${self.final_equity:,.2f}",
        }


class StrategyArena:
    """
    Tournament arena for backtesting and ranking all 6 institutional strategies
    head-to-head on the active asset, plus multi-asset opportunity ranking.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        strategies: Optional[List[BaseStrategy]] = None,
        **kwargs: Any,
    ):
        self.initial_capital = kwargs.get("starting_capital", initial_capital)
        self.strategies: List[BaseStrategy] = strategies or [
            DeepQuantileSniper(),
            InstitutionalWhaleFlow(),
            VolatilityBreakout(),
            StatisticalMeanReversion(),
            NewsCatalystMomentum(),
            EnsembleAlpha(),
        ]

    def run_tournament(
        self,
        df: pd.DataFrame,
        ticker: str = "ASSET",
        prediction: Optional[QuantilePrediction] = None,
        starting_capital: Optional[float] = None,
        **kwargs: Any,
    ) -> List[TournamentResult]:
        """
        Execute parallel backtests of all strategies and return a ranked list of TournamentResult.
        Auto-engineers required technical indicators and evaluates intra-bar High/Low TP/SL brackets.
        """
        results: List[TournamentResult] = []

        if df is None or df.empty or len(df) < 30:
            return results

        # Guarantee all required indicators (RSI_14, EMA_9, EMA_20, ATR_14, BB, VWAP, Whale_Z) exist
        work_df = ensure_strategy_indicators(df)

        cap = starting_capital or kwargs.get("initial_capital", self.initial_capital)
        close_prices = work_df["Close"].values
        benchmark_return = ((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100.0

        step = 1 if len(work_df) <= 260 else max(1, len(work_df) // 200)

        for strat in self.strategies:
            sim = Gauntlet(
                starting_capital=cap,
                slippage_range=(0.0001, 0.0005),
                commission_flat=1.00,
                commission_pct=0.001,
                persist=False,
            )

            for i in range(25, len(work_df), step):
                bar_row = work_df.iloc[i]
                price = float(bar_row["Close"])

                # 1. First evaluate intra-bar High/Low bracket triggers for any existing position
                if ticker in sim.positions:
                    sim.evaluate_bar(ticker, bar_row)

                # 2. Generate strategy signal on bar i
                chunk = work_df.iloc[: i + 1]
                sig = strat.generate_signal(chunk, current_idx=-1, prediction=prediction)

                if sig.signal_type == SignalType.BUY and ticker not in sim.positions:
                    qty = sim.calculate_position_size(
                        price=price,
                        stop_price=sig.stop_price or price * 0.98,
                        risk_pct=0.02,
                    )
                    if qty > 0:
                        sim.submit_bracket_order(
                            TradeOrder(ticker=ticker, side=Side.BUY, quantity=qty, price=price),
                            stop_loss=sig.stop_price,
                            take_profit=sig.target_price,
                        )
                        sim.mark_to_market({ticker: price})
                elif sig.signal_type == SignalType.SELL and ticker in sim.positions:
                    pos = sim.positions[ticker]
                    sim.submit_order(TradeOrder(ticker=ticker, side=Side.SELL, quantity=pos.quantity, price=price))
                    sim.mark_to_market({ticker: price})
                elif ticker not in sim.positions:
                    sim.mark_to_market({ticker: price})

            # Final close of any open positions at end of tournament window
            if ticker in sim.positions:
                pos = sim.positions[ticker]
                sim.submit_order(
                    TradeOrder(ticker=ticker, side=Side.SELL, quantity=pos.quantity, price=float(close_prices[-1]))
                )
                sim.mark_to_market({ticker: float(close_prices[-1])})

            m = sim.metrics
            strat_ret = ((m.total_equity - cap) / cap) * 100.0
            alpha = strat_ret - benchmark_return

            results.append(
                TournamentResult(
                    strategy_name=strat.name,
                    total_return_pct=round(strat_ret, 2),
                    alpha_pct=round(alpha, 2),
                    sharpe_ratio=round(m.sharpe_ratio, 2),
                    max_drawdown_pct=round(m.max_drawdown_pct, 2),
                    win_rate=round(m.win_rate * 100, 1),
                    profit_factor=round(min(m.profit_factor, 99.99), 2),
                    total_trades=m.total_trades,
                    final_equity=round(m.total_equity, 2),
                    equity_curve=list(sim._equity_curve),
                    trades=[f.to_dict() for f in sim.fills],
                )
            )

        # Rank by Sharpe Ratio, then Return
        results.sort(key=lambda r: (r.sharpe_ratio, r.total_return_pct), reverse=True)
        return results

    def run_tournament_df(
        self,
        df: pd.DataFrame,
        ticker: str = "ASSET",
        prediction: Optional[QuantilePrediction] = None,
        starting_capital: Optional[float] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Convenience method returning tournament results as a formatted DataFrame."""
        results = self.run_tournament(df, ticker=ticker, prediction=prediction, starting_capital=starting_capital, **kwargs)
        if not results:
            return pd.DataFrame()
        board = pd.DataFrame([r.to_dict() for r in results])
        board.index = board.index + 1
        board.index.name = "Rank"
        return board

    def rank_watchlist_opportunities(
        self,
        tickers: List[str],
        lake: Any,
        timeframe: str = "1d",
        lookback: str = "3mo",
    ) -> List[Dict[str, Any]]:
        """
        Scan and rank multiple watchlist tickers by Composite Edge Score (0-100)
        to surface the Top 3 Actionable Wealthsimple Setups right now.
        """
        from src.data.market_intel import market_intel as intel_engine
        from src.models.wealthsimple_copier import CanadianAssetProxy

        ensemble = EnsembleAlpha()
        ranked: List[Dict[str, Any]] = []

        for t in tickers:
            try:
                raw_df = lake.fetch(t, timeframe=timeframe, lookback=lookback)
                if raw_df is None or raw_df.empty or len(raw_df) < 25:
                    continue
                feat_df = ensure_strategy_indicators(raw_df)
                intel = intel_engine.get_asset_intelligence(t)
                feat_df["Sentiment"] = float(intel.aggregate_sentiment)

                attr = ensemble.get_decision_attribution(feat_df, current_idx=-1, prediction=None)
                price = float(feat_df["Close"].iloc[-1])
                bar = feat_df.iloc[-1]
                whale_z = float(bar.get("Whale_Z", 0.0))
                rsi = float(bar.get("RSI_14", 50.0))
                ema9 = float(bar.get("EMA_9", price))
                ema20 = float(bar.get("EMA_20", price))

                # Find highest-conviction active BUY strategy if Ensemble is on hold
                best_strat_name = attr["strategy_name"]
                sig_type = attr["signal_type"]
                upside_pct = float(attr["expected_return_pct"])
                rrr = float(attr["rrr"])
                target_price = float(attr["target_price"])
                stop_price = float(attr["stop_price"])

                for sub in self.strategies:
                    if sub.name == "Ensemble Multi-Strategy Alpha":
                        continue
                    s_sig = sub.generate_signal(feat_df, current_idx=-1, prediction=None)
                    if s_sig.signal_type == SignalType.BUY and sig_type != "BUY":
                        sig_type = "BUY"
                        best_strat_name = sub.name
                        target_price = float(s_sig.target_price or price * 1.03)
                        stop_price = float(s_sig.stop_price or price * 0.985)
                        upside_pct = abs(target_price - price) / max(price, 1e-6) * 100.0
                        downside_pct = abs(price - stop_price) / max(price, 1e-6) * 100.0
                        rrr = upside_pct / max(0.01, downside_pct)
                        break

                # Calculate Composite Edge Score (0 - 100)
                # 1. Strategy Consensus & Signal Direction (0 - 30 pts)
                buy_cnt = int(attr.get("buy_count", 0))
                consensus_pts = min(30.0, (buy_cnt / max(1, attr.get("total_strategies", 5))) * 30.0)
                if sig_type == "BUY":
                    consensus_pts = max(consensus_pts, 18.0)
                elif ema9 >= ema20 and 42.0 <= rsi <= 65.0:
                    consensus_pts = max(consensus_pts, 12.0)

                # 2. Risk-to-Reward & Wealthsimple 1.5% Spread Hurdle (0 - 25 pts)
                hurdle_pts = min(15.0, (upside_pct / 3.0) * 15.0) + min(10.0, (rrr / 2.5) * 10.0)

                # 3. Live News & Catalyst Sentiment (0 - 20 pts)
                sent_norm = (float(intel.aggregate_sentiment) + 1.0) / 2.0  # [0, 1]
                news_pts = sent_norm * 20.0

                # 4. Institutional Whale Volume Flow (0 - 15 pts)
                whale_pts = float(np.clip((whale_z + 1.0) / 3.5, 0.0, 1.0)) * 15.0

                # 5. Trend & Regime Alignment (0 - 10 pts)
                regime_name = (
                    ensemble.last_regime_context.regime.value
                    if ensemble.last_regime_context is not None
                    else "RANGING_CHOP"
                )
                regime_pts = 10.0 if regime_name == "BULL_TREND" else (6.0 if regime_name == "RANGING_CHOP" else 1.0)
                if intel.earnings_warning:
                    regime_pts = max(0.0, regime_pts - 5.0)

                composite_score = int(round(np.clip(
                    consensus_pts + hurdle_pts + news_pts + whale_pts + regime_pts, 12.0, 99.0
                )))

                proxy_info = CanadianAssetProxy.resolve(t)
                top_headline = intel.headlines[0].title if intel.headlines else f"{t} active microstructure monitoring"

                ranked.append({
                    "ticker": t,
                    "ws_proxy_ticker": proxy_info["ws_ticker"],
                    "ws_badge": proxy_info["badge"],
                    "composite_edge_score": composite_score,
                    "signal_type": sig_type,
                    "champion_strategy": best_strat_name,
                    "expected_upside_pct": round(upside_pct, 2),
                    "rrr": round(rrr, 2),
                    "sentiment_score": round(float(intel.aggregate_sentiment), 2),
                    "sentiment_label": intel.sentiment_label,
                    "catalyst_badge": intel.dominant_catalyst,
                    "headline_title": top_headline,
                    "whale_z": round(whale_z, 2),
                    "regime": regime_name,
                    "current_price": price,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "earnings_warning": intel.earnings_warning,
                    "earnings_days_away": intel.earnings_days_away,
                })
            except Exception as exc:
                logger.debug("rank_watchlist_opportunities skip %s: %s", t, exc)

        ranked.sort(
            key=lambda item: (
                1 if item["signal_type"] == "BUY" else 0,
                item["composite_edge_score"],
                item["expected_upside_pct"],
            ),
            reverse=True,
        )
        return ranked

