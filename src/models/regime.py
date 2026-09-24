"""
TradeSense Market Regime Classifier & Quantitative Conductor
=============================================================

Identifies real-time macro and micro market regimes:
1. BULL_TREND: Persistent directional upward momentum (ADX >= 22, EMA_9 > EMA_20, +DI > -DI).
2. BEAR_TREND: Downward drift or institutional sell-off (ADX >= 22, EMA_9 < EMA_20, -DI > +DI).
   Enforces 100% Cash Preservation for long-only accounts.
3. RANGING_CHOP: Sideways consolidation / mean-reverting corridor (ADX < 22).
   Strictly suppresses breakout chasing and enables mean-reversion at band extremes.
4. HIGH_VOLATILITY: Extreme price turbulence or volatility expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGING_CHOP = "RANGING_CHOP"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class RegimeContext:
    """Quantitative snapshot of the active market regime and execution gates."""

    regime: MarketRegime
    conviction: float  # 0.0 to 1.0
    adx: float
    plus_di: float
    minus_di: float
    bb_pct_b: float  # 0.0 = Lower Band, 0.5 = Mid, 1.0 = Upper Band
    bb_width_pct: float  # (Upper - Lower) / Mid * 100
    ema_spread_pct: float  # (EMA_9 - EMA_20) / EMA_20 * 100
    rsi: float
    allow_breakout_buys: bool
    allow_mean_reversion_buys: bool
    cash_preservation_active: bool
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "conviction": round(self.conviction, 3),
            "adx": round(self.adx, 2),
            "plus_di": round(self.plus_di, 2),
            "minus_di": round(self.minus_di, 2),
            "bb_pct_b": round(self.bb_pct_b, 4),
            "bb_width_pct": round(self.bb_width_pct, 3),
            "ema_spread_pct": round(self.ema_spread_pct, 3),
            "rsi": round(self.rsi, 2),
            "allow_breakout_buys": self.allow_breakout_buys,
            "allow_mean_reversion_buys": self.allow_mean_reversion_buys,
            "cash_preservation_active": self.cash_preservation_active,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class RegimeClassifier:
    """
    Computes mathematical trend strength (Wilder's ADX / DMI), volatility spread,
    and Bollinger position to classify market regime and gate execution.
    """

    def __init__(
        self,
        adx_trend_threshold: float = 22.0,
        adx_strong_trend_threshold: float = 30.0,
    ):
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_trend_threshold = adx_strong_trend_threshold

    @staticmethod
    def compute_adx_dmi(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute genuine Wilder's DMI (+DI, -DI) and ADX.
        Returns: (ADX, Plus_DI, Minus_DI)
        """
        if len(df) < period + 2:
            s_zero = pd.Series(20.0, index=df.index)
            return s_zero, s_zero, s_zero

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # 1. True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 2. Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s = pd.Series(plus_dm, index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)

        # 3. Smoothed Series (Wilder RMA / EWM)
        tr_smooth = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        plus_smooth = plus_dm_s.ewm(alpha=1.0 / period, adjust=False).mean()
        minus_smooth = minus_dm_s.ewm(alpha=1.0 / period, adjust=False).mean()

        plus_di = 100.0 * (plus_smooth / (tr_smooth + 1e-9))
        minus_di = 100.0 * (minus_smooth / (tr_smooth + 1e-9))

        # 4. DX and ADX
        dx = 100.0 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

        return adx.fillna(20.0), plus_di.fillna(20.0), minus_di.fillna(20.0)

    def classify(self, df: pd.DataFrame, current_idx: int = -1) -> RegimeContext:
        """
        Evaluate market regime for the current bar in df.
        """
        if df.empty or len(df) < 5:
            return RegimeContext(
                regime=MarketRegime.RANGING_CHOP,
                conviction=0.5,
                adx=20.0,
                plus_di=20.0,
                minus_di=20.0,
                bb_pct_b=0.5,
                bb_width_pct=2.0,
                ema_spread_pct=0.0,
                rsi=50.0,
                allow_breakout_buys=False,
                allow_mean_reversion_buys=False,
                cash_preservation_active=False,
                reason="Insufficient data bars for regime classification",
            )

        row = df.iloc[current_idx]
        price = float(row["Close"])

        # Compute dynamic ADX / DMI if not already present
        if "Real_ADX" not in df.columns:
            adx_s, plus_s, minus_s = self.compute_adx_dmi(df)
            adx_val = float(adx_s.iloc[current_idx])
            plus_val = float(plus_s.iloc[current_idx])
            minus_val = float(minus_s.iloc[current_idx])
        else:
            adx_val = float(row.get("Real_ADX", 20.0))
            plus_val = float(row.get("Plus_DI", 20.0))
            minus_val = float(row.get("Minus_DI", 20.0))

        # Indicators
        ema9 = float(row.get("EMA_9", price))
        ema20 = float(row.get("EMA_20", price))
        bb_u = float(row.get("BB_Upper", price * 1.02))
        bb_m = float(row.get("BB_Mid", price))
        bb_l = float(row.get("BB_Lower", price * 0.98))
        rsi = float(row.get("RSI_14", 50.0))
        atr = float(row.get("ATR_14", price * 0.01))

        # Bollinger Metrics
        bb_spread = max(0.01, bb_u - bb_l)
        bb_pct_b = (price - bb_l) / bb_spread
        bb_width_pct = (bb_spread / max(0.01, bb_m)) * 100.0

        # EMA Trend Spread
        ema_spread_pct = ((ema9 - ema20) / max(0.01, ema20)) * 100.0

        # Regime Evaluation Logic
        is_trending = adx_val >= self.adx_trend_threshold
        is_bullish = (ema9 >= ema20) and (plus_val >= minus_val)
        is_bearish = (ema9 < ema20) and (minus_val > plus_val)
        is_high_vol = (atr / max(0.01, price)) > 0.025 or bb_width_pct > 4.5

        if is_high_vol and is_trending:
            regime = MarketRegime.BULL_TREND if is_bullish else MarketRegime.BEAR_TREND
            conviction = min(1.0, adx_val / 50.0)
        elif is_trending and is_bullish:
            regime = MarketRegime.BULL_TREND
            conviction = min(1.0, max(0.60, adx_val / 40.0))
        elif is_trending and is_bearish:
            regime = MarketRegime.BEAR_TREND
            conviction = min(1.0, max(0.60, adx_val / 40.0))
        else:
            regime = MarketRegime.RANGING_CHOP
            conviction = min(1.0, max(0.50, (30.0 - adx_val) / 20.0))

        # Actionable Execution Permissions
        if regime == MarketRegime.BULL_TREND:
            # In a bull trend, breakouts are valid as long as not at an insane extreme (%B <= 0.92)
            allow_breakout = bb_pct_b <= 0.92
            allow_mean_rev = False  # Suppress counter-trend fades
            cash_preserve = False
            reason = f"Bull Trend Confirmed (ADX={adx_val:.1f}, EMA9>EMA20, +DI={plus_val:.1f} > -DI={minus_val:.1f})"
        elif regime == MarketRegime.BEAR_TREND:
            # Bear trend: 100% Cash Preservation for long accounts
            allow_breakout = False
            allow_mean_rev = False
            cash_preserve = True
            reason = f"Bear Trend Detected (ADX={adx_val:.1f}, EMA9<EMA20, -DI={minus_val:.1f} > +DI={plus_val:.1f}) - CASH LOCK"
        else:
            # Ranging Chop: Breakouts are strictly forbidden (prevents buying ceilings)
            allow_breakout = False
            # Mean reversion buy only allowed if at lower boundary and oversold
            allow_mean_rev = (bb_pct_b <= 0.20) and (rsi < 38.0)
            cash_preserve = False
            reason = f"Range-Bound Chop (ADX={adx_val:.1f} < {self.adx_trend_threshold}, Width={bb_width_pct:.2f}%) - Breakouts Suppressed"

        return RegimeContext(
            regime=regime,
            conviction=conviction,
            adx=adx_val,
            plus_di=plus_val,
            minus_di=minus_val,
            bb_pct_b=bb_pct_b,
            bb_width_pct=bb_width_pct,
            ema_spread_pct=ema_spread_pct,
            rsi=rsi,
            allow_breakout_buys=allow_breakout,
            allow_mean_reversion_buys=allow_mean_rev,
            cash_preservation_active=cash_preserve,
            reason=reason,
        )
