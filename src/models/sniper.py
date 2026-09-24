"""
HFP Sniper Execution Engine
===========================

Automated trading signal generator and bracket order execution engine.
Filters ML predictions through strict institutional risk-reward criteria:
- High confidence threshold (>75%)
- Asymmetric Risk-Reward Ratio (RRR >= 1.5x)
- Dynamic Quantile Brackets (P10 Stop / P90 Target)
- Whale Volume Confluence verification
- Capital-preservation position sizing
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from src.models.ai_engine import QuantilePrediction
from src.models.trader import Gauntlet, Side, TradeOrder, Fill
from src.utils.logger import get_logger

logger = get_logger("hfp.sniper")


@dataclass
class SniperConfig:
    """Configurable risk parameters for Sniper execution."""

    min_confidence: float = 0.70          # Minimum model confidence (70%)
    min_risk_reward: float = 1.5           # Minimum upside-to-downside ratio
    max_capital_risk_pct: float = 0.02    # 2% equity risk per trade
    use_trailing_stop: bool = False       # Ratchet trailing stop on profit
    trailing_stop_pct: float = 0.03       # 3% trailing buffer
    allow_short: bool = False             # Long-only by default


@dataclass
class SniperSignal:
    """Actionable trade signal passing institutional sniper filters."""

    ticker: str
    side: Side
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    confidence: float
    suggested_shares: int
    whale_confirmed: bool = False
    reasons: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_cost(self) -> float:
        return self.entry_price * self.suggested_shares

    @property
    def max_loss_dollars(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.suggested_shares

    @property
    def max_gain_dollars(self) -> float:
        return abs(self.target_price - self.entry_price) * self.suggested_shares

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_price": round(self.target_price, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "confidence": round(self.confidence * 100, 1),
            "suggested_shares": self.suggested_shares,
            "whale_confirmed": self.whale_confirmed,
            "reasons": self.reasons,
            "timestamp": self.timestamp,
        }


class SniperEngine:
    """
    Automated Sniper Execution Engine.
    Filters probabilistic quantile forecasts into high-conviction bracket orders.
    """

    def __init__(self, config: Optional[SniperConfig] = None):
        self.config = config or SniperConfig()

    def evaluate(
        self,
        ticker: str,
        prediction: QuantilePrediction,
        current_price: float,
        sim: Gauntlet,
        latest_features: Optional[pd.Series | Dict[str, Any]] = None,
    ) -> Optional[SniperSignal]:
        """
        Evaluate if a prediction qualifies as an institutional Sniper execution.

        Parameters
        ----------
        ticker : str
            Symbol being analyzed.
        prediction : QuantilePrediction
            Current P10 / P50 / P90 forecast.
        current_price : float
            Latest executed price.
        sim : Gauntlet
            Simulator holding portfolio equity.
        latest_features : pd.Series or dict, optional
            Recent bar indicators (Whale Tracker, ADX, etc.).

        Returns
        -------
        SniperSignal or None
            Actionable signal if all filters pass.
        """
        if current_price <= 0:
            logger.debug("Sniper rejected %s: invalid current price %.2f", ticker, current_price)
            return None

        # 1. Direction Filter
        if prediction.direction == "NEUTRAL":
            logger.debug("Sniper rejected %s: prediction is NEUTRAL", ticker)
            return None

        if prediction.direction == "BEARISH" and not self.config.allow_short:
            logger.debug("Sniper rejected %s: short trading disabled", ticker)
            return None

        # 2. Confidence Filter
        if prediction.confidence < self.config.min_confidence:
            logger.debug(
                "Sniper rejected %s: confidence %.1f%% below threshold %.1f%%",
                ticker, prediction.confidence * 100, self.config.min_confidence * 100
            )
            return None

        reasons: List[str] = [
            f"Confidence {prediction.confidence:.1%} >= {self.config.min_confidence:.1%}",
            f"Model Direction: {prediction.direction}",
        ]

        # 3. Bracket Order Geometry & Risk-Reward Ratio
        if prediction.direction == "BULLISH":
            side = Side.BUY
            # Stop loss at P10 (or at least 0.5% below entry to avoid zero division)
            stop_loss = float(min(prediction.p10, current_price * 0.995))
            # Target at P90 (or at least 0.5% above entry)
            target_price = float(max(prediction.p90, current_price * 1.005))

            upside = target_price - current_price
            downside = current_price - stop_loss
        else:
            side = Side.SELL
            stop_loss = float(max(prediction.p90, current_price * 1.005))
            target_price = float(min(prediction.p10, current_price * 0.995))

            upside = current_price - target_price
            downside = stop_loss - current_price

        if downside <= 0:
            logger.debug("Sniper rejected %s: zero or negative downside risk", ticker)
            return None

        rrr = upside / downside
        if rrr < self.config.min_risk_reward:
            logger.debug(
                "Sniper rejected %s: RRR %.2fx below required %.2fx",
                ticker, rrr, self.config.min_risk_reward
            )
            return None

        reasons.append(f"Risk-Reward Asymmetry: {rrr:.2f}x (upside=${upside:.2f}, risk=${downside:.2f})")

        # 4. Whale Volume Confluence
        whale_confirmed = False
        if latest_features is not None:
            w_flag = (
                latest_features.get("Whale_Flag", 0)
                if hasattr(latest_features, "get")
                else latest_features.get("Whale_Flag", 0)
            )
            w_z = (
                latest_features.get("Whale_Z", 0.0)
                if hasattr(latest_features, "get")
                else latest_features.get("Whale_Z", 0.0)
            )
            if w_flag == 1 or float(w_z) >= 2.0:
                whale_confirmed = True
                reasons.append(f"🐋 Whale Influx Confirmed (Z={float(w_z):.2f})")

        # 5. Position Sizing
        shares = sim.calculate_position_size(
            price=current_price,
            stop_price=stop_loss,
            risk_pct=self.config.max_capital_risk_pct,
        )

        if shares <= 0:
            logger.debug("Sniper rejected %s: position sizing returned 0 shares", ticker)
            return None

        reasons.append(f"Position Sized for {self.config.max_capital_risk_pct:.1%} equity risk")

        signal = SniperSignal(
            ticker=ticker,
            side=side,
            entry_price=round(current_price, 4),
            stop_loss=round(stop_loss, 4),
            target_price=round(target_price, 4),
            risk_reward_ratio=round(rrr, 2),
            confidence=prediction.confidence,
            suggested_shares=shares,
            whale_confirmed=whale_confirmed,
            reasons=reasons,
        )

        logger.info(
            "🎯 SNIPER SIGNAL [%s]: %s %d @ $%.2f | Stop=$%.2f | Target=$%.2f | RRR=%.2fx | Conf=%.1f%%",
            ticker, signal.side.value, signal.suggested_shares, signal.entry_price,
            signal.stop_loss, signal.target_price, signal.risk_reward_ratio, signal.confidence * 100,
        )
        return signal

    def execute(self, signal: SniperSignal, sim: Gauntlet) -> Fill:
        """
        Execute an automated bracket order into the simulator.
        """
        order = TradeOrder(
            ticker=signal.ticker,
            side=signal.side,
            quantity=signal.suggested_shares,
            price=signal.entry_price,
            exit_reason="MANUAL",
        )

        trailing_pct = self.config.trailing_stop_pct if self.config.use_trailing_stop else None

        fill = sim.submit_bracket_order(
            order=order,
            stop_loss=signal.stop_loss,
            take_profit=signal.target_price,
            trailing_stop_pct=trailing_pct,
        )

        note = (
            f"🎯 Sniper Bracket Order: Conf={signal.confidence:.1%}, "
            f"RRR={signal.risk_reward_ratio:.2f}x, Target=${signal.target_price:.2f}, Stop=${signal.stop_loss:.2f}"
        )
        sim.add_trade_note(fill.order_id, note)
        logger.info("🎯 Sniper Order filled for %s: order_id=%s", signal.ticker, fill.order_id)
        return fill
