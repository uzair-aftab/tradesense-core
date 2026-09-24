"""
TradeSense Autonomous Sniper Trading Bot Engine
=================================================

Provides a fully automated institutional trading agent capable of:
1. Subscribing to live or simulated tick streams (LiveMarketFeed)
2. Evaluating multi-strategy quant signals (EnsembleAlpha, DeepQuantileSniper, WhaleFlow)
3. Enforcing institutional risk management (RRR >= 1.5x, dynamic position sizing)
4. Placing and monitoring bracket orders (Entry, Stop Loss, Take Profit)
5. Actively managing ratcheting dynamic trailing stops
6. Logging tick-by-tick order microstructure events and PnL telemetry
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from src.data.live_feed import LiveMarketFeed, MarketTick
from src.models.ai_engine import QuantilePrediction
from src.models.session_logger import BotSessionLogger, SessionLogManager
from src.models.strategies import (
    BaseStrategy,
    EnsembleAlpha,
    SignalType,
)
from src.models.trader import Gauntlet, Side, TradeOrder
from src.utils.logger import get_logger

logger = get_logger("hfp.bot")


class BotState(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    PAUSED = "PAUSED"


@dataclass
class BotConfig:
    symbol: str = "BTC-CAD"
    initial_balance: float = 100_000.0
    risk_per_trade: float = 0.02          # 2% of portfolio per trade
    min_confidence: float = 0.70          # 70% threshold
    min_rrr: float = 1.50                 # Minimum 1.5x Reward-to-Risk ratio
    max_open_positions: int = 3
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.015      # 1.5% dynamic trailing stop
    cooldown_ticks: int = 3               # Prevent instant re-entry


@dataclass
class BotLogEntry:
    timestamp: str
    event_type: str  # TICK, SIGNAL, ORDER_BUY, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP, WHALE_ALERT, INFO
    message: str
    price: float = 0.0
    pnl: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Timestamp": self.timestamp,
            "Event": self.event_type,
            "Message": self.message,
            "Price": round(self.price, 4),
            "PnL": round(self.pnl, 2),
        }


class AutonomousSniperBot:
    """
    Autonomous Execution Agent connecting real-time streaming market data,
    institutional strategy engines, and the Gauntlet portfolio risk engine.
    """

    def __init__(
        self,
        feed: LiveMarketFeed,
        strategy: Optional[BaseStrategy] = None,
        config: Optional[BotConfig] = None,
        persist_portfolio: bool = False,
        ai_engine: Optional[Any] = None,
    ):
        self.feed = feed
        self.config = config or BotConfig(symbol=feed.ticker)
        self.strategy = strategy or EnsembleAlpha()
        self.ai_engine = ai_engine
        self.state = BotState.DISARMED

        # Portfolio and order execution using Gauntlet
        self.gauntlet = Gauntlet(
            starting_capital=self.config.initial_balance,
            persist=persist_portfolio,
        )

        # Telemetry, logs & risk circuit breakers
        self._logs: deque[BotLogEntry] = deque(maxlen=500)
        self._cooldown_counter = 0
        self._total_ticks_processed = 0
        self._consecutive_losses: int = 0
        self._circuit_breaker_ticks: int = 0
        self._processed_fills_count: int = 0
        self.last_wealthsimple_signal: Optional[Dict[str, Any]] = None


        # Session & Timed Execution Management
        self.session_logger: Optional[BotSessionLogger] = None
        self.session_duration_minutes: Optional[float] = None
        self.session_start_time: Optional[datetime] = None
        self.session_end_time: Optional[datetime] = None
        self.last_completed_session_summary: Optional[Dict[str, Any]] = None
        self._last_execution_event: Dict[str, Any] = {"action": "STANDBY"}
        self._last_signal_type: str = "HOLD"
        self._last_signal_strength: float = 0.0
        self._last_signal_reason: str = "Standby"


        self.log(
            "INFO",
            f"Autonomous Sniper Bot initialized for {self.config.symbol} with {self.strategy.name}",
            price=feed.current_price,
        )

    def log(
        self,
        event_type: str,
        message: str,
        price: float = 0.0,
        pnl: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = BotLogEntry(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
            event_type=event_type,
            message=message,
            price=price,
            pnl=pnl,
            details=details or {},
        )
        self._logs.append(entry)
        logger.info("[%s] %s (Price: %s, PnL: %s)", event_type, message, price, pnl)

    def arm(
        self,
        duration_minutes: Optional[float] = None,
        session_tag: Optional[str] = None,
    ) -> Optional[str]:
        """
        Activate autonomous execution.
        If duration_minutes is specified, arms a timed session, logs every tick and event
        to a dedicated JSONL telemetry file, and automatically disarms when the timer expires.
        """
        self.state = BotState.ARMED
        now = datetime.now(timezone.utc)

        # Snapshot active bot configuration
        config_snap = {
            "symbol": self.config.symbol,
            "strategy": self.strategy.name,
            "risk_per_trade": self.config.risk_per_trade,
            "min_confidence": self.config.min_confidence,
            "min_rrr": self.config.min_rrr,
            "trailing_stop_pct": self.config.trailing_stop_pct,
            "trailing_stop_enabled": self.config.trailing_stop_enabled,
            "cooldown_ticks": self.config.cooldown_ticks,
        }

        if duration_minutes and duration_minutes > 0:
            self.session_duration_minutes = float(duration_minutes)
            self.session_start_time = now
            self.session_end_time = now + timedelta(minutes=duration_minutes)
            dur_label = f"{duration_minutes:.0f}m"
        else:
            self.session_duration_minutes = None
            self.session_start_time = now
            self.session_end_time = None
            dur_label = "indefinite"

        self.session_logger = BotSessionLogger(
            symbol=self.config.symbol,
            target_duration_minutes=self.session_duration_minutes or 0.0,
            initial_equity=self.gauntlet.equity,
            initial_price=self.feed.current_price,
            config_snapshot=config_snap,
            session_tag=session_tag or dur_label,
        )

        end_str = (
            self.session_end_time.strftime("%H:%M:%S UTC")
            if self.session_end_time
            else "Manual Stop"
        )
        msg = (
            f"🤖 Sniper Bot ARMED ({dur_label} session, ends {end_str}). "
            f"Logging to: {self.session_logger.jsonl_path.name}"
        )
        self.log("ARM", msg, price=self.feed.current_price)
        self._last_execution_event = {"action": "SESSION_ARMED", "duration": dur_label}
        return self.session_logger.session_id

    def disarm(self, reason: str = "MANUAL_DISARM") -> Optional[Dict[str, Any]]:
        """
        Deactivate autonomous execution; stops order entry and finalizes
        active session log files with complete post-trade analytics.
        """
        self.state = BotState.DISARMED
        summary = None

        if self.session_logger is not None:
            summary = self.session_logger.finalize(
                current_equity=self.gauntlet.equity,
                reason=reason,
            )
            self.last_completed_session_summary = summary
            self.session_logger = None
            self.session_duration_minutes = None
            self.session_start_time = None
            self.session_end_time = None

        self._last_execution_event = {"action": "SESSION_DISARMED", "reason": reason}
        self.log("DISARM", f"🛑 Sniper Bot DISARMED ({reason}). Execution paused.")
        return summary

    def get_session_status(self) -> Dict[str, Any]:
        """Return real-time telemetry for the active timed session."""
        if not self.session_logger:
            return {
                "active": False,
                "is_timed": False,
                "duration_minutes": 0.0,
                "remaining_seconds": 0.0,
                "remaining_formatted": "00:00",
                "progress_pct": 0.0,
                "session_id": None,
                "file_path": None,
                "ticks_processed": 0,
                "trades_count": 0,
            }

        now = datetime.now(timezone.utc)
        elapsed_sec = (now - self.session_start_time).total_seconds() if self.session_start_time else 0.0
        total_sec = (self.session_duration_minutes * 60.0) if self.session_duration_minutes else 0.0

        if total_sec > 0:
            remaining_sec = max(0.0, total_sec - elapsed_sec)
            progress_pct = min(100.0, (elapsed_sec / total_sec) * 100.0)
            rem_m = int(remaining_sec // 60)
            rem_s = int(remaining_sec % 60)
            rem_str = f"{rem_m:02d}:{rem_s:02d}"
        else:
            remaining_sec = 0.0
            progress_pct = 100.0
            rem_str = "Indefinite"

        return {
            "active": True,
            "is_timed": bool(self.session_duration_minutes and self.session_duration_minutes > 0),
            "duration_minutes": self.session_duration_minutes or 0.0,
            "elapsed_seconds": elapsed_sec,
            "remaining_seconds": remaining_sec,
            "remaining_formatted": rem_str,
            "progress_pct": progress_pct,
            "session_id": self.session_logger.session_id,
            "file_path": str(self.session_logger.jsonl_path),
            "report_path": str(self.session_logger.report_path),
            "ticks_processed": self.session_logger.ticks_processed,
            "trades_count": len(self.session_logger.trades),
        }

    def step(self, prediction: Optional[QuantilePrediction] = None) -> MarketTick:
        """
        Execute one autonomous cycle:
        1. Check timed session expiry (auto-disarms if duration reached)
        2. Fetch next market tick from LiveMarketFeed
        3. Evaluate existing open positions against tick (Intra-bar Stop Loss, Take Profit, Trailing Stop)
        4. Whale alert detection
        5. Strategy signal generation & order placement
        6. Stream high-density telemetry snapshot to session logger file
        """
        # 1. Check Timed Session Expiration
        if (
            self.state == BotState.ARMED
            and self.session_end_time is not None
            and datetime.now(timezone.utc) >= self.session_end_time
        ):
            self.disarm(reason=f"TIMESLOT_EXPIRED_{int(self.session_duration_minutes or 0)}M")

        # 2. Advance Tick
        tick = self.feed.next_tick()
        self._total_ticks_processed += 1
        current_price = tick.price

        # 3. Update open positions via Gauntlet mark_to_market
        self.gauntlet.mark_to_market({self.config.symbol: current_price})

        # Check if any auto-close or exit orders executed
        if len(self.gauntlet.fills) > self._processed_fills_count:
            for fill in self.gauntlet.fills[self._processed_fills_count:]:
                if fill.side == Side.SELL:
                    event = fill.exit_reason or "AUTO_EXIT"
                    if self.session_logger:
                        self.session_logger.record_exit(fill)

                    self._last_execution_event = {
                        "action": event,
                        "side": "SELL",
                        "quantity": fill.quantity,
                        "price": fill.fill_price,
                        "pnl": fill.pnl,
                        "exit_reason": event,
                    }
                    self.log(
                        event,
                        f"🎯 Position closed on {event} @ ${fill.fill_price:,.2f} | PnL: ${fill.pnl:,.2f}",
                        price=fill.fill_price,
                        pnl=fill.pnl,
                    )

                    # Consecutive Loss Circuit Breaker
                    if "STOP_LOSS" in event or fill.pnl < 0:
                        self._consecutive_losses += 1
                        if self._consecutive_losses >= 2:
                            self._circuit_breaker_ticks = 60
                            self.log(
                                "CIRCUIT_BREAKER",
                                "🛑 Circuit Breaker Triggered (2 consecutive stop-outs)! Halting execution for 60 ticks (~2 min).",
                                price=fill.fill_price,
                                pnl=fill.pnl,
                            )
                    elif "TAKE_PROFIT" in event or fill.pnl > 0:
                        self._consecutive_losses = 0

            self._processed_fills_count = len(self.gauntlet.fills)


        # 4. Check Whale Flow
        if tick.is_whale:
            self.log(
                "WHALE_ALERT",
                f"🐋 Institutional Whale Order detected! Volume: {tick.volume:,.0f} | Z-Score: {tick.whale_z:.2f}",
                price=current_price,
            )

        # 5. Circuit Breaker Countdown & Cooldowns
        if self._circuit_breaker_ticks > 0:
            self._circuit_breaker_ticks -= 1
            if self._circuit_breaker_ticks % 15 == 0 and self._circuit_breaker_ticks > 0:
                self.log(
                    "CIRCUIT_BREAKER",
                    f"🛡️ Circuit Breaker Active: {self._circuit_breaker_ticks} ticks remaining before trading resumes.",
                    price=current_price,
                )

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # 6. Auto-Generate Quantile Prediction if not passed
        if prediction is None:
            if self.ai_engine is not None and getattr(self.ai_engine, "_trained", False):
                try:
                    prediction = self.ai_engine.predict(self.feed.streaming_df)
                except Exception:
                    pass
            if prediction is None and len(self.feed.streaming_df) >= 5:
                df = self.feed.streaming_df
                closes = df["Close"]
                atr = float(df.get("ATR_14", pd.Series(current_price * 0.01, index=df.index)).iloc[-1])
                p50 = float(df.get("EMA_20", pd.Series(current_price, index=df.index)).iloc[-1])
                prediction = QuantilePrediction(
                    p10=round(p50 - 1.645 * atr, 4),
                    p50=round(p50, 4),
                    p90=round(p50 + 1.645 * atr, 4),
                    confidence=0.80,
                    current_price=current_price,
                )

        # 7. Strategy Evaluation (if unblocked, cooldown passed, and positions available)
        if (
            self.state == BotState.ARMED
            and self._circuit_breaker_ticks == 0
            and self._cooldown_counter == 0
            and len(self.gauntlet.positions) < self.config.max_open_positions
        ):
            self._evaluate_signals_and_execute(current_price, prediction)
        elif self.state == BotState.ARMED and self._total_ticks_processed % 8 == 0:
            open_count = len(self.gauntlet.positions)
            cb_msg = f" (Circuit Breaker: {self._circuit_breaker_ticks} ticks)" if self._circuit_breaker_ticks > 0 else ""
            self.log(
                "INFO",
                f"🤖 Sniper Bot scanning order flow ({open_count}/{self.config.max_open_positions} positions open){cb_msg}",
                price=current_price,
            )


        # 6. Stream High-Density Tick Snapshot to Active Session File
        if self.session_logger is not None:
            df = self.feed.streaming_df
            indicators_dict = df.iloc[-1].to_dict() if not df.empty else {}

            sub_signals = []
            if hasattr(self.strategy, "last_sub_signals") and self.strategy.last_sub_signals:
                for s in self.strategy.last_sub_signals:
                    sub_signals.append({
                        "strategy": s.strategy_name,
                        "signal": s.signal_type.value,
                        "strength": round(float(s.strength), 4),
                        "target": round(float(s.target_price), 4) if s.target_price else None,
                        "stop": round(float(s.stop_price), 4) if s.stop_price else None,
                        "reason": s.reason,
                    })

            regime_val = "BULL_TREND"
            if hasattr(self.strategy, "last_regime_context") and self.strategy.last_regime_context:
                regime_val = self.strategy.last_regime_context.regime.value

            decision_snapshot = {
                "strategy_name": self.strategy.name,
                "consensus_signal": self._last_signal_type,
                "consensus_strength": self._last_signal_strength,
                "consensus_reason": self._last_signal_reason,
                "regime": regime_val,
                "circuit_breaker_ticks": self._circuit_breaker_ticks,
                "consecutive_losses": self._consecutive_losses,
                "sub_strategy_signals": sub_signals,
            }


            positions_snapshot = []
            for sym, pos in self.gauntlet.positions.items():
                unrealized = (current_price - pos.avg_cost) * pos.quantity
                t_stop = self.gauntlet._trailing_stops.get(sym, {}).get("stop_price", None)
                positions_snapshot.append({
                    "symbol": sym,
                    "quantity": round(pos.quantity, 6),
                    "avg_cost": round(pos.avg_cost, 4),
                    "current_price": round(current_price, 4),
                    "unrealized_pnl": round(unrealized, 2),
                    "trailing_stop": round(t_stop, 4) if t_stop else None,
                })

            portfolio_snapshot = {
                "cash": round(self.gauntlet.cash, 2),
                "equity": round(self.gauntlet.equity, 2),
                "open_positions_count": len(self.gauntlet.positions),
                "positions": positions_snapshot,
                "realized_pnl": round(self.gauntlet.metrics.realized_pnl, 2),
                "unrealized_pnl": round(self.gauntlet.metrics.unrealized_pnl, 2),
            }

            self.session_logger.log_tick_snapshot(
                tick=tick,
                df_indicators=indicators_dict,
                prediction=prediction,
                decision=decision_snapshot,
                execution_event=self._last_execution_event,
                portfolio=portfolio_snapshot,
            )
            self._last_execution_event = {"action": "TICK"}

        return tick

    def _evaluate_signals_and_execute(
        self,
        current_price: float,
        prediction: Optional[QuantilePrediction] = None,
    ) -> None:
        """
        Query strategy engine, validate RRR and confidence, and dispatch bracket orders.
        """
        df = self.feed.streaming_df
        if df.empty or len(df) < 5:
            return

        signal = self.strategy.generate_signal(
            df=df,
            current_idx=-1,
            prediction=prediction,
        )

        self._last_signal_type = signal.signal_type.value
        self._last_signal_strength = float(signal.strength)
        self._last_signal_reason = signal.reason

        if self._circuit_breaker_ticks > 0:
            return

        if signal.signal_type == SignalType.BUY and signal.strength >= self.config.min_confidence:
            stop_loss = signal.stop_price or (current_price * 0.98)
            take_profit = signal.target_price or (current_price * 1.04)

            risk = current_price - stop_loss
            reward = take_profit - current_price

            if risk <= 0:
                return

            rrr = reward / risk
            if rrr < self.config.min_rrr:
                return

            reward_pct = (reward / max(0.01, current_price)) * 100.0
            # Wealthsimple fee hurdle filter: minimum target must be at least 1.0% to beat spreads
            if reward_pct < 1.0:
                self.log(
                    "INFO",
                    f"Order skipped: Target distance too tight for fee hurdle (+{reward_pct:.2f}% < 1.0%)",
                    price=current_price,
                )
                return

            # Fractional position sizing based on risk capital
            risk_capital = self.gauntlet.cash * self.config.risk_per_trade
            raw_qty = risk_capital / risk
            max_qty = (self.gauntlet.cash * 0.95) / current_price
            qty = min(raw_qty, max_qty)

            if current_price >= 1000.0:
                qty = round(qty, 4)
            elif current_price >= 10.0:
                qty = round(qty, 2)
            else:
                qty = round(qty, 1)

            if qty <= 0 or (qty * current_price) < 5.0:
                min_fractional = round(min(100.0 / current_price, max_qty), 4 if current_price >= 1000 else 2)
                if min_fractional > 0 and (min_fractional * current_price) <= self.gauntlet.cash * 0.95:
                    qty = min_fractional
                else:
                    self.log("INFO", f"Order skipped: Insufficient capital for minimum lot size @ ${current_price:,.2f}", price=current_price)
                    return

            trailing_pct = self.config.trailing_stop_pct if self.config.trailing_stop_enabled else None


            try:
                order = TradeOrder(
                    ticker=self.config.symbol,
                    side=Side.BUY,
                    quantity=qty,
                    price=current_price,
                )
                fill = self.gauntlet.submit_bracket_order(
                    order=order,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_pct=trailing_pct,
                )

                if self.session_logger:
                    self.session_logger.record_entry(fill)

                self._processed_fills_count = len(self.gauntlet.fills)


                self._last_execution_event = {
                    "action": "ORDER_BUY",
                    "side": "BUY",
                    "quantity": fill.quantity,
                    "price": fill.fill_price,
                    "order_id": fill.order_id,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "rrr": rrr,
                }

                ws_ticker = "BTCC.TO" if "BTC" in self.config.symbol else ("ETHX.TO" if "ETH" in self.config.symbol else self.config.symbol)
                regime_str = "BULL_TREND"
                if hasattr(self.strategy, "last_regime_context") and self.strategy.last_regime_context:
                    regime_str = self.strategy.last_regime_context.regime.value

                self.last_wealthsimple_signal = {
                    "symbol": self.config.symbol,
                    "ws_ticker": ws_ticker,
                    "action": "BUY",
                    "entry_price": round(fill.fill_price, 2),
                    "take_profit": round(take_profit, 2),
                    "stop_loss": round(stop_loss, 2),
                    "rrr": round(rrr, 2),
                    "reward_pct": round(reward_pct, 2),
                    "risk_pct": round(((current_price - stop_loss) / max(0.01, current_price)) * 100.0, 2),
                    "confidence": round(float(signal.strength) * 100.0, 1),
                    "regime": regime_str,
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                }

                self._cooldown_counter = self.config.cooldown_ticks
                self.log(
                    "ORDER_BUY",
                    f"🟢 Executed BUY {fill.quantity} {self.config.symbol} @ ${fill.fill_price:,.2f} "
                    f"| TP: ${take_profit:,.2f} | SL: ${stop_loss:,.2f} (RRR: {rrr:.2f}x)",
                    price=fill.fill_price,
                    details={
                        "order_id": fill.order_id,
                        "quantity": fill.quantity,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "rrr": rrr,
                        "strategy": self.strategy.name,
                    },
                )

            except Exception as e:
                self.log("ERROR", f"Execution error: {e}", price=current_price)
                logger.error("Failed to execute buy order: %s", e)

    @property
    def current_regime(self) -> str:
        """Return current market regime string from active strategy if available."""
        if hasattr(self.strategy, "last_regime_context") and self.strategy.last_regime_context:
            return self.strategy.last_regime_context.regime.value
        return "BULL_TREND"

    def get_metrics(self) -> Dict[str, Any]:
        """Compute key portfolio performance metrics."""
        m = self.gauntlet.metrics
        starting = self.gauntlet.starting_capital
        total_ret = ((m.total_equity - starting) / starting * 100.0) if starting > 0 else 0.0

        return {
            "Total Return %": round(total_ret, 2),
            "Portfolio Value": round(m.total_equity, 2),
            "Cash Balance": round(m.cash, 2),
            "Open Positions": len(self.gauntlet.positions),
            "Total Trades": m.total_trades,
            "Win Rate %": round(m.win_rate, 1),
            "Profit Factor": round(m.profit_factor, 2),
            "Sharpe Ratio": round(m.sharpe_ratio, 2),
            "Max Drawdown %": round(m.max_drawdown_pct, 2),
            "State": self.state.value,
            "Ticks Processed": self._total_ticks_processed,
            "Regime": self.current_regime,
            "CircuitBreakerTicks": self._circuit_breaker_ticks,
        }

    def get_recent_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in list(self._logs)[-count:]]

    def get_trades_df(self) -> pd.DataFrame:
        """Return closed trade history as DataFrame."""
        records = []
        for fill in self.gauntlet.fills:
            if fill.side == Side.SELL:
                records.append({
                    "Order ID": fill.order_id[:8],
                    "Symbol": fill.ticker,
                    "Side": fill.side.value,
                    "Quantity": fill.quantity,
                    "Fill Price": round(fill.fill_price, 2),
                    "PnL ($)": round(fill.pnl, 2),
                    "Exit Reason": fill.exit_reason,
                    "Timestamp": fill.timestamp,
                })
        return pd.DataFrame(records)

    def export_trades_csv(self) -> str:
        """Export closed trades as CSV string."""
        df = self.get_trades_df()
        if df.empty:
            return "Order ID,Symbol,Side,Quantity,Fill Price,PnL ($),Exit Reason,Timestamp\n"
        return df.to_csv(index=False)
