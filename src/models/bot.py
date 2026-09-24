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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from src.data.live_feed import LiveMarketFeed, MarketTick
from src.models.ai_engine import QuantilePrediction
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
    ):
        self.feed = feed
        self.config = config or BotConfig(symbol=feed.ticker)
        self.strategy = strategy or EnsembleAlpha()
        self.state = BotState.DISARMED

        # Portfolio and order execution using Gauntlet
        self.gauntlet = Gauntlet(
            starting_capital=self.config.initial_balance,
            persist=persist_portfolio,
        )

        # Telemetry & logs
        self._logs: deque[BotLogEntry] = deque(maxlen=500)
        self._cooldown_counter = 0
        self._total_ticks_processed = 0

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

    def arm(self) -> None:
        """Activate autonomous execution."""
        self.state = BotState.ARMED
        self.log("INFO", f"🤖 Sniper Bot ARMED. Real-time strategy: {self.strategy.name}")

    def disarm(self) -> None:
        """Deactivate autonomous execution; stops new order entry."""
        self.state = BotState.DISARMED
        self.log("INFO", "🛑 Sniper Bot DISARMED. Execution paused.")

    def step(self, prediction: Optional[QuantilePrediction] = None) -> MarketTick:
        """
        Execute one autonomous cycle:
        1. Fetch next market tick from LiveMarketFeed
        2. Evaluate existing open positions against tick (Intra-bar Stop Loss, Take Profit, Trailing Stop)
        3. Whale alert detection
        4. Strategy signal generation
        5. If ARMED and valid signal, size and dispatch bracket order
        """
        tick = self.feed.next_tick()
        self._total_ticks_processed += 1
        current_price = tick.price

        # 1. Update open positions via Gauntlet mark_to_market
        prev_fills_len = len(self.gauntlet.fills)
        self.gauntlet.mark_to_market({self.config.symbol: current_price})

        # Check if any auto-close orders executed during mark_to_market
        if len(self.gauntlet.fills) > prev_fills_len:
            for fill in self.gauntlet.fills[prev_fills_len:]:
                if fill.side == Side.SELL:
                    event = fill.exit_reason or "AUTO_EXIT"
                    self.log(
                        event,
                        f"🎯 Position closed on {event} @ ${fill.fill_price:,.2f} | PnL: ${fill.pnl:,.2f}",
                        price=fill.fill_price,
                        pnl=fill.pnl,
                    )

        # 2. Check Whale Flow
        if tick.is_whale:
            self.log(
                "WHALE_ALERT",
                f"🐋 Institutional Whale Order detected! Volume: {tick.volume:,.0f} | Z-Score: {tick.whale_z:.2f}",
                price=current_price,
            )

        # 3. Strategy Evaluation (if cooldown passed and positions available)
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        if (
            self.state == BotState.ARMED
            and self._cooldown_counter == 0
            and len(self.gauntlet.positions) < self.config.max_open_positions
        ):
            self._evaluate_signals_and_execute(current_price, prediction)

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

            # Position sizing (Risk 2% of portfolio cash)
            risk_capital = self.gauntlet.cash * self.config.risk_per_trade
            qty = max(1, int(risk_capital / risk))

            # Ensure we have enough cash for position
            max_qty = int(self.gauntlet.cash * 0.95 / current_price)
            qty = min(qty, max_qty)
            if qty <= 0:
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
                logger.error("Failed to execute buy order: %s", e)

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
