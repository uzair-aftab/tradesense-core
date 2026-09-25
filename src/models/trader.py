"""
HFP Trader — "The Gauntlet" Event-Driven Simulator
====================================================

Institutional-grade paper-trading and backtesting engine with
realistic market-friction modelling.

Realism Constraints
-------------------
* **Slippage** — random 0.01 %–0.05 % cost per trade (market impact)
* **Commission** — $1.00 flat OR 0.1 % per trade (whichever is larger)
* **Latency** — optional simulated execution delay

Tracking
--------
Real-time metrics computed after every trade:
* Sharpe Ratio (annualised, risk-free = 0)
* Max Drawdown (peak-to-trough %)
* Win / Loss Ratio
* Profit Factor
* Cumulative P&L curve

Position Sizing
---------------
Built-in Kelly-criterion and fixed-risk (% of equity) calculators
that integrate with the AI Engine's predicted P10 (stop) level.

Usage
-----
    from src.models.trader import Gauntlet, TradeOrder, Side

    sim = Gauntlet(starting_capital=10_000)
    sim.submit_order(TradeOrder(
        ticker="AAPL", side=Side.BUY, quantity=10, price=152.50
    ))
    sim.mark_to_market({"AAPL": 155.00})
    print(sim.metrics)
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.config import config
from src.utils.logger import get_logger

logger = get_logger("hfp.trader")

# Portfolio persistence path
PORTFOLIO_PATH = Path(config.PAPER_PORTFOLIO_PATH)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Enums & Data Classes                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class TradeOrder:
    """Inbound order request."""

    ticker: str
    side: Side
    quantity: float
    price: float  # intended limit / market price
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    exit_reason: str = "MANUAL"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Fill:
    """Executed trade record (after slippage + commission)."""

    order_id: str
    ticker: str
    side: Side
    quantity: float
    intended_price: float
    fill_price: float  # after slippage
    commission: float
    slippage_pct: float
    total_cost: float  # quantity * fill_price + commission
    pnl: float = 0.0  # realised P&L (set on closing trades)
    exit_reason: str = "MANUAL"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        return d


@dataclass
class Position:
    """Open position tracker."""

    ticker: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "unrealised_pnl_pct": round(self.unrealised_pnl_pct, 2),
        }


@dataclass
class PortfolioMetrics:
    """Real-time portfolio analytics."""

    total_equity: float = 0.0
    cash: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    @property
    def realized_pnl(self) -> float:
        return self.realised_pnl

    @property
    def unrealized_pnl(self) -> float:
        return self.unrealised_pnl

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Gauntlet — The Simulator                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class Gauntlet:
    """
    Event-driven paper-trading simulator with institutional realism.

    Parameters
    ----------
    starting_capital : float
        Initial cash balance (default $10,000).
    slippage_range : tuple[float, float]
        Min / max random slippage per trade as a fraction (default 0.01 %–0.05 %).
    commission_flat : float
        Minimum flat commission per trade (default $1.00).
    commission_pct : float
        Percentage commission per trade (default 0.1 %).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calc (default 0.0).
    seed : int | None
        Random seed for reproducible slippage.
    """

    def __init__(
        self,
        starting_capital: float = 10_000.0,
        slippage_range: Tuple[float, float] = (0.0001, 0.0005),
        commission_flat: float = 1.00,
        commission_pct: float = 0.001,
        risk_free_rate: float = 0.0,
        seed: Optional[int] = None,
        persist: bool = True,
        user_id: Optional[int] = None,
    ):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.slippage_range = slippage_range
        self.commission_flat = commission_flat
        self.commission_pct = commission_pct
        self.risk_free_rate = risk_free_rate
        self.persist = persist
        self.user_id = user_id

        self._rng = random.Random(seed)
        self.positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []
        self._equity_curve: List[float] = [starting_capital]
        self._daily_returns: List[float] = []
        self._stop_losses: Dict[str, float] = {}   # ticker → stop price
        self._take_profits: Dict[str, float] = {}  # ticker → target price
        self._trailing_stops: Dict[str, Dict[str, float]] = {}  # ticker → {trail_pct, peak_price, stop_price}
        self._trade_notes: Dict[str, str] = {}     # order_id → note

        # Load existing portfolio: DB if user_id given, else JSON fallback
        if persist:
            if self.user_id is not None:
                self._load_from_db()
            else:
                self._load_portfolio()

        logger.info(
            "Gauntlet initialised: $%.2f (user_id=%s)  slip=[%.4f%%–%.4f%%]  comm=$%.2f/%.2f%%",
            self.cash,
            str(user_id),
            slippage_range[0] * 100,
            slippage_range[1] * 100,
            commission_flat,
            commission_pct * 100,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Order Execution
    # ══════════════════════════════════════════════════════════════════════

    def submit_order(self, order: TradeOrder) -> Fill:
        """
        Execute an order with realistic slippage and commission.

        Parameters
        ----------
        order : TradeOrder
            Buy or sell order with intended price.

        Returns
        -------
        Fill
            Executed trade record.

        Raises
        ------
        ValueError
            If selling more shares than held, or insufficient cash.
        """
        # ── Slippage ──────────────────────────────────────────────────
        slip_pct = self._rng.uniform(*self.slippage_range)
        if order.side == Side.BUY:
            fill_price = order.price * (1 + slip_pct)  # pay more
        else:
            fill_price = order.price * (1 - slip_pct)  # receive less

        # ── Validation ────────────────────────────────────────────────
        if order.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {order.quantity}")

        trade_value = order.quantity * fill_price
        commission = max(self.commission_flat, trade_value * self.commission_pct)

        if order.side == Side.BUY:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                raise ValueError(
                    f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}"
                )
        else:
            pos = self.positions.get(order.ticker)
            if pos is None or pos.quantity < order.quantity - 1e-6:
                held = pos.quantity if pos else 0.0
                raise ValueError(
                    f"Cannot sell {order.quantity} {order.ticker} — only hold {held}"
                )

        # ── Execute ───────────────────────────────────────────────────
        pnl = 0.0

        if order.side == Side.BUY:
            self.cash -= trade_value + commission
            if order.ticker in self.positions:
                pos = self.positions[order.ticker]
                total_qty = pos.quantity + order.quantity
                pos.avg_cost = (
                    (pos.avg_cost * pos.quantity + fill_price * order.quantity)
                    / total_qty
                )
                pos.quantity = round(total_qty, 6)
            else:
                self.positions[order.ticker] = Position(
                    ticker=order.ticker,
                    quantity=round(order.quantity, 6),
                    avg_cost=fill_price,
                    current_price=fill_price,
                )
        else:
            # SELL
            pos = self.positions[order.ticker]
            pnl = (fill_price - pos.avg_cost) * order.quantity - commission
            self.cash += trade_value - commission
            pos.quantity = round(pos.quantity - order.quantity, 6)
            if pos.quantity <= 1e-6:
                del self.positions[order.ticker]

        fill = Fill(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            intended_price=order.price,
            fill_price=round(fill_price, 4),
            commission=round(commission, 4),
            slippage_pct=round(slip_pct * 100, 4),
            total_cost=round(trade_value + commission, 2),
            pnl=round(pnl, 2),
            exit_reason=order.exit_reason,
        )
        self.fills.append(fill)

        # Persist trade fill to DB if user_id present
        if self.user_id is not None:
            try:
                from src.data.database import get_database
                db = get_database()
                db.record_user_trade(
                    user_id=self.user_id,
                    order_id=fill.order_id,
                    ticker=fill.ticker,
                    side=fill.side.value,
                    quantity=fill.quantity,
                    fill_price=fill.fill_price,
                    total_cost=fill.total_cost,
                    pnl=fill.pnl,
                    exit_reason=fill.exit_reason,
                )
            except Exception as e:
                logger.warning("Could not persist trade fill to DB: %s", e)

        # Auto-save portfolio
        if self.persist:
            self._save_portfolio()

        logger.info(
            "%s %g %s @ $%.2f (slip=%.4f%%, comm=$%.2f, P&L=$%.2f)",
            fill.side.value,
            fill.quantity,
            fill.ticker,
            fill.fill_price,
            fill.slippage_pct,
            fill.commission,
            fill.pnl,
        )
        return fill

    # ══════════════════════════════════════════════════════════════════════
    #  Mark to Market
    # ══════════════════════════════════════════════════════════════════════

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """
        Update all open positions to current market prices and
        record the equity snapshot for Sharpe / drawdown tracking.

        Parameters
        ----------
        prices : dict[str, float]
            Ticker → current price mapping.

        Returns
        -------
        float
            Total portfolio equity after update.
        """
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.current_price = prices[ticker]
                # Ratchet trailing stops if any
                self.update_trailing_stops(ticker, prices[ticker])

        # ── Auto stop-loss / trailing-stop / take-profit execution ──────
        tickers_to_close = []
        for ticker, pos in self.positions.items():
            price = pos.current_price
            effective_stop = None
            stop_reason = "STOP_LOSS"
            if ticker in self._stop_losses:
                effective_stop = self._stop_losses[ticker]
            if ticker in self._trailing_stops:
                ts_stop = self._trailing_stops[ticker]["stop_price"]
                if effective_stop is None or ts_stop > effective_stop:
                    effective_stop = ts_stop
                    stop_reason = "TRAILING_STOP"

            if effective_stop is not None and price <= effective_stop:
                logger.warning("🛑 %s triggered for %s @ $%.2f (stop=$%.2f)",
                               stop_reason, ticker, price, effective_stop)
                tickers_to_close.append((ticker, pos.quantity, price, stop_reason))
            elif ticker in self._take_profits and price >= self._take_profits[ticker]:
                logger.info("🎯 Take-profit triggered for %s @ $%.2f (target=$%.2f)",
                            ticker, price, self._take_profits[ticker])
                tickers_to_close.append((ticker, pos.quantity, price, "TAKE_PROFIT"))

        for ticker, qty, price, reason in tickers_to_close:
            try:
                self.submit_order(TradeOrder(
                    ticker=ticker, side=Side.SELL, quantity=qty, price=price, exit_reason=reason,
                ))
                self._stop_losses.pop(ticker, None)
                self._trailing_stops.pop(ticker, None)
                self._take_profits.pop(ticker, None)
            except ValueError as e:
                logger.error("Auto-close failed for %s: %s", ticker, e)

        equity = self.total_equity
        self._equity_curve.append(equity)

        if len(self._equity_curve) >= 2:
            prev = self._equity_curve[-2]
            daily_ret = (equity - prev) / prev if prev != 0 else 0.0
            self._daily_returns.append(daily_ret)

        return equity

    def evaluate_bar(self, ticker: str, bar: Any) -> float:
        """
        Evaluate realistic intra-bar High/Low price action against bracket orders.
        Triggers exits if the High or Low penetrated target or stop levels during
        the bar, rather than waiting for the Close price.

        Parameters
        ----------
        ticker : str
            Symbol to evaluate.
        bar : dict or pd.Series
            OHLCV bar data with 'High', 'Low', 'Close', and optional 'Open'.

        Returns
        -------
        float
            Updated portfolio equity.
        """
        if ticker not in self.positions:
            close_p = float(bar.get("Close", bar.get("close", 0.0))) if hasattr(bar, "get") else float(bar["Close"])
            return self.mark_to_market({ticker: close_p}) if close_p > 0 else self.total_equity

        pos = self.positions[ticker]
        qty = pos.quantity

        def _val(k1: str, k2: str, d: float) -> float:
            if hasattr(bar, "get"):
                return float(bar.get(k1, bar.get(k2, d)))
            if k1 in bar:
                return float(bar[k1])
            if k2 in bar:
                return float(bar[k2])
            return d

        open_p = _val("Open", "open", pos.current_price)
        high_p = _val("High", "high", open_p)
        low_p = _val("Low", "low", open_p)
        close_p = _val("Close", "close", open_p)

        # Update current price
        pos.current_price = close_p

        is_bullish = close_p >= open_p

        def _get_effective_stop() -> Tuple[Optional[float], str]:
            eff = None
            reason = "STOP_LOSS"
            if ticker in self._stop_losses:
                eff = self._stop_losses[ticker]
            if ticker in self._trailing_stops:
                ts = self._trailing_stops[ticker]["stop_price"]
                if eff is None or ts > eff:
                    eff = ts
                    reason = "TRAILING_STOP"
            return eff, reason

        stop_hit = False
        tp_hit = False
        effective_stop = None
        stop_reason = "STOP_LOSS"
        take_profit = self._take_profits.get(ticker)

        if is_bullish:
            # Bullish sequence: Open -> Low -> High -> Close
            # 1. Test low against existing stop before ratcheting
            effective_stop, stop_reason = _get_effective_stop()
            if effective_stop is not None and low_p <= effective_stop:
                stop_hit = True
            else:
                # 2. Ratchet trailing stop to bar high
                self.update_trailing_stops(ticker, high_p)
                effective_stop, stop_reason = _get_effective_stop()
                # 3. Test high against take profit
                if take_profit is not None and high_p >= take_profit:
                    tp_hit = True
        else:
            # Bearish sequence: Open -> High -> Low -> Close
            # 1. Ratchet trailing stop to bar high
            self.update_trailing_stops(ticker, high_p)
            effective_stop, stop_reason = _get_effective_stop()
            if take_profit is not None and high_p >= take_profit:
                tp_hit = True
            # 2. Test low against updated stop
            if not tp_hit and effective_stop is not None and low_p <= effective_stop:
                stop_hit = True

        if stop_hit:
            exec_price = min(open_p, effective_stop) if open_p < effective_stop else effective_stop
            logger.warning("🛑 Intra-bar %s triggered for %s @ $%.2f (Low=$%.2f, Stop=$%.2f)",
                           stop_reason, ticker, exec_price, low_p, effective_stop)
            try:
                self.submit_order(TradeOrder(
                    ticker=ticker, side=Side.SELL, quantity=qty, price=exec_price, exit_reason=stop_reason,
                ))
            except ValueError as e:
                logger.error("Intra-bar stop execution failed for %s: %s", ticker, e)
            self._stop_losses.pop(ticker, None)
            self._trailing_stops.pop(ticker, None)
            self._take_profits.pop(ticker, None)

        elif tp_hit:
            exec_price = max(open_p, take_profit) if open_p > take_profit else take_profit
            logger.info("🎯 Intra-bar Take-profit triggered for %s @ $%.2f (High=$%.2f, Target=$%.2f)",
                        ticker, exec_price, high_p, take_profit)
            try:
                self.submit_order(TradeOrder(
                    ticker=ticker, side=Side.SELL, quantity=qty, price=exec_price, exit_reason="TAKE_PROFIT",
                ))
            except ValueError as e:
                logger.error("Intra-bar take-profit execution failed for %s: %s", ticker, e)
            self._stop_losses.pop(ticker, None)
            self._trailing_stops.pop(ticker, None)
            self._take_profits.pop(ticker, None)

        return self.mark_to_market({ticker: close_p})

    def submit_bracket_order(
        self,
        order: TradeOrder,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
    ) -> Fill:
        """
        Execute an order and attach bracket conditions (Stop-Loss, Take-Profit, Trailing-Stop).
        """
        fill = self.submit_order(order)
        if order.side == Side.BUY:
            if stop_loss is not None and stop_loss > 0:
                self.set_stop_loss(order.ticker, stop_loss)
            if take_profit is not None and take_profit > 0:
                self.set_take_profit(order.ticker, take_profit)
            if trailing_stop_pct is not None and trailing_stop_pct > 0:
                self.set_trailing_stop(order.ticker, trailing_stop_pct, current_price=fill.fill_price)
        return fill

    # ══════════════════════════════════════════════════════════════════════
    #  Position Sizing Calculators
    # ══════════════════════════════════════════════════════════════════════

    def calculate_position_size(
        self,
        price: float,
        stop_price: float,
        risk_pct: float = 0.02,
        allow_fractional: bool = True,
    ) -> float:
        """
        Fixed-risk position sizer.

        Calculates how many shares to buy so that if the price drops
        to *stop_price*, the loss equals *risk_pct* of current equity.

        Parameters
        ----------
        price : float
            Entry price.
        stop_price : float
            Stop-loss price (often the AI's P10 prediction).
        risk_pct : float
            Max fraction of equity to risk (default 2 %).
        allow_fractional : bool
            Whether fractional quantities are permitted (e.g. for crypto).

        Returns
        -------
        float
            Number of shares.
        """
        equity = self.total_equity
        risk_dollars = equity * risk_pct
        risk_per_share = abs(price - stop_price)

        if risk_per_share <= 0:
            logger.warning("Stop price >= entry price — returning 0 shares")
            return 0.0

        raw_shares = risk_dollars / risk_per_share
        max_affordable = self.cash * 0.95 / (price * (1 + self.slippage_range[1]))
        target = min(raw_shares, max_affordable)

        if not allow_fractional:
            return float(max(0, int(target)))

        if price >= 1000.0:
            return round(target, 4)
        elif price >= 10.0:
            return round(target, 2)
        return round(target, 1)

    def calculate_max_shares(self, price: float, cash_buffer: float = 0.95) -> float:
        """Calculate maximum affordable shares with cash safety buffer."""
        if price <= 0:
            return 0.0
        affordable = (self.cash * cash_buffer) / price
        if price >= 1000.0:
            return round(affordable, 4)
        elif price >= 10.0:
            return round(affordable, 2)
        return round(affordable, 1)

    def kelly_position_size(
        self,
        price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        kelly_fraction: float = 0.25,  # quarter-Kelly for safety
    ) -> int:
        """
        Kelly-criterion position sizer (fractional Kelly for conservatism).

        Parameters
        ----------
        price : float
            Entry price.
        win_rate : float
            Historical win rate (0–1).
        avg_win_pct : float
            Average winning trade return (e.g. 0.03 = 3 %).
        avg_loss_pct : float
            Average losing trade return (e.g. 0.01 = 1 %).
        kelly_fraction : float
            ¼-Kelly by default for risk control.

        Returns
        -------
        int
            Number of shares.
        """
        if avg_loss_pct == 0:
            return 0
        b = avg_win_pct / avg_loss_pct  # win/loss ratio
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b
        kelly = max(0, kelly * kelly_fraction)  # fractional Kelly

        capital_to_deploy = self.cash * kelly
        shares = int(capital_to_deploy / price)
        return max(0, shares)

    # ══════════════════════════════════════════════════════════════════════
    #  Metrics
    # ══════════════════════════════════════════════════════════════════════

    @property
    def total_equity(self) -> float:
        """Cash + market value of all open positions."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def equity(self) -> float:
        """Alias for total_equity."""
        return self.total_equity

    @property
    def metrics(self) -> PortfolioMetrics:
        """Compute all portfolio analytics."""
        m = PortfolioMetrics()
        m.cash = round(self.cash, 2)
        m.total_equity = round(self.total_equity, 2)
        m.unrealised_pnl = round(
            sum(p.unrealised_pnl for p in self.positions.values()), 2
        )

        # Realised P&L from closed trades
        selling_fills = [f for f in self.fills if f.side == Side.SELL]
        m.realised_pnl = round(sum(f.pnl for f in selling_fills), 2)
        m.total_trades = len(selling_fills)

        wins = [f for f in selling_fills if f.pnl > 0]
        losses = [f for f in selling_fills if f.pnl <= 0]
        m.winning_trades = len(wins)
        m.losing_trades = len(losses)
        m.win_rate = round(len(wins) / max(len(selling_fills), 1), 4)
        m.avg_win = round(np.mean([f.pnl for f in wins]), 2) if wins else 0.0
        m.avg_loss = round(np.mean([abs(f.pnl) for f in losses]), 2) if losses else 0.0

        # Profit factor = gross wins / gross losses
        gross_win = sum(f.pnl for f in wins) if wins else 0.0
        gross_loss = sum(abs(f.pnl) for f in losses) if losses else 1.0
        m.profit_factor = round(gross_win / gross_loss, 4) if gross_loss > 0 else 0.0

        # Sharpe Ratio (annualised)
        if len(self._daily_returns) >= 2:
            ret = np.array(self._daily_returns)
            excess = ret - self.risk_free_rate / 252
            std = np.std(excess, ddof=1)
            m.sharpe_ratio = round(
                float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0, 4
            )

        # Max Drawdown
        if len(self._equity_curve) >= 2:
            curve = np.array(self._equity_curve)
            peak = np.maximum.accumulate(curve)
            drawdown = (curve - peak) / peak
            m.max_drawdown_pct = round(float(np.min(drawdown) * 100), 4)

        return m

    @property
    def equity_curve(self) -> pd.Series:
        """Return the equity curve as a Pandas Series."""
        return pd.Series(self._equity_curve, name="Equity")

    # ══════════════════════════════════════════════════════════════════════
    #  Backtesting helpers
    # ══════════════════════════════════════════════════════════════════════

    def backtest(
        self,
        signals: pd.DataFrame,
        price_col: str = "Close",
        signal_col: str = "signal",
        quantity: float = 10.0,
    ) -> PortfolioMetrics:
        """
        Run a vectorised backtest from a signal DataFrame.

        Parameters
        ----------
        signals : pd.DataFrame
            Must contain *price_col* and *signal_col* (1 = buy, -1 = sell, 0 = hold).
        price_col : str
            Column with prices.
        signal_col : str
            Column with trade signals.
        quantity : float
            Shares per trade.

        Returns
        -------
        PortfolioMetrics
            Full metrics at end of backtest.
        """
        logger.info("Backtesting %d bars …", len(signals))

        for ts, row in signals.iterrows():
            price = row[price_col]
            sig = row[signal_col]

            if sig == 1:  # BUY
                if "BACKTEST" not in self.positions:
                    try:
                        self.submit_order(
                            TradeOrder(
                                ticker="BACKTEST",
                                side=Side.BUY,
                                quantity=quantity,
                                price=price,
                            )
                        )
                    except ValueError:
                        pass  # insufficient cash

            elif sig == -1:  # SELL
                if "BACKTEST" in self.positions:
                    pos = self.positions["BACKTEST"]
                    try:
                        self.submit_order(
                            TradeOrder(
                                ticker="BACKTEST",
                                side=Side.SELL,
                                quantity=pos.quantity,
                                price=price,
                            )
                        )
                    except ValueError:
                        pass

            self.mark_to_market({"BACKTEST": price})

        return self.metrics

    # ══════════════════════════════════════════════════════════════════════
    #  Trade History
    # ══════════════════════════════════════════════════════════════════════

    def trade_history(self) -> pd.DataFrame:
        """Return all fills as a DataFrame."""
        if not self.fills:
            return pd.DataFrame()
        return pd.DataFrame([f.to_dict() for f in self.fills])

    def positions_summary(self) -> pd.DataFrame:
        """Return all open positions as a DataFrame."""
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.positions.values()])

    # ══════════════════════════════════════════════════════════════════════
    #  Reset
    # ══════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Reset the simulator to initial state."""
        self.cash = self.starting_capital
        self.positions.clear()
        self.fills.clear()
        self._equity_curve = [self.starting_capital]
        self._daily_returns.clear()
        self._stop_losses.clear()
        self._take_profits.clear()
        self._trailing_stops.clear()
        self._trade_notes.clear()
        if self.user_id is not None:
            try:
                from src.data.database import get_database
                db = get_database()
                db.reset_user_account(self.user_id)
            except Exception as e:
                logger.error("Failed to reset user account in db: %s", e)
        if self.persist:
            self._save_portfolio()
        logger.info("Gauntlet reset to $%.2f", self.starting_capital)

    # ══════════════════════════════════════════════════════════════════════
    #  Risk Management
    # ══════════════════════════════════════════════════════════════════════

    def set_stop_loss(self, ticker: str, price: float) -> None:
        """Set a stop-loss price for a position."""
        self._stop_losses[ticker] = price
        logger.info("🛑 Stop-loss set for %s @ $%.2f", ticker, price)

    def set_take_profit(self, ticker: str, price: float) -> None:
        """Set a take-profit price for a position."""
        self._take_profits[ticker] = price
        logger.info("🎯 Take-profit set for %s @ $%.2f", ticker, price)

    def set_trailing_stop(self, ticker: str, trail_pct: float, current_price: Optional[float] = None) -> None:
        """
        Set a dynamic trailing stop-loss for a position.
        trail_pct: trailing percentage (e.g. 0.03 for 3%).
        """
        ref_price = current_price
        if ref_price is None:
            if ticker in self.positions:
                pos = self.positions[ticker]
                ref_price = pos.current_price or pos.avg_cost
            else:
                ref_price = 100.0

        stop_price = ref_price * (1.0 - trail_pct)
        self._trailing_stops[ticker] = {
            "trail_pct": trail_pct,
            "peak_price": ref_price,
            "stop_price": stop_price,
        }
        logger.info("🛡️ Trailing stop set for %s: trail=%.2f%%, initial stop=$%.2f (peak=$%.2f)",
                    ticker, trail_pct * 100, stop_price, ref_price)

    def update_trailing_stops(self, ticker: str, high_price: float) -> Optional[float]:
        """
        Ratchet trailing stop upwards if high_price exceeds the previous peak.
        Returns the updated stop price.
        """
        if ticker in self._trailing_stops:
            ts = self._trailing_stops[ticker]
            if high_price > ts["peak_price"]:
                ts["peak_price"] = high_price
                new_stop = high_price * (1.0 - ts["trail_pct"])
                ts["stop_price"] = max(ts["stop_price"], new_stop)
                logger.debug("🛡️ Trailing stop ratcheted for %s: peak=$%.2f, new stop=$%.2f",
                             ticker, high_price, ts["stop_price"])
            return ts["stop_price"]
        return None

    def add_trade_note(self, order_id: str, note: str) -> None:
        """Add a note to a trade (trade journal)."""
        self._trade_notes[order_id] = note

    # ══════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════

    def _save_portfolio(self) -> None:
        """Save portfolio state to SQLite database (if user_id) and JSON for persistence."""
        if self.user_id is not None:
            try:
                from src.data.database import get_database
                db = get_database()
                db.update_user_balance(self.user_id, self.cash)

                # Upsert all active positions
                for t, p in self.positions.items():
                    ts = self._trailing_stops.get(t)
                    db.save_user_position(
                        user_id=self.user_id,
                        ticker=t,
                        quantity=p.quantity,
                        avg_cost=p.avg_cost,
                        current_price=p.current_price,
                        stop_loss=self._stop_losses.get(t),
                        take_profit=self._take_profits.get(t),
                        trailing_stop_pct=ts["trail_pct"] if ts else None,
                        trailing_stop_peak=ts["peak_price"] if ts else None,
                    )

                # Clean up any positions closed in memory from DB
                existing_positions = db.get_user_positions(self.user_id)
                for t in set(existing_positions.keys()) - set(self.positions.keys()):
                    db.delete_user_position(self.user_id, t)
            except Exception as e:
                logger.error("Failed to save portfolio to DB for user %s: %s", self.user_id, e)

        # JSON fallback
        try:
            state = {
                "cash": self.cash,
                "starting_capital": self.starting_capital,
                "positions": {
                    t: p.to_dict() for t, p in self.positions.items()
                },
                "fills": [f.to_dict() for f in self.fills[-100:]],  # Keep last 100
                "equity_curve": self._equity_curve[-500:],  # Keep last 500 points
                "stop_losses": self._stop_losses,
                "take_profits": self._take_profits,
                "trailing_stops": self._trailing_stops,
                "trade_notes": self._trade_notes,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PORTFOLIO_PATH, "w") as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug("Portfolio saved → %s", PORTFOLIO_PATH)
        except Exception as e:
            logger.error("Failed to save portfolio to JSON: %s", e)

    def _load_from_db(self) -> None:
        """Load portfolio state from SQLite database for authenticated user."""
        if self.user_id is None:
            return
        try:
            from src.data.database import get_database
            db = get_database()
            user = db.get_user_by_id(self.user_id)
            if user:
                self.cash = float(user["cash_balance"])
                self.starting_capital = float(user["starting_capital"])

            # Load positions
            db_positions = db.get_user_positions(self.user_id)
            self.positions.clear()
            for t, p in db_positions.items():
                self.positions[t] = Position(
                    ticker=p["ticker"],
                    quantity=float(p["quantity"]),
                    avg_cost=float(p["avg_cost"]),
                    current_price=float(p["current_price"]),
                )
                if p.get("stop_loss"):
                    self._stop_losses[t] = float(p["stop_loss"])
                if p.get("take_profit"):
                    self._take_profits[t] = float(p["take_profit"])
                if p.get("trailing_stop_pct"):
                    peak = float(p.get("trailing_stop_peak") or p["current_price"])
                    pct = float(p["trailing_stop_pct"])
                    self._trailing_stops[t] = {
                        "trail_pct": pct,
                        "peak_price": peak,
                        "stop_price": peak * (1.0 - pct),
                    }

            # Load fills
            self.fills.clear()
            trades = db.get_user_trades(self.user_id, limit=100)
            for tr in reversed(trades):
                self.fills.append(
                    Fill(
                        order_id=tr["order_id"],
                        ticker=tr["ticker"],
                        side=Side.BUY if tr["side"] == "BUY" else Side.SELL,
                        quantity=float(tr["quantity"]),
                        intended_price=float(tr["fill_price"]),
                        fill_price=float(tr["fill_price"]),
                        commission=0.0,
                        slippage_pct=0.0,
                        total_cost=float(tr["total_cost"]),
                        pnl=float(tr.get("pnl") or 0.0),
                        exit_reason=tr.get("exit_reason") or "MANUAL",
                        timestamp=tr.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    )
                )

            # Initialize equity curve with current equity
            self._equity_curve = [self.total_equity]
            logger.info(
                "User %d portfolio loaded from DB: $%.2f cash, %d positions, %d fills",
                self.user_id, self.cash, len(self.positions), len(self.fills),
            )
        except Exception as e:
            logger.error("Failed to load portfolio from DB for user %s: %s", self.user_id, e)
            self._load_portfolio()

    def _load_portfolio(self) -> None:
        """Load portfolio state from JSON if it exists."""
        if not PORTFOLIO_PATH.exists():
            return
        try:
            with open(PORTFOLIO_PATH) as f:
                state = json.load(f)
            self.cash = state.get("cash", self.starting_capital)
            self._equity_curve = state.get("equity_curve", [self.starting_capital])
            self._stop_losses = state.get("stop_losses", {})
            self._take_profits = state.get("take_profits", {})
            self._trailing_stops = state.get("trailing_stops", {})
            self._trade_notes = state.get("trade_notes", {})

            # Restore positions
            for ticker, pdata in state.get("positions", {}).items():
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=pdata["quantity"],
                    avg_cost=pdata["avg_cost"],
                    current_price=pdata.get("current_price", 0.0),
                )

            logger.info(
                "Portfolio loaded: $%.2f cash, %d positions, %d equity points",
                self.cash, len(self.positions), len(self._equity_curve),
            )
        except Exception as e:
            logger.warning("Could not load portfolio: %s — starting fresh", e)

    def __repr__(self) -> str:
        return (
            f"Gauntlet(equity=${self.total_equity:,.2f}, "
            f"cash=${self.cash:,.2f}, "
            f"positions={len(self.positions)}, "
            f"fills={len(self.fills)})"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Smoke test  (python -m src.models.trader)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  HFP Gauntlet — Smoke Test")
    print("=" * 60)

    sim = Gauntlet(starting_capital=10_000, seed=42)

    # ── Simulate a round-trip trade ───────────────────────────────────
    print("\n📊 Opening position: BUY 20 AAPL @ $150")
    buy_fill = sim.submit_order(
        TradeOrder(ticker="AAPL", side=Side.BUY, quantity=20, price=150.00)
    )
    print(f"   Fill: {buy_fill.fill_price:.4f} (slip={buy_fill.slippage_pct:.4f}%, comm=${buy_fill.commission:.2f})")

    # Mark to market at $155
    sim.mark_to_market({"AAPL": 155.00})
    print(f"\n📈 MtM @ $155 → equity=${sim.total_equity:,.2f}")

    # Position sizing
    shares = sim.calculate_position_size(price=155, stop_price=148, risk_pct=0.02)
    print(f"\n🎯 Position sizer (2% risk, stop=$148): {shares} shares")

    # Sell
    print(f"\n📊 Closing position: SELL 20 AAPL @ $155")
    sell_fill = sim.submit_order(
        TradeOrder(ticker="AAPL", side=Side.SELL, quantity=20, price=155.00)
    )
    print(f"   Fill: {sell_fill.fill_price:.4f} (P&L=${sell_fill.pnl:.2f})")

    # Metrics
    m = sim.metrics
    print(f"\n📋 Metrics:")
    print(f"   Total equity:   ${m.total_equity:,.2f}")
    print(f"   Cash:           ${m.cash:,.2f}")
    print(f"   Realised P&L:   ${m.realised_pnl:,.2f}")
    print(f"   Win rate:       {m.win_rate:.0%}")
    print(f"   Sharpe:         {m.sharpe_ratio}")
    print(f"   Max drawdown:   {m.max_drawdown_pct}%")

    # ── Backtest with signals ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Backtest Demo")
    print("=" * 60)

    sim2 = Gauntlet(starting_capital=10_000, seed=42)
    # Synthetic price + signal data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    signals = np.zeros(50, dtype=int)
    signals[5] = 1     # buy
    signals[15] = -1   # sell
    signals[20] = 1    # buy
    signals[40] = -1   # sell

    df = pd.DataFrame({"Close": prices, "signal": signals})
    result = sim2.backtest(df)
    print(f"\n📋 Backtest result:")
    print(f"   Final equity:   ${result.total_equity:,.2f}")
    print(f"   Realised P&L:   ${result.realised_pnl:,.2f}")
    print(f"   Total trades:   {result.total_trades}")
    print(f"   Win rate:       {result.win_rate:.0%}")
    print(f"   Max drawdown:   {result.max_drawdown_pct:.2f}%")

    print("\n✅ Gauntlet smoke test passed.")
