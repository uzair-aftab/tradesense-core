"""
Tests for Gauntlet Paper Trading Simulator
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from src.models.trader import Gauntlet, TradeOrder, Side


@pytest.fixture
def sim(tmp_path):
    """Create a Gauntlet instance backed by a temporary state file."""
    return Gauntlet(
        starting_capital=100000.0,
        slippage_range=(0.001, 0.001),
        commission_flat=1.00,
        commission_pct=0.001,
        persist=False,
    )


def test_buy_and_sell_execution(sim):
    """Verify order filling, cash deduction, slippage, and PnL calculation."""
    # Buy 10 shares of NVDA at $100.00
    buy_order = TradeOrder(ticker="NVDA", side=Side.BUY, quantity=10, price=100.0)
    fill_buy = sim.submit_order(buy_order)

    assert fill_buy.quantity == 10
    # Price + slippage (0.1% of 100 = 100.10)
    assert fill_buy.fill_price > 100.0
    assert fill_buy.commission > 0
    assert "NVDA" in sim.positions
    assert sim.positions["NVDA"].quantity == 10

    # Sell 10 shares of NVDA at $110.00
    sell_order = TradeOrder(ticker="NVDA", side=Side.SELL, quantity=10, price=110.0)
    fill_sell = sim.submit_order(sell_order)

    assert fill_sell.quantity == 10
    assert fill_sell.pnl > 0.0  # Profitable trade
    assert "NVDA" not in sim.positions  # Fully closed


def test_stop_loss_trigger(sim):
    """Verify automatic stop-loss execution when price drops below threshold."""
    # Buy 10 shares at $100
    sim.submit_order(TradeOrder(ticker="AAPL", side=Side.BUY, quantity=10, price=100.0))
    sim.set_stop_loss("AAPL", price=95.0)

    # Mark to market above stop price -> remains open
    sim.mark_to_market({"AAPL": 96.0})
    assert "AAPL" in sim.positions

    # Mark to market below stop price -> triggers auto-exit
    sim.mark_to_market({"AAPL": 94.0})
    assert "AAPL" not in sim.positions
    assert len(sim.fills) == 2  # Buy + Auto-Sell


def test_take_profit_trigger(sim):
    """Verify automatic take-profit execution when price exceeds threshold."""
    # Buy 10 shares at $100
    sim.submit_order(TradeOrder(ticker="MSFT", side=Side.BUY, quantity=10, price=100.0))
    sim.set_take_profit("MSFT", price=110.0)

    # Mark to market below target -> remains open
    sim.mark_to_market({"MSFT": 105.0})
    assert "MSFT" in sim.positions

    # Mark to market at or above target -> triggers auto-exit
    sim.mark_to_market({"MSFT": 112.0})
    assert "MSFT" not in sim.positions


def test_position_sizer(sim):
    """Verify Fixed-Risk and Kelly sizing calculations."""
    # Fixed risk sizing (1% of 100,000 = 1,000 risk; $5 risk/share -> 200 shares)
    shares_fixed = sim.calculate_position_size(
        price=100.0,
        stop_price=95.0,
        risk_pct=0.01,
    )
    assert shares_fixed == 200

    # Kelly criterion sizing
    shares_kelly = sim.kelly_position_size(
        price=100.0,
        win_rate=0.60,
        avg_win_pct=0.03,
        avg_loss_pct=0.02,
    )
    assert shares_kelly > 0
