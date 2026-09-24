"""
Tests for Intra-Bar Bracket Orders, Trailing Stops, and Sniper Execution Engine
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from src.models.ai_engine import QuantilePrediction
from src.models.sniper import SniperConfig, SniperEngine, SniperSignal
from src.models.trader import Gauntlet, Side, TradeOrder


@pytest.fixture
def sim():
    """Create fresh in-memory Gauntlet simulator."""
    return Gauntlet(
        starting_capital=100000.0,
        slippage_range=(0.0001, 0.0001),
        commission_flat=1.0,
        commission_pct=0.001,
        persist=False,
    )


def test_intra_bar_stop_loss_hit(sim):
    """Verify stop-loss executes on intra-bar Low breach even if Close rebounds."""
    # Buy 10 shares at $100
    sim.submit_order(TradeOrder(ticker="AAPL", side=Side.BUY, quantity=10, price=100.0))
    sim.set_stop_loss("AAPL", price=95.0)

    # Bar opens at 98, drops to Low=94 (breaches 95), then rebounds to Close=97
    bar = {"Open": 98.0, "High": 99.0, "Low": 94.0, "Close": 97.0}
    sim.evaluate_bar("AAPL", bar)

    # Should have triggered stop loss and closed position
    assert "AAPL" not in sim.positions
    assert len(sim.fills) == 2
    exit_fill = sim.fills[-1]
    assert exit_fill.side == Side.SELL
    assert exit_fill.exit_reason == "STOP_LOSS"
    assert exit_fill.fill_price <= 95.0


def test_intra_bar_take_profit_hit(sim):
    """Verify take-profit executes on intra-bar High breach even if Close pulls back."""
    # Buy 10 shares at $100
    sim.submit_order(TradeOrder(ticker="MSFT", side=Side.BUY, quantity=10, price=100.0))
    sim.set_take_profit("MSFT", price=110.0)

    # Bar opens at 105, peaks at High=112 (breaches 110), then pulls back to Close=107
    bar = {"Open": 105.0, "High": 112.0, "Low": 104.0, "Close": 107.0}
    sim.evaluate_bar("MSFT", bar)

    # Should have triggered take profit and closed position
    assert "MSFT" not in sim.positions
    assert len(sim.fills) == 2
    exit_fill = sim.fills[-1]
    assert exit_fill.side == Side.SELL
    assert exit_fill.exit_reason == "TAKE_PROFIT"
    assert exit_fill.fill_price >= 109.9  # after minute slippage


def test_trailing_stop_ratchet_and_trigger(sim):
    """Verify trailing stop ratchets up on new highs and locks in profits on pullback."""
    # Buy 10 shares of NVDA at $100
    order = TradeOrder(ticker="NVDA", side=Side.BUY, quantity=10, price=100.0)
    sim.submit_bracket_order(order, trailing_stop_pct=0.05)  # 5% trailing buffer

    # Initial stop is $100 * 0.95 = $95.00
    assert "NVDA" in sim._trailing_stops
    assert pytest.approx(sim._trailing_stops["NVDA"]["stop_price"], 0.1) == 95.0

    # Bar 1: Price rallies strongly to High=$120, Close=$118
    bar1 = {"Open": 102.0, "High": 120.0, "Low": 101.0, "Close": 118.0}
    sim.evaluate_bar("NVDA", bar1)

    # Stop should have ratcheted up to $120 * 0.95 = $114.00
    assert pytest.approx(sim._trailing_stops["NVDA"]["stop_price"], 0.1) == 114.0
    assert "NVDA" in sim.positions  # Still open

    # Bar 2: Price pulls back to Low=$112, Close=$115
    bar2 = {"Open": 117.0, "High": 117.0, "Low": 112.0, "Close": 115.0}
    sim.evaluate_bar("NVDA", bar2)

    # Should have triggered trailing stop exit at ~$114, locking in profit!
    assert "NVDA" not in sim.positions
    exit_fill = sim.fills[-1]
    assert exit_fill.exit_reason == "TRAILING_STOP"
    assert exit_fill.pnl > 100.0  # Profitable exit


def test_sniper_signal_evaluation_and_execution(sim):
    """Verify Sniper engine generates high-conviction signals with proper bracket execution."""
    engine = SniperEngine(SniperConfig(min_confidence=0.75, min_risk_reward=1.5))

    pred = QuantilePrediction(
        p10=95.0,
        p50=110.0,
        p90=120.0,
        confidence=0.82,
        current_price=100.0,
    )
    features = {"Whale_Flag": 1, "Whale_Z": 3.5}

    signal = engine.evaluate(
        ticker="GOOGL",
        prediction=pred,
        current_price=100.0,
        sim=sim,
        latest_features=features,
    )

    assert signal is not None
    assert signal.side == Side.BUY
    assert signal.stop_loss == 95.0
    assert signal.target_price == 120.0
    assert signal.risk_reward_ratio == 4.0  # ($120-$100) / ($100-$95) = 20 / 5 = 4.0
    assert signal.whale_confirmed is True
    assert signal.suggested_shares > 0

    # Execute the signal into Gauntlet
    fill = engine.execute(signal, sim)
    assert fill.quantity == signal.suggested_shares
    assert "GOOGL" in sim.positions
    assert "GOOGL" in sim._stop_losses
    assert "GOOGL" in sim._take_profits
    assert fill.order_id in sim._trade_notes


def test_sniper_filters_reject_low_confidence(sim):
    """Verify Sniper rejects predictions below confidence threshold."""
    engine = SniperEngine(SniperConfig(min_confidence=0.75))

    pred = QuantilePrediction(
        p10=90.0,
        p50=105.0,
        p90=115.0,
        confidence=0.60,  # Below 0.75
        current_price=100.0,
    )

    signal = engine.evaluate(ticker="TSLA", prediction=pred, current_price=100.0, sim=sim)
    assert signal is None


def test_sniper_filters_reject_poor_rrr(sim):
    """Verify Sniper rejects signals with unfavorable risk-reward ratio."""
    engine = SniperEngine(SniperConfig(min_risk_reward=1.5))

    # Upside $1 ($101-$100) vs Downside $5 ($100-$95) -> RRR = 0.2x
    pred = QuantilePrediction(
        p10=95.0,
        p50=100.5,
        p90=101.0,
        confidence=0.85,
        current_price=100.0,
    )

    signal = engine.evaluate(ticker="AMZN", prediction=pred, current_price=100.0, sim=sim)
    assert signal is None
