"""
Unit tests for AutonomousSniperBot in src/models/bot.py.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import pytest

from src.data.live_feed import LiveMarketFeed, StreamMode
from src.models.bot import AutonomousSniperBot, BotConfig, BotState
from src.models.strategies import BaseStrategy, SignalType, StrategySignal
from src.models.trader import Side


class DummyBullishStrategy(BaseStrategy):
    """Deterministic strategy for testing bot execution."""

    def __init__(self):
        super().__init__("Dummy Bullish")

    def generate_signal(self, df, current_idx=-1, prediction=None):
        price = float(df["Close"].iloc[current_idx])
        return StrategySignal(
            strategy_name=self.name,
            ticker="BTC-CAD",
            signal_type=SignalType.BUY,
            strength=0.85,
            target_price=price * 1.10,
            stop_price=price * 0.95,
            reason="Test bullish breakout",
        )


def _create_sample_df(n_rows: int = 50, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(start="2026-01-01", periods=n_rows, freq="1D")
    prices = [start_price]
    for _ in range(n_rows - 1):
        prices.append(prices[-1] * 1.01)

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [10000.0] * n_rows,
            "Whale_Flag": [0] * n_rows,
            "Whale_Z": [0.0] * n_rows,
        },
        index=dates,
    )
    return df


def test_bot_initialization():
    df = _create_sample_df(40)
    feed = LiveMarketFeed(ticker="BTC-CAD", mode=StreamMode.HISTORICAL_REPLAY, base_df=df)
    bot = AutonomousSniperBot(feed=feed)

    assert bot.state == BotState.DISARMED
    assert bot.gauntlet.cash == 100_000.0
    metrics = bot.get_metrics()
    assert metrics["State"] == "DISARMED"
    assert metrics["Ticks Processed"] == 0


def test_bot_arm_and_disarm():
    df = _create_sample_df(40)
    feed = LiveMarketFeed(ticker="BTC-CAD", mode=StreamMode.HISTORICAL_REPLAY, base_df=df)
    bot = AutonomousSniperBot(feed=feed)

    bot.arm()
    assert bot.state == BotState.ARMED
    bot.disarm()
    assert bot.state == BotState.DISARMED


def test_bot_execution_on_signal():
    df = _create_sample_df(40, start_price=100.0)
    feed = LiveMarketFeed(ticker="BTC-CAD", mode=StreamMode.HISTORICAL_REPLAY, base_df=df)
    strategy = DummyBullishStrategy()
    config = BotConfig(symbol="BTC-CAD", initial_balance=100_000.0, risk_per_trade=0.02)
    bot = AutonomousSniperBot(feed=feed, strategy=strategy, config=config)

    bot.arm()
    tick = bot.step()

    # Bot should have submitted a BUY order
    assert len(bot.gauntlet.positions) == 1
    assert len(bot.gauntlet.fills) == 1
    fill = bot.gauntlet.fills[0]
    assert fill.side == Side.BUY
    assert fill.quantity > 0


def test_bot_trailing_stop_triggered():
    df = _create_sample_df(40, start_price=100.0)
    feed = LiveMarketFeed(ticker="BTC-CAD", mode=StreamMode.HISTORICAL_REPLAY, base_df=df)
    strategy = DummyBullishStrategy()
    config = BotConfig(
        symbol="BTC-CAD",
        initial_balance=100_000.0,
        trailing_stop_enabled=True,
        trailing_stop_pct=0.02,  # 2% trailing stop
    )
    bot = AutonomousSniperBot(feed=feed, strategy=strategy, config=config)
    bot.arm()
    bot.step()  # Buys at ~100.0
    assert "BTC-CAD" in bot.gauntlet.positions

    # Simulate price climbing to 118.0 (below TP 122.72, trailing stop ratchets to 118 * 0.98 = 115.64)
    bot.gauntlet.mark_to_market({"BTC-CAD": 118.0})
    assert bot.gauntlet._trailing_stops["BTC-CAD"]["stop_price"] == pytest.approx(118.0 * 0.98, rel=1e-3)

    # Now simulate price pulling back to 114.0 (below 115.64 stop)
    bot.gauntlet.mark_to_market({"BTC-CAD": 114.0})

    # Order should now be closed with reason TRAILING_STOP
    assert "BTC-CAD" not in bot.gauntlet.positions
    sell_fills = [f for f in bot.gauntlet.fills if f.side == Side.SELL]
    assert len(sell_fills) >= 1
    assert sell_fills[-1].exit_reason == "TRAILING_STOP"

    csv_data = bot.export_trades_csv()
    assert "TRAILING_STOP" in csv_data
