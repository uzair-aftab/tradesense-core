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


def test_bot_crypto_fractional_execution():
    """Verify bot correctly calculates fractional position size when trading high-priced assets ($80k BTC) with $10k cash."""
    df = _create_sample_df(40, start_price=80_000.0)
    feed = LiveMarketFeed(ticker="BTC-USD", mode=StreamMode.HISTORICAL_REPLAY, base_df=df)
    strategy = DummyBullishStrategy()
    config = BotConfig(
        symbol="BTC-USD",
        initial_balance=10_000.0,  # $10,000 cash vs $80,000 BTC
        risk_per_trade=0.02,       # $200 risk capital
    )
    bot = AutonomousSniperBot(feed=feed, strategy=strategy, config=config)
    bot.arm()
    bot.step()

    # The bot must successfully enter a fractional position, not 0!
    assert "BTC-USD" in bot.gauntlet.positions
    pos = bot.gauntlet.positions["BTC-USD"]
    assert pos.quantity > 0.0
    assert pos.quantity < 1.0  # Fractional coin (e.g. ~0.025 to 0.11 BTC)
    assert pos.market_value <= 10_000.0


def test_live_feed_candle_aggregation():
    """Verify ticks update the current candle and open new candles with continuous prices."""
    feed = LiveMarketFeed(ticker="BTC-CAD", mode=StreamMode.SYNTHETIC_MICROSTRUCTURE, ticks_per_bar=4)
    init_bars = len(feed.streaming_df)

    # First 3 ticks should update the current bar without creating a new row
    for _ in range(3):
        feed.next_tick()
    assert len(feed.streaming_df) == init_bars

    # 4th tick should complete the bar and create a new bar
    feed.next_tick()
    assert len(feed.streaming_df) == init_bars + 1

    # Check continuity: new bar Open == previous bar Close
    prev_close = feed.streaming_df["Close"].iloc[-2]
    new_open = feed.streaming_df["Open"].iloc[-1]
    assert new_open == pytest.approx(prev_close, rel=1e-5)


def test_ensemble_alpha_no_name_error():
    """Verify EnsembleAlpha generates signals without NameError and computes valid target/stop prices."""
    from src.models.strategies import EnsembleAlpha
    from src.models.ai_engine import QuantilePrediction

    df = _create_sample_df(60, start_price=80_000.0)
    df["Whale_Z"] = 3.5
    df["Whale_Flag"] = 1
    df["VWAP"] = 79_000.0

    ensemble = EnsembleAlpha()
    pred = QuantilePrediction(
        p10=78_000.0,
        p50=82_000.0,
        p90=85_000.0,
        confidence=0.88,
        current_price=80_000.0,
    )

    signal = ensemble.generate_signal(df=df, current_idx=-1, prediction=pred)
    assert signal is not None
    assert signal.signal_type in (SignalType.BUY, SignalType.HOLD, SignalType.SELL)
    if signal.signal_type == SignalType.BUY:
        assert signal.target_price is not None
        assert signal.stop_price is not None
        assert signal.target_price > 0
        assert signal.stop_price > 0
        assert signal.target_price > signal.stop_price


def test_live_feed_mode_switch():
    """Verify switching LiveMarketFeed mode re-initializes stream baseline cleanly."""
    base_df = _create_sample_df(50, start_price=150.0)
    feed = LiveMarketFeed(ticker="SOL-USD", mode=StreamMode.HISTORICAL_REPLAY, base_df=base_df)
    assert feed.mode == StreamMode.HISTORICAL_REPLAY
    assert len(feed.streaming_df) > 0

    # Switch to SYNTHETIC_MICROSTRUCTURE
    feed.mode = StreamMode.SYNTHETIC_MICROSTRUCTURE
    assert feed.mode == StreamMode.SYNTHETIC_MICROSTRUCTURE
    assert len(feed.streaming_df) == 60
    # Current price should remain continuous near 150
    assert feed.current_price > 0


def test_chart_time_label_uniqueness():
    """Verify _format_chart_time_labels guarantees category uniqueness for daily and intraday bars."""
    from src.ui.app import _format_chart_time_labels

    # 1. Test 50 daily bars (all 00:00:00 midnight)
    daily_idx = pd.date_range("2026-01-01", periods=50, freq="1D")
    labels = _format_chart_time_labels(daily_idx)
    assert len(labels) == 50
    assert len(set(labels)) == 50  # Must all be completely unique!
    assert "00:00:00" not in labels[0]  # Should format as month-day, e.g. 'Jan 01'

    # 2. Test duplicate timestamps
    dup_idx = pd.to_datetime(["2026-03-24 14:00:00", "2026-03-24 14:00:00", "2026-03-24 14:00:01"])
    dup_labels = _format_chart_time_labels(dup_idx)
    assert len(dup_labels) == 3
    assert len(set(dup_labels)) == 3
    assert "#2" in dup_labels[1]


def test_bot_consecutive_loss_circuit_breaker():
    """Verify 2 consecutive stop-outs trigger a 60-tick circuit breaker pause."""
    feed = LiveMarketFeed("BTC-USD", mode=StreamMode.SYNTHETIC_MICROSTRUCTURE)
    bot = AutonomousSniperBot(feed)
    assert bot._circuit_breaker_ticks == 0
    assert bot._consecutive_losses == 0

    # Simulate first loss
    bot._consecutive_losses = 1
    assert bot._circuit_breaker_ticks == 0

    # Simulate second loss via Fill event
    from src.models.trader import Fill, Side
    fill = Fill(
        order_id="test1",
        ticker="BTC-USD",
        side=Side.SELL,
        quantity=0.1,
        intended_price=80000.0,
        fill_price=79000.0,
        commission=5.0,
        slippage_pct=0.01,
        total_cost=7905.0,
        pnl=-105.0,
        exit_reason="STOP_LOSS",
    )
    bot.gauntlet.fills.append(fill)

    # Step bot once to trigger fill processing
    bot.step()
    assert bot._consecutive_losses == 2
    assert bot._circuit_breaker_ticks > 0


def test_bot_wealthsimple_signal_payload():
    """Verify autonomous bot formats structured Wealthsimple copier signal."""
    feed = LiveMarketFeed("BTC-USD", mode=StreamMode.SYNTHETIC_MICROSTRUCTURE)
    bot = AutonomousSniperBot(feed)
    assert bot.last_wealthsimple_signal is None

    # Force a BUY execution
    class AlwaysBuyStrategy(BaseStrategy):
        def generate_signal(self, df, current_idx=-1, prediction=None):
            p = float(df["Close"].iloc[-1])
            return StrategySignal(self.name, "BTC-USD", SignalType.BUY, 0.90, target_price=p * 1.05, stop_price=p * 0.98, reason="Strong Bull Run")

    bot.strategy = AlwaysBuyStrategy("AlwaysBuy")
    bot.arm()
    bot.step()

    assert bot.last_wealthsimple_signal is not None
    sig = bot.last_wealthsimple_signal
    assert sig["symbol"] == "BTC-USD"
    assert sig["ws_ticker"] == "BTCC.TO"
    assert sig["action"] == "BUY"
    assert sig["take_profit"] > sig["entry_price"]
    assert sig["entry_price"] > sig["stop_loss"]
    assert sig["reward_pct"] >= 1.0


def test_bot_regime_metrics_telemetry():
    """Verify bot metrics include Regime and CircuitBreakerTicks."""
    feed = LiveMarketFeed("BTC-USD", mode=StreamMode.SYNTHETIC_MICROSTRUCTURE)
    bot = AutonomousSniperBot(feed)
    metrics = bot.get_metrics()
    assert "Regime" in metrics
    assert "CircuitBreakerTicks" in metrics
    assert metrics["CircuitBreakerTicks"] == 0
    assert bot.current_regime in ("BULL_TREND", "BEAR_TREND", "RANGING_CHOP", "HIGH_VOLATILITY")



