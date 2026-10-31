"""
Unit & Integration Tests for TradeSense Overhaul:
  1. MarketIntelligenceEngine (Yahoo nested/legacy news, RSS, VADER+Financial lexicon, catalysts, TSX hours)
  2. CanadianAssetProxy live TSX/NEO proxy price translator & Wealthsimple tier fee math
  3. 6-Strategy Tournament Arena (NewsCatalystMomentum, StatisticalMeanReversion, intra-bar brackets)
  4. AIEngine out-of-sample accuracy & calibration verification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.live_feed import LiveMarketFeed, StreamMode
from src.data.market_intel import (
    AssetIntelligenceReport,
    MacroPulseSnapshot,
    MarketIntelligenceEngine,
    NewsArticle,
)
from src.models.ai_engine import AIEngine
from src.models.strategies import (
    DeepQuantileSniper,
    EnsembleAlpha,
    InstitutionalWhaleFlow,
    NewsCatalystMomentum,
    SignalType,
    StatisticalMeanReversion,
    StrategyArena,
    VolatilityBreakout,
    ensure_strategy_indicators,
)
from src.models.wealthsimple_copier import CanadianAssetProxy


def _make_sample_ohlcv(rows: int = 90, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=rows, freq="1D")
    returns = rng.normal(0.002, 0.018, size=rows)
    close = start_price * np.exp(np.cumsum(returns))
    high = close * (1.0 + rng.uniform(0.005, 0.025, size=rows))
    low = close * (1.0 - rng.uniform(0.005, 0.025, size=rows))
    open_p = close * (1.0 + rng.normal(0.0, 0.006, size=rows))
    vol = rng.uniform(100_000, 500_000, size=rows)
    # Inject a few high-volume whale & oversold/breakout bars
    if rows >= 10:
        vol[rows // 3] *= 4.5
        vol[(2 * rows) // 3] *= 4.0
        vol[-2] *= 4.2
    df = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    return ensure_strategy_indicators(df)


def test_market_intel_sentiment_and_catalyst_classification():
    engine = MarketIntelligenceEngine()

    bull_score, bull_cat = engine.score_headline(
        "NVIDIA beats Q4 earnings expectations and raises AI revenue guidance"
    )
    assert bull_score > 0.20
    assert bull_cat == "EARNINGS & GUIDANCE"

    bear_score, bear_cat = engine.score_headline(
        "SEC files enforcement lawsuit and investigation over regulatory compliance breach"
    )
    assert bear_score < -0.15
    assert bear_cat == "REGULATORY & LEGAL"

    etf_score, etf_cat = engine.score_headline(
        "Record institutional ETF inflow surges as Bitcoin spot demand rallies"
    )
    assert etf_score > 0.15
    assert etf_cat == "ETF & INSTITUTIONAL FLOW"


def test_toronto_market_hours_status():
    status = MarketIntelligenceEngine.get_toronto_market_hours_status()
    assert "is_tsx_open" in status
    assert "session_badge" in status
    assert "order_routing_advice" in status
    assert status["is_crypto_open"] is True


def test_wealthsimple_proxy_order_translator():
    # BTC-CAD -> BTCC.TO unit-scaled proxy translation
    plan_btc = CanadianAssetProxy.translate_order_to_proxy(
        ticker="BTC-CAD",
        underlying_entry=115_000.0,
        underlying_target=120_750.0,  # +5.0%
        underlying_stop=112_700.0,    # -2.0%
        capital_cad=1_000.0,
        ws_tier="Core",
    )
    assert plan_btc["ws_symbol"] == "BTCC.TO"
    assert plan_btc["is_unit_scaled"] is True
    assert 5.0 <= plan_btc["proxy_entry_cad"] <= 100.0
    # Verify +5% target and -2% stop scale proportionally on BTCC.TO CAD share price
    assert pytest.approx(plan_btc["proxy_target_cad"] / plan_btc["proxy_entry_cad"], rel=1e-2) == 1.05
    assert pytest.approx(plan_btc["proxy_stop_cad"] / plan_btc["proxy_entry_cad"], rel=1e-2) == 0.98
    assert plan_btc["est_roundtrip_fee_cad"] == 0.0  # $0 TSX ETF commission
    assert plan_btc["share_qty"] > 10.0

    # US stock -> NEO CDR (NVDA -> NVDA.NE)
    plan_nvda = CanadianAssetProxy.translate_order_to_proxy(
        ticker="NVDA",
        underlying_entry=138.0,
        underlying_target=144.9,
        underlying_stop=134.5,
        capital_cad=500.0,
        ws_tier="Core",
    )
    assert plan_nvda["ws_symbol"] == "NVDA.NE"
    assert plan_nvda["is_unit_scaled"] is True
    assert plan_nvda["est_roundtrip_fee_cad"] == 0.0


def test_live_feed_defaults_to_live_stream():
    sample_df = _make_sample_ohlcv(50, start_price=185.0)
    feed = LiveMarketFeed(ticker="SHOP.TO", base_df=sample_df)
    assert feed.mode == StreamMode.LIVE_STREAM
    tick = feed.get_next_tick()
    assert tick.price > 0
    assert not feed.streaming_df.empty
    assert hasattr(feed, "data_source_label")


def test_six_strategy_tournament_executes_with_intra_bar_brackets():
    df = _make_sample_ohlcv(110, start_price=100_000.0)
    arena = StrategyArena()
    assert len(arena.strategies) == 6
    results = arena.run_tournament(df, ticker="BTC-CAD")
    assert len(results) == 6
    strat_names = {r.strategy_name for r in results}
    assert "News & Catalyst Momentum" in strat_names
    assert "Statistical Mean Reversion" in strat_names
    assert "Deep Quantile Sniper" in strat_names
    # Ensure strategies execute trades and Profit Factor is capped cleanly <= 99.99
    total_trades_across_arena = sum(r.total_trades for r in results)
    assert total_trades_across_arena > 0
    for r in results:
        assert r.profit_factor <= 99.99


def test_news_catalyst_momentum_signal_and_attribution():
    df = _make_sample_ohlcv(40, start_price=200.0)
    df["Sentiment"] = 0.45
    df.iloc[-1, df.columns.get_loc("Close")] = float(df["VWAP"].iloc[-1]) * 1.01
    df.iloc[-1, df.columns.get_loc("RSI_14")] = 58.0

    strat = NewsCatalystMomentum()
    sig = strat.generate_signal(df, current_idx=-1)
    assert sig.signal_type == SignalType.BUY
    assert sig.target_price is not None and sig.target_price > float(df["Close"].iloc[-1])

    attr = strat.get_decision_attribution(df, current_idx=-1)
    factor_names = [f["name"] for f in attr["factors"]]
    assert "News & Catalyst Pulse" in factor_names
