"""
Tests for StrategyArena and Institutional Quantitative Strategies
"""

import numpy as np
import pandas as pd
import pytest

from src.models.ai_engine import QuantilePrediction
from src.models.strategies import (
    DeepQuantileSniper,
    EnsembleAlpha,
    InstitutionalWhaleFlow,
    SignalType,
    StatisticalMeanReversion,
    StrategyArena,
    TournamentResult,
    VolatilityBreakout,
)


def _generate_ohlcv_data(n_bars: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="1h")
    p0 = 85_000.0
    shocks = np.random.normal(0.001, 0.015, size=n_bars)
    closes = p0 * np.exp(np.cumsum(shocks))
    df = pd.DataFrame(
        {
            "Open": closes * 0.998,
            "High": closes * 1.008,
            "Low": closes * 0.992,
            "Close": closes,
            "Volume": np.random.uniform(500, 3000, size=n_bars),
            "Whale_Z": np.random.uniform(0.1, 3.5, size=n_bars),
            "Whale_Flag": [1 if i % 10 == 0 else 0 for i in range(n_bars)],
            "EMA_9": closes * 0.999,
            "EMA_20": closes * 0.997,
            "EMA_50": closes * 0.995,
            "BB_Upper": closes * 1.02,
            "BB_Lower": closes * 0.98,
            "BB_Mid": closes,
            "RSI_14": np.random.uniform(30, 70, size=n_bars),
            "ADX_14": np.random.uniform(18, 35, size=n_bars),
            "Plus_DI": np.random.uniform(15, 30, size=n_bars),
            "Minus_DI": np.random.uniform(10, 25, size=n_bars),
            "ATR_14": closes * 0.01,
            "VWAP": closes * 0.999,
        },
        index=dates,
    )
    return df


def test_strategy_arena_initialization():
    """Verify StrategyArena accepts default, positional, and keyword arguments."""
    arena1 = StrategyArena()
    assert len(arena1.strategies) == 6
    assert arena1.initial_capital == 100_000.0

    strats = [DeepQuantileSniper(), VolatilityBreakout()]
    arena2 = StrategyArena(strategies=strats, initial_capital=50_000.0)
    assert len(arena2.strategies) == 2
    assert arena2.initial_capital == 50_000.0

    # Also test starting_capital keyword argument
    arena3 = StrategyArena(starting_capital=75_000.0)
    assert arena3.initial_capital == 75_000.0


def test_strategy_arena_run_tournament():
    """Verify run_tournament executes across all strategies and returns ranked TournamentResult."""
    df = _generate_ohlcv_data(60)
    arena = StrategyArena(
        strategies=[
            DeepQuantileSniper(),
            InstitutionalWhaleFlow(),
            VolatilityBreakout(),
            StatisticalMeanReversion(),
            EnsembleAlpha(),
        ]
    )

    pred = QuantilePrediction(
        p10=83_000.0,
        p50=86_000.0,
        p90=89_000.0,
        confidence=0.85,
        current_price=float(df["Close"].iloc[-1]),
    )

    results = arena.run_tournament(df, ticker="BTC-CAD", starting_capital=100_000.0, prediction=pred)

    assert isinstance(results, list)
    assert len(results) == 5

    for r in results:
        assert isinstance(r, TournamentResult)
        assert isinstance(r.strategy_name, str)
        assert isinstance(r.total_return_pct, float)
        assert isinstance(r.sharpe_ratio, float)
        assert isinstance(r.win_rate, float)
        assert isinstance(r.equity_curve, list)
        assert len(r.equity_curve) > 0

        d = r.to_dict()
        assert "Strategy" in d
        assert "Return %" in d
        assert "Sharpe" in d
        assert "Win Rate %" in d
        assert "Equity ($)" in d

    # Results must be sorted by Sharpe then Return descending
    for i in range(len(results) - 1):
        assert results[i].sharpe_ratio >= results[i + 1].sharpe_ratio or (
            results[i].sharpe_ratio == results[i + 1].sharpe_ratio
            and results[i].total_return_pct >= results[i + 1].total_return_pct
        )


def test_strategy_arena_empty_data():
    """Verify StrategyArena handles empty or insufficient data gracefully."""
    arena = StrategyArena()
    empty_df = pd.DataFrame()
    results = arena.run_tournament(empty_df)
    assert results == []

    short_df = _generate_ohlcv_data(10)
    results_short = arena.run_tournament(short_df)
    assert results_short == []


def test_strategy_arena_run_tournament_df():
    """Verify run_tournament_df returns a formatted leaderboard DataFrame."""
    df = _generate_ohlcv_data(50)
    arena = StrategyArena()
    df_board = arena.run_tournament_df(df, ticker="BTC-CAD")
    assert isinstance(df_board, pd.DataFrame)
    assert not df_board.empty
    assert len(df_board) == 6
    assert "Strategy" in df_board.columns
    assert "Return %" in df_board.columns
    assert "Sharpe" in df_board.columns


def test_all_strategies_generate_signals():
    """Verify all individual strategies generate valid signals without exceptions."""
    df = _generate_ohlcv_data(50)
    strategies = [
        DeepQuantileSniper(),
        InstitutionalWhaleFlow(),
        VolatilityBreakout(),
        StatisticalMeanReversion(),
        EnsembleAlpha(),
    ]

    pred = QuantilePrediction(
        p10=80_000.0,
        p50=85_000.0,
        p90=90_000.0,
        confidence=0.80,
        current_price=float(df["Close"].iloc[-1]),
    )

    for strat in strategies:
        sig = strat.generate_signal(df, current_idx=-1, prediction=pred)
        assert sig is not None
        assert sig.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert 0.0 <= sig.strength <= 1.0
        assert isinstance(sig.reason, str)
