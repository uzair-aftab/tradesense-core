import numpy as np
import pandas as pd
import pytest

from src.models.regime import MarketRegime, RegimeClassifier, RegimeContext


def test_regime_classifier_insufficient_data():
    clf = RegimeClassifier()
    df_empty = pd.DataFrame()
    ctx = clf.classify(df_empty)
    assert ctx.regime == MarketRegime.RANGING_CHOP
    assert ctx.allow_breakout_buys is False
    assert ctx.allow_mean_reversion_buys is False


def test_regime_classifier_bull_trend():
    clf = RegimeClassifier(adx_trend_threshold=22.0)
    # Generate clean upward trend
    n = 60
    closes = [100.0 * (1.005 ** i) for i in range(n)]
    highs = [c * 1.002 for c in closes]
    lows = [c * 0.998 for c in closes]
    opens = [closes[max(0, i - 1)] for i in range(n)]

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(n, 1000.0),
        }
    )
    # Inject indicators
    df["EMA_9"] = df["Close"].ewm(span=9).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    df["RSI_14"] = 65.0
    df["ATR_14"] = 1.0

    ctx = clf.classify(df)
    assert ctx.regime == MarketRegime.BULL_TREND
    assert ctx.cash_preservation_active is False
    assert ctx.allow_breakout_buys is True
    assert ctx.allow_mean_reversion_buys is False


def test_regime_classifier_bear_trend_cash_lock():
    clf = RegimeClassifier(adx_trend_threshold=22.0)
    # Generate downward trend
    n = 60
    closes = [100.0 * (0.995 ** i) for i in range(n)]
    highs = [c * 1.002 for c in closes]
    lows = [c * 0.998 for c in closes]
    opens = [closes[max(0, i - 1)] for i in range(n)]

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(n, 1000.0),
        }
    )
    df["EMA_9"] = df["Close"].ewm(span=9).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    df["RSI_14"] = 35.0
    df["ATR_14"] = 1.0

    ctx = clf.classify(df)
    assert ctx.regime == MarketRegime.BEAR_TREND
    assert ctx.cash_preservation_active is True
    assert ctx.allow_breakout_buys is False
    assert "CASH LOCK" in ctx.reason


def test_regime_classifier_ranging_chop():
    clf = RegimeClassifier(adx_trend_threshold=25.0)
    # Generate oscillating sideways chop
    n = 60
    closes = [100.0 + (1.5 if i % 2 == 0 else -1.5) for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    opens = [closes[max(0, i - 1)] for i in range(n)]

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(n, 1000.0),
        }
    )
    df["EMA_9"] = df["Close"].ewm(span=9).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["BB_Mid"] = 100.0
    df["BB_Upper"] = 103.0
    df["BB_Lower"] = 97.0
    df["RSI_14"] = 50.0
    df["ATR_14"] = 1.0

    ctx = clf.classify(df)
    assert ctx.regime == MarketRegime.RANGING_CHOP
    assert ctx.allow_breakout_buys is False
