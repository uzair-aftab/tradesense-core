"""
Tests for DataLake and Feature Engineering Pipeline
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import pytest

from src.data.data_feed import DataLake


@pytest.fixture
def sample_ohlcv():
    """Create a realistic synthetic OHLCV dataframe with 100 bars."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.random.uniform(0.1, 1.0, 100)
    low = close - np.random.uniform(0.1, 1.0, 100)
    open_p = close + np.random.uniform(-0.5, 0.5, 100)
    volume = np.random.randint(1000, 10000, 100).astype(float)
    # Inject one anomalous whale volume spike
    volume[50] = 500000.0

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def test_feature_engineering_row_count_integrity(sample_ohlcv):
    """Verify feature engineering produces exactly len(raw) rows without future leakage."""
    lake = DataLake()
    feat = lake.engineer_features(sample_ohlcv)

    # Must preserve exact row count and datetime index
    assert len(feat) == len(sample_ohlcv), f"Expected {len(sample_ohlcv)} rows, got {len(feat)}"
    assert feat.index.equals(sample_ohlcv.index), "Index timestamps altered by indicator calculation"


def test_ichimoku_no_ghost_rows(sample_ohlcv):
    """Verify Ichimoku Cloud does not leak 26 forward projection periods into DataFrame."""
    lake = DataLake()
    feat = lake.engineer_features(sample_ohlcv)

    # Check for presence of Ichimoku columns
    ichimoku_cols = [c for c in feat.columns if "ISA" in c or "ISB" in c or "ITS" in c or "IKS" in c]
    assert len(ichimoku_cols) > 0, "Ichimoku features should be present"

    # Ensure last index timestamp matches input raw dataframe
    assert feat.index[-1] == sample_ohlcv.index[-1]


def test_whale_tracker_anomalous_volume(sample_ohlcv):
    """Verify Whale Tracker flags extreme volume spike at index 50."""
    lake = DataLake()
    feat = lake.engineer_features(sample_ohlcv)

    assert "Whale_Flag" in feat.columns
    assert "Whale_Z" in feat.columns

    # Index 50 was injected with 500,000 volume vs average ~5,000
    assert feat["Whale_Flag"].iloc[50] == 1, "Whale_Flag should trigger on anomalous volume spike"
    assert feat["Whale_Z"].iloc[50] > 3.0, "Z-score for whale volume should exceed threshold"


def test_sentiment_scoring():
    """Verify VADER sentiment scoring calculates appropriate polarity scores."""
    lake = DataLake()
    headlines = [
        "Company reports record-breaking quarterly revenue and exceptional growth",
        "Analyst upgrades stock to strong buy on stellar performance",
    ]
    score = lake._score_vader(headlines)
    assert score > 0.0, f"Expected positive sentiment score, got {score}"


def test_scan_watchlist(monkeypatch, sample_ohlcv):
    """Verify scan_watchlist produces ranking matrix with whale and trend regimes."""
    lake = DataLake()
    monkeypatch.setattr(lake, "fetch", lambda ticker, tf, lb: sample_ohlcv)
    monkeypatch.setattr(lake, "_get_sentiment_score", lambda ticker: 0.45)

    df_scan = lake.scan_watchlist(["NVDA", "AAPL"])
    assert not df_scan.empty
    assert len(df_scan) == 2
    assert "Whale Z" in df_scan.columns
    assert "Regime" in df_scan.columns
    assert "Sentiment" in df_scan.columns
    assert "Whale Alert" in df_scan.columns

