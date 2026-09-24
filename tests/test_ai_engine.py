"""
Tests for AIEngine (LSTM + XGBoost + Stacking Meta-Learner)
"""

import shutil
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import pytest

from src.config import config
from src.data.data_feed import DataLake
from src.models.ai_engine import AIEngine, QuantilePrediction


@pytest.fixture
def trained_engine_and_features():
    """Train a fast lightweight AIEngine on synthetic features."""
    dates = pd.date_range("2026-01-01", periods=150, freq="1h")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(150) * 0.5)
    high = close + np.random.uniform(0.1, 1.0, 150)
    low = close - np.random.uniform(0.1, 1.0, 150)
    open_p = close + np.random.uniform(-0.5, 0.5, 150)
    volume = np.random.randint(1000, 10000, 150).astype(float)

    raw_df = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )

    lake = DataLake()
    feat_df = lake.engineer_features(raw_df)

    engine = AIEngine(
        seq_len=10,
        lstm_hidden=16,
        lstm_layers=1,
        lstm_epochs=5,
        xgb_estimators=15,
        test_ratio=0.2,
    )
    result = engine.train(feat_df, target_col="Close")
    return engine, feat_df, result


def test_ai_engine_train_metrics(trained_engine_and_features):
    """Verify engine training returns valid performance and health metrics."""
    engine, feat_df, result = trained_engine_and_features

    assert result.meta_mae > 0.0
    assert result.relative_mae_pct is not None
    assert result.health_color in ("green", "amber", "red")
    assert engine._trained is True


def test_ai_engine_quantile_monotonicity(trained_engine_and_features):
    """Verify non-crossing quantile property: P10 <= P50 <= P90."""
    engine, feat_df, _ = trained_engine_and_features

    pred = engine.predict(feat_df)
    assert isinstance(pred, QuantilePrediction)
    assert pred.p10 <= pred.p50, f"Quantile crossing: P10 ({pred.p10}) > P50 ({pred.p50})"
    assert pred.p50 <= pred.p90, f"Quantile crossing: P50 ({pred.p50}) > P90 ({pred.p90})"
    assert 0.0 <= pred.confidence <= 1.0, f"Invalid confidence score: {pred.confidence}"
    assert pred.direction in ("BULLISH", "BEARISH", "NEUTRAL")


def test_ai_engine_predict_series(trained_engine_and_features):
    """Verify predict_series produces consistent predictions across rolling window."""
    engine, feat_df, _ = trained_engine_and_features

    series = engine.predict_series(feat_df, horizon=5)
    assert len(series) > 0
    for p in series:
        assert p.p10 <= p.p50 <= p.p90


def test_ai_engine_save_and_reload(trained_engine_and_features):
    """Verify full save and reload cycle preserves prediction weights and outputs."""
    engine, feat_df, _ = trained_engine_and_features

    tag = "test_pytest_cycle"
    save_path = engine.save(tag=tag)
    assert save_path.exists()

    new_engine = AIEngine()
    new_engine.load(tag=tag)

    pred_original = engine.predict(feat_df)
    pred_reloaded = new_engine.predict(feat_df)

    assert np.isclose(pred_original.p50, pred_reloaded.p50, atol=1e-3)
    assert np.isclose(pred_original.p10, pred_reloaded.p10, atol=1e-3)
    assert np.isclose(pred_original.p90, pred_reloaded.p90, atol=1e-3)

    # Cleanup test model directory
    shutil.rmtree(save_path, ignore_errors=True)
