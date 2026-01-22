"""
Unit tests for technical analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from src.analysis.technical import (
    RSIIndicator,
    MACDIndicator,
    BollingerBands,
    VolumeAnalyzer,
    TechnicalSignalGenerator
)
from src.analysis.support_resistance import SupportResistance


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    
    # Create uptrend then downtrend
    close = np.concatenate([
        np.linspace(100, 150, 50),
        np.linspace(150, 100, 50)
    ])
    
    # Add some noise
    np.random.seed(42)
    noise = np.random.normal(0, 2, 100)
    close += noise
    
    data = pd.DataFrame({
        "Open": close,
        "High": close + 5,
        "Low": close - 5,
        "Close": close,
        "Volume": np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    return data


def test_rsi_indicator(sample_data):
    """Test RSI calculation and signals."""
    rsi = RSIIndicator(period=14, oversold=30, overbought=70)
    df = rsi.calculate(sample_data)
    
    assert "RSI" in df.columns
    assert not df["RSI"].isnull().all()
    
    # Check signal generation
    # Create a new testing dataframe to avoid SettingWithCopy and logic issues
    test_df = df.copy()
    test_df.loc[test_df.index[-1], "RSI"] = 25  # Force oversold
    
    signal = rsi.get_signal(test_df)
    assert signal["signal"] is True, f"Expected True (Buy) for RSI 25, got {signal['signal']}"
    assert signal["condition"].startswith("Oversold")
    
    test_df.loc[test_df.index[-1], "RSI"] = 75  # Force overbought
    signal = rsi.get_signal(test_df)
    assert signal["signal"] is False, f"Expected False (Sell) for RSI 75, got {signal['signal']}"
    assert signal["condition"].startswith("Overbought")


def test_macd_indicator(sample_data):
    """Test MACD calculation and signals."""
    macd = MACDIndicator()
    df = macd.calculate(sample_data)
    
    # Check for expected columns (pandas-ta naming)
    # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    cols = [c for c in df.columns if c.startswith("MACD")]
    assert len(cols) >= 3
    
    # Test signal
    signal = macd.get_signal(df)
    assert "signal" in signal
    assert "histogram" in signal


def test_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    bb = BollingerBands()
    df = bb.calculate(sample_data)
    
    # Check for BB columns
    bb_cols = [c for c in df.columns if c.startswith("BB")]
    assert len(bb_cols) >= 3  # Lower, Mid, Upper
    
    signal = bb.get_signal(df)
    assert "percent_b" in signal


def test_support_resistance(sample_data):
    """Test S/R level detection."""
    sr = SupportResistance()
    levels = sr.find_levels(sample_data)
    
    assert "current_price" in levels
    assert "nearest_support" in levels
    assert "nearest_resistance" in levels
    assert "pivot_points" in levels
    
    # Pivot points should exist
    assert levels["pivot_points"]["pivot"] > 0


def test_technical_signal_generator(sample_data):
    """Test composite signal generation."""
    gen = TechnicalSignalGenerator()
    result = gen.analyze(sample_data)
    
    assert "technical_score" in result
    assert "composite_signal" in result
    assert 0.0 <= result["technical_score"] <= 1.0
