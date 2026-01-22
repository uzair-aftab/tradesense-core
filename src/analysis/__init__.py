"""
TradeSense Analysis Module

Provides technical analysis indicators and signal generation.
"""

from src.analysis.technical import (
    RSIIndicator,
    MACDIndicator,
    BollingerBands,
    VolumeAnalyzer,
    TechnicalSignalGenerator,
    rsi_indicator,
    macd_indicator,
    bollinger_bands,
    volume_analyzer,
    technical_signal_generator,
)
from src.analysis.support_resistance import (
    SupportResistance,
    support_resistance,
)
from src.analysis.adaptive import AdaptiveThresholds, adaptive_thresholds

__all__ = [
    # Indicators
    "RSIIndicator",
    "MACDIndicator",
    "BollingerBands",
    "VolumeAnalyzer",
    "TechnicalSignalGenerator",
    "SupportResistance",
    # Singleton instances
    "rsi_indicator",
    "macd_indicator",
    "bollinger_bands",
    "volume_analyzer",
    "technical_signal_generator",
    "support_resistance",
]
