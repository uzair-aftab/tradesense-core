"""
TradeSense Signals Module

Provides trading signal generation and management.
"""

from src.signals.generator import (
    SignalGenerator,
    TradingSignal,
    SignalType,
    SignalWeights,
    signal_generator,
)

__all__ = [
    "SignalGenerator",
    "TradingSignal",
    "SignalType",
    "SignalWeights",
    "signal_generator",
]
