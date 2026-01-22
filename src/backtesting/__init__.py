"""
TradeSense Backtesting Module

Provides backtesting capabilities using vectorbt.
"""

from src.backtesting.engine import (
    BacktestEngine,
    backtest_engine,
)

__all__ = [
    "BacktestEngine",
    "backtest_engine",
]
