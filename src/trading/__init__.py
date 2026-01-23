"""
Trading Module

Provides paper trading functionality for testing signals with virtual money.
"""

from src.trading.paper_trading import paper_trader, PaperTradingEngine, Trade, Position, Portfolio

__all__ = ['paper_trader', 'PaperTradingEngine', 'Trade', 'Position', 'Portfolio']
