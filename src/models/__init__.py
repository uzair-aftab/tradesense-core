"""
HFP Models — Hybrid Ensemble Prediction Engine & Trading Simulator

Modules:
    ai_engine   – Stacking ensemble (LSTM + XGBoost) with quantile output
    trader      – "The Gauntlet" event-driven paper trading simulator
"""

from src.models.ai_engine import AIEngine, QuantilePrediction, TrainResult
from src.models.trader import Gauntlet, TradeOrder, Side, Fill, PortfolioMetrics

__all__ = [
    "AIEngine",
    "QuantilePrediction",
    "TrainResult",
    "Gauntlet",
    "TradeOrder",
    "Side",
    "Fill",
    "PortfolioMetrics",
]
