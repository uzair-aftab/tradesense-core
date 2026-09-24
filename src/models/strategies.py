"""
TradeSense Institutional Multi-Strategy Engine & Tournament Arena
==================================================================

Implements top-tier hedge-fund quantitative trading strategies:
1. DeepQuantileSniper — ML Quantile Asymmetry + Whale Confluence
2. InstitutionalWhaleFlow — Large-Block Order Flow + VWAP Anchoring
3. VolatilityBreakout — Donchian Channel Expansion + ADX Trend Acceleration
4. StatisticalMeanReversion — Bollinger Band 2.5-Sigma Extremes + RSI Exhaustion
5. EnsembleAlpha — Meta-Consensus Voting Algorithm

Provides the StrategyArena tournament runner for parallel strategy evaluation.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.models.ai_engine import AIEngine, QuantilePrediction
from src.models.trader import Gauntlet, Side, TradeOrder
from src.utils.logger import get_logger

logger = get_logger("hfp.strategies")


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    strategy_name: str
    ticker: str
    signal_type: SignalType
    strength: float           # 0.0 to 1.0 confidence/strength
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class BaseStrategy(ABC):
    """Abstract base class for all institutional trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        """Evaluate market data and generate a trading signal."""
        pass


class DeepQuantileSniper(BaseStrategy):
    """
    Strategy 1: Deep Quantile Sniper.
    Exploits asymmetric probabilistic forecast distributions from LSTM + XGBoost.
    Requires RRR >= 1.5x and model confidence >= 70%.
    """

    def __init__(self, min_conf: float = 0.70, min_rrr: float = 1.5):
        super().__init__("Deep Quantile Sniper")
        self.min_conf = min_conf
        self.min_rrr = min_rrr

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if df.empty or prediction is None:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="No prediction available")

        price = float(df["Close"].iloc[current_idx])
        if prediction.direction == "BULLISH" and prediction.confidence >= self.min_conf:
            stop = float(min(prediction.p10, price * 0.99))
            target = float(max(prediction.p90, price * 1.01))
            upside = target - price
            downside = price - stop
            rrr = upside / max(0.01, downside)
            if rrr >= self.min_rrr:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.BUY,
                    strength=prediction.confidence,
                    target_price=target,
                    stop_price=stop,
                    reason=f"Quantile Asymmetry RRR={rrr:.2f}x | Conf={prediction.confidence:.0%}",
                )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Conditions not met")


class InstitutionalWhaleFlow(BaseStrategy):
    """
    Strategy 2: Institutional Whale Flow Tracker.
    Capitalizes on smart-money block order accumulation when Volume Z-score >= 2.5
    and price trades above VWAP with positive volume momentum.
    """

    def __init__(self, z_threshold: float = 2.5):
        super().__init__("Institutional Whale Flow")
        self.z_threshold = z_threshold

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if df.empty or len(df) < 5:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        bar = df.iloc[current_idx]
        price = float(bar["Close"])
        whale_z = float(bar.get("Whale_Z", 0.0))
        vwap = float(bar.get("VWAP", price))
        atr = float(bar.get("ATR_14", price * 0.02))

        # Whale accumulation check: strong volume spike trading above VWAP
        if whale_z >= self.z_threshold and price >= vwap:
            stop = price - 1.5 * atr
            target = price + 3.0 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=min(1.0, whale_z / 4.0),
                target_price=target,
                stop_price=stop,
                reason=f"Whale volume surge Z={whale_z:.2f} above VWAP",
            )
        elif whale_z >= self.z_threshold and price < vwap:
            # Whale distribution / dump
            stop = price + 1.5 * atr
            target = price - 3.0 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=min(1.0, whale_z / 4.0),
                target_price=target,
                stop_price=stop,
                reason=f"Whale liquidation Z={whale_z:.2f} below VWAP",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Normal volume")


class VolatilityBreakout(BaseStrategy):
    """
    Strategy 3: Donchian Volatility Breakout.
    Enters trend continuation on 20-bar channel high break with ADX > 22 confirmation.
    """

    def __init__(self, lookback: int = 20, adx_threshold: float = 22.0):
        super().__init__("Donchian Volatility Breakout")
        self.lookback = lookback
        self.adx_threshold = adx_threshold

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if len(df) < self.lookback + 5:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        recent = df.iloc[: current_idx + 1 if current_idx != -1 else len(df)]
        bar = recent.iloc[-1]
        price = float(bar["Close"])
        highs = recent["High"].iloc[-self.lookback - 1 : -1]
        highest_high = float(highs.max()) if not highs.empty else price
        adx = float(bar.get("ADX_14", 20.0))
        atr = float(bar.get("ATR_14", price * 0.02))
        bb_u = float(bar.get("BB_Upper", price * 1.02))
        bb_l = float(bar.get("BB_Lower", price * 0.98))
        bb_b = (price - bb_l) / max(0.01, (bb_u - bb_l))

        # Breakout Gating: strictly reject if market is chopping (ADX < threshold)
        # or if buying directly into an unconfirmed extreme ceiling without huge momentum
        if price > highest_high:
            if adx < self.adx_threshold:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.HOLD,
                    0.0,
                    reason=f"Breakout suppressed in chop (ADX={adx:.1f} < {self.adx_threshold:.1f})",
                )
            if bb_b > 0.90 and adx < 28.0:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.HOLD,
                    0.0,
                    reason=f"Breakout suppressed at ceiling (BB%B={bb_b*100:.1f}%, ADX={adx:.1f})",
                )

            stop = price - 2.0 * atr
            target = price + 4.0 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=min(1.0, max(0.70, adx / 35.0)),
                target_price=target,
                stop_price=stop,
                reason=f"Breakout above {self.lookback}-bar high (${highest_high:.2f}) | ADX={adx:.1f}",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Within channel")


class StatisticalMeanReversion(BaseStrategy):
    """
    Strategy 4: Statistical Mean Reversion.
    Fades 2-sigma Bollinger Band extensions when RSI < 35 (oversold) or RSI > 65 (overbought).
    """

    def __init__(self):
        super().__init__("Statistical Mean Reversion")

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        if len(df) < 25:
            return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Insufficient history")

        bar = df.iloc[current_idx]
        price = float(bar["Close"])
        lower_bb = float(bar.get("BB_Lower", price * 0.98))
        upper_bb = float(bar.get("BB_Upper", price * 1.02))
        mid_bb = float(bar.get("BB_Mid", price))
        rsi = float(bar.get("RSI_14", 50.0))
        atr = float(bar.get("ATR_14", price * 0.01))

        # Oversold bounce: Price touching lower band AND RSI oversold (< 38)
        if price <= lower_bb and rsi <= 38.0:
            target = max(mid_bb, price + 2.0 * atr)
            stop = price - 1.5 * atr
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=0.80,
                target_price=target,
                stop_price=stop,
                reason=f"Mean reversion floor bounce (${lower_bb:.2f}) | RSI={rsi:.1f}",
            )
        elif price >= upper_bb and rsi >= 62.0:
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=0.75,
                target_price=mid_bb,
                stop_price=price + 1.5 * atr,
                reason=f"Mean reversion ceiling fade (${upper_bb:.2f}) | RSI={rsi:.1f}",
            )

        return StrategySignal(self.name, ticker, SignalType.HOLD, 0.0, reason="Price inside bands")


class EnsembleAlpha(BaseStrategy):
    """
    Strategy 5: Ensemble Multi-Strategy Alpha.
    Aggregates consensus votes across all sub-strategies with dynamic market regime gating.
    """

    def __init__(self, strategies: Optional[List[BaseStrategy]] = None):
        super().__init__("Ensemble Multi-Strategy Alpha")
        self.strategies = strategies or [
            DeepQuantileSniper(),
            InstitutionalWhaleFlow(),
            VolatilityBreakout(),
            StatisticalMeanReversion(),
        ]
        from src.models.regime import RegimeClassifier
        self.regime_classifier = RegimeClassifier()
        self.last_regime_context = None

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
        prediction: Optional[QuantilePrediction] = None,
    ) -> StrategySignal:
        ticker = "ASSET"
        price = float(df["Close"].iloc[current_idx]) if not df.empty else 100.0

        # 1. Evaluate Active Market Regime
        regime_ctx = self.regime_classifier.classify(df, current_idx)
        self.last_regime_context = regime_ctx

        # 2. Hard Cash Preservation Guard in Bear Trends
        if regime_ctx.cash_preservation_active:
            self.last_sub_signals = []
            self.last_buy_votes = 0.0
            self.last_sell_votes = 1.0
            return StrategySignal(
                self.name,
                ticker,
                SignalType.HOLD,
                0.0,
                reason=f"CASH PRESERVATION LOCK: {regime_ctx.reason}",
            )

        # 3. Query Sub-Strategies
        signals = [s.generate_signal(df, current_idx, prediction) for s in self.strategies]
        self.last_sub_signals = signals

        # 4. Regime-Adaptive Conviction Weighting
        buy_votes = 0.0
        sell_votes = 0.0
        from src.models.regime import MarketRegime

        for s in signals:
            strat_name = s.strategy_name
            weight = 1.0

            if regime_ctx.regime == MarketRegime.BULL_TREND:
                # In Bull Trend: amplify breakout & whale flow; mute mean reversion fades
                if strat_name in ("Volatility Breakout", "Donchian Volatility Breakout"):
                    weight = 1.3
                elif strat_name == "Institutional Whale Flow":
                    weight = 1.2
                elif strat_name == "Statistical Mean Reversion":
                    weight = 0.2
            elif regime_ctx.regime == MarketRegime.RANGING_CHOP:
                # In Range Chop: suppress breakouts; amplify mean reversion
                if strat_name in ("Volatility Breakout", "Donchian Volatility Breakout"):
                    weight = 0.0  # Zero weight to breakout buys in chop!
                elif strat_name == "Statistical Mean Reversion":
                    weight = 1.5
                elif strat_name == "Institutional Whale Flow":
                    weight = 0.5

            if s.signal_type == SignalType.BUY:
                buy_votes += s.strength * weight
            elif s.signal_type == SignalType.SELL:
                sell_votes += s.strength * weight

        self.last_buy_votes = buy_votes
        self.last_sell_votes = sell_votes

        min_threshold = 0.65 if prediction is None else 0.90

        # In Ranging Chop, never allow BUY if price is at upper half of Bollinger Band
        if regime_ctx.regime == MarketRegime.RANGING_CHOP and buy_votes >= min_threshold:
            if not regime_ctx.allow_mean_reversion_buys:
                return StrategySignal(
                    self.name,
                    ticker,
                    SignalType.HOLD,
                    0.0,
                    reason=f"BUY suppressed in range chop (BB%B={regime_ctx.bb_pct_b*100:.1f}% > 20%)",
                )

        if buy_votes > sell_votes and buy_votes >= min_threshold:
            reasons = [f"{s.strategy_name}: {s.reason}" for s in signals if s.signal_type == SignalType.BUY]
            targets = [s.target_price for s in signals if s.target_price]
            stops = [s.stop_price for s in signals if s.stop_price]
            strength = min(1.0, max(0.70, buy_votes / len(self.strategies) * 1.5))
            return StrategySignal(
                self.name,
                ticker,
                SignalType.BUY,
                strength=strength,
                target_price=float(np.mean(targets)) if targets else price * 1.03,
                stop_price=float(np.mean(stops)) if stops else price * 0.98,
                reason=f"[{regime_ctx.regime.value}] " + " | ".join(reasons[:2]),
            )
        elif sell_votes > buy_votes and sell_votes >= min_threshold:
            reasons = [f"{s.strategy_name}: {s.reason}" for s in signals if s.signal_type == SignalType.SELL]
            strength = min(1.0, max(0.70, sell_votes / len(self.strategies) * 1.5))
            return StrategySignal(
                self.name,
                ticker,
                SignalType.SELL,
                strength=strength,
                target_price=price * 0.97,
                stop_price=price * 1.02,
                reason=f"[{regime_ctx.regime.value}] " + " | ".join(reasons[:2]),
            )

        return StrategySignal(
            self.name,
            ticker,
            SignalType.HOLD,
            0.0,
            reason=f"Consensus below threshold ({buy_votes:.2f} / {min_threshold:.2f}) [{regime_ctx.regime.value}]",
        )



class StrategyArena:
    """
    Tournament arena for backtesting and ranking all institutional strategies
    head-to-head on the active asset.
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.strategies: List[BaseStrategy] = [
            DeepQuantileSniper(),
            InstitutionalWhaleFlow(),
            VolatilityBreakout(),
            StatisticalMeanReversion(),
            EnsembleAlpha(),
        ]

    def run_tournament(
        self,
        df: pd.DataFrame,
        ticker: str = "ASSET",
        prediction: Optional[QuantilePrediction] = None,
    ) -> pd.DataFrame:
        """
        Execute parallel backtests of all strategies and return a ranked leaderboard.
        """
        results: List[Dict[str, Any]] = []

        if df.empty or len(df) < 30:
            return pd.DataFrame()

        close_prices = df["Close"].values
        benchmark_return = ((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100.0

        for strat in self.strategies:
            sim = Gauntlet(
                starting_capital=self.initial_capital,
                slippage_range=(0.0001, 0.0005),
                commission_flat=1.00,
                commission_pct=0.001,
                persist=False,
            )

            # Rolling forward test
            step = max(1, len(df) // 100)  # test across up to 100 points
            for i in range(25, len(df), step):
                chunk = df.iloc[: i + 1]
                sig = strat.generate_signal(chunk, current_idx=-1, prediction=prediction)
                price = float(df["Close"].iloc[i])

                if sig.signal_type == SignalType.BUY and ticker not in sim.positions:
                    qty = sim.calculate_position_size(
                        price=price,
                        stop_price=sig.stop_price or price * 0.98,
                        risk_pct=0.02,
                    )
                    if qty > 0:
                        sim.submit_bracket_order(
                            TradeOrder(ticker=ticker, side=Side.BUY, quantity=qty, price=price),
                            stop_loss=sig.stop_price,
                            take_profit=sig.target_price,
                        )
                elif sig.signal_type == SignalType.SELL and ticker in sim.positions:
                    pos = sim.positions[ticker]
                    sim.submit_order(TradeOrder(ticker=ticker, side=Side.SELL, quantity=pos.quantity, price=price))

                sim.mark_to_market({ticker: price})

            # Final close of any open positions
            if ticker in sim.positions:
                pos = sim.positions[ticker]
                sim.submit_order(TradeOrder(ticker=ticker, side=Side.SELL, quantity=pos.quantity, price=close_prices[-1]))
                sim.mark_to_market({ticker: close_prices[-1]})

            m = sim.metrics
            strat_ret = ((m.total_equity - self.initial_capital) / self.initial_capital) * 100.0
            alpha = strat_ret - benchmark_return

            results.append({
                "Strategy": strat.name,
                "Return %": round(strat_ret, 2),
                "Alpha %": round(alpha, 2),
                "Sharpe": round(m.sharpe_ratio, 2),
                "Max DD %": round(m.max_drawdown_pct, 2),
                "Win Rate %": round(m.win_rate * 100, 1),
                "Profit Factor": round(m.profit_factor, 2),
                "Total Trades": m.total_trades,
                "Equity ($)": round(m.total_equity, 2),
            })

        board = pd.DataFrame(results)
        # Rank by Sharpe Ratio, then Return
        board = board.sort_values(by=["Sharpe", "Return %"], ascending=False).reset_index(drop=True)
        board.index = board.index + 1  # 1-based rank
        board.index.name = "Rank"
        return board
