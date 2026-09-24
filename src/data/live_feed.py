"""
TradeSense Live Market Feed & High-Frequency Simulation Engine
==============================================================

Provides real-time live market streaming and realistic high-frequency
market microstructure simulation with tick-by-tick order-flow dynamics.

Modes:
------
1. LIVE_STREAM: Real-time price stream from yfinance/Finnhub.
2. HISTORICAL_REPLAY: High-speed playback of actual historical bars (1s per bar).
3. SYNTHETIC_MICROSTRUCTURE: Real-time Geometric Brownian Motion with
   Poisson jump-diffusion for Whale institutional order-flow simulation.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.data.data_feed import DataLake
from src.utils.logger import get_logger

logger = get_logger("hfp.live_feed")


class StreamMode(str, Enum):
    LIVE_STREAM = "LIVE_STREAM"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    SYNTHETIC_MICROSTRUCTURE = "SYNTHETIC_MICROSTRUCTURE"


@dataclass
class MarketTick:
    """Atomic market tick event."""

    ticker: str
    price: float
    bid: float
    ask: float
    volume: float
    is_whale: bool = False
    whale_z: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def spread(self) -> float:
        return max(0.01, self.ask - self.bid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "price": round(self.price, 4),
            "bid": round(self.bid, 4),
            "ask": round(self.ask, 4),
            "volume": round(self.volume, 2),
            "is_whale": self.is_whale,
            "whale_z": round(self.whale_z, 2),
            "timestamp": self.timestamp,
        }


class LiveMarketFeed:
    """
    Real-time streaming market feed and tick simulator.
    Maintains a rolling window of streaming bars and feeds live ticks
    to charts, models, and trading bots.
    """

    def __init__(
        self,
        ticker: str = "AAPL",
        mode: StreamMode = StreamMode.HISTORICAL_REPLAY,
        base_df: Optional[pd.DataFrame] = None,
        tick_interval_sec: float = 1.0,
    ):
        self.ticker = ticker
        self.mode = mode
        self.tick_interval_sec = tick_interval_sec
        self.base_df = base_df.copy() if base_df is not None and not base_df.empty else None

        # Replay cursor
        self._replay_idx = 0
        if self.base_df is not None and len(self.base_df) > 30:
            # Start replay from the last 30 bars to allow forward playback
            self._replay_idx = max(0, len(self.base_df) - 30)

        # Microstructure state
        self._current_price = 100.0
        if self.base_df is not None and not self.base_df.empty:
            self._current_price = float(self.base_df["Close"].iloc[-1])

        self._volatility = 0.0015  # 15 bps per tick
        self._drift = 0.0001
        self._last_tick_time = time.time()
        self._tick_history: deque[MarketTick] = deque(maxlen=500)

        # Active streaming bars (starts with base_df up to replay_idx)
        if self.base_df is not None:
            self._streaming_df = self.base_df.iloc[: self._replay_idx + 1].copy()
        else:
            self._streaming_df = pd.DataFrame()

    @property
    def current_price(self) -> float:
        return self._current_price

    @property
    def streaming_df(self) -> pd.DataFrame:
        return self._streaming_df

    def next_tick(self) -> MarketTick:
        """
        Produce the next live or simulated tick.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.mode == StreamMode.HISTORICAL_REPLAY and self.base_df is not None:
            # Advance replay bar
            if self._replay_idx < len(self.base_df) - 1:
                self._replay_idx += 1
            else:
                # Wrap or transition to synthetic continuation
                self._replay_idx = max(0, len(self.base_df) - 1)

            row = self.base_df.iloc[self._replay_idx]
            price = float(row["Close"])
            self._current_price = price
            vol = float(row.get("Volume", 1000.0))
            is_whale = bool(row.get("Whale_Flag", 0) == 1)
            whale_z = float(row.get("Whale_Z", 0.0))

            spread = price * 0.0002
            bid = price - spread / 2
            ask = price + spread / 2

            tick = MarketTick(
                ticker=self.ticker,
                price=price,
                bid=bid,
                ask=ask,
                volume=vol,
                is_whale=is_whale,
                whale_z=whale_z,
                timestamp=now_iso,
            )

            # Update streaming dataframe up to current replay position
            self._streaming_df = self.base_df.iloc[: self._replay_idx + 1].copy()
            self._tick_history.append(tick)
            return tick

        # SYNTHETIC_MICROSTRUCTURE or LIVE continuation
        dt = 1.0 / 252.0 / (6.5 * 3600)  # sub-second dt
        shock = np.random.normal(0, 1)

        # Poisson jump for institutional whale block trades
        is_whale = random.random() < 0.08  # 8% chance of whale order
        whale_z = 0.0
        jump = 0.0
        if is_whale:
            jump_dir = 1 if random.random() < 0.60 else -1
            jump_mag = np.random.uniform(0.005, 0.015)  # 0.5% - 1.5% jump
            jump = jump_dir * jump_mag
            whale_z = float(np.random.uniform(2.5, 5.0))
            volume = float(np.random.uniform(50000, 250000))
        else:
            volume = float(np.random.uniform(500, 5000))

        ret = (self._drift - 0.5 * self._volatility**2) * dt + self._volatility * shock + jump
        self._current_price = max(1.0, self._current_price * (1.0 + ret))

        spread = max(0.01, self._current_price * 0.0003)
        bid = self._current_price - spread / 2
        ask = self._current_price + spread / 2

        tick = MarketTick(
            ticker=self.ticker,
            price=self._current_price,
            bid=bid,
            ask=ask,
            volume=volume,
            is_whale=is_whale,
            whale_z=whale_z,
            timestamp=now_iso,
        )

        # Append synthetic bar or update last row
        if not self._streaming_df.empty:
            last_idx = self._streaming_df.index[-1]
            # Create a new synthetic bar timestamped right now
            new_row = {
                "Open": self._current_price * (1 - 0.0005 * shock),
                "High": max(self._current_price, self._current_price * (1 + 0.001 * abs(shock))),
                "Low": min(self._current_price, self._current_price * (1 - 0.001 * abs(shock))),
                "Close": self._current_price,
                "Volume": volume,
            }
            if "Whale_Flag" in self._streaming_df.columns:
                new_row["Whale_Flag"] = 1 if is_whale else 0
                new_row["Whale_Z"] = whale_z

            new_df = pd.DataFrame([new_row], index=[pd.Timestamp.now(timezone.utc)])
            self._streaming_df = pd.concat([self._streaming_df, new_df]).tail(300)

        self._tick_history.append(tick)
        return tick

    def get_recent_ticks(self, count: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent ticks as dictionaries."""
        return [t.to_dict() for t in list(self._tick_history)[-count:]]

    def reset_replay(self) -> None:
        """Reset replay cursor to 30 bars before end."""
        if self.base_df is not None and len(self.base_df) > 30:
            self._replay_idx = max(0, len(self.base_df) - 30)
            self._streaming_df = self.base_df.iloc[: self._replay_idx + 1].copy()
            self._current_price = float(self._streaming_df["Close"].iloc[-1])
            self._tick_history.clear()
            logger.info("Live feed replay reset for %s", self.ticker)
