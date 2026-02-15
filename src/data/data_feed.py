"""
HFP Data Lake — High-Frequency Prediction Data Feed
====================================================

Institutional-grade data ingestion and feature engineering pipeline.

Responsibilities
----------------
1. Fetch multi-timeframe OHLCV bars from yfinance (1m, 5m, 1h, 1d).
2. Engineer technical features via pandas-ta:
   • Trend   — ADX, Ichimoku Cloud, MACD
   • Vol     — ATR, Bollinger Band Width
   • Volume  — OBV, VWAP
   • Custom  — "Whale Tracker" (anomalous volume vs 20-day average)
3. Attach FinBERT sentiment (GPU) with VADER fallback, normalised to [-1, 1].
4. Clean NaN rows and MinMax-scale for downstream LSTM consumption.
5. Expose an AsyncIO-friendly interface for streaming refreshes.

Usage
-----
    from src.data.data_feed import DataLake

    lake = DataLake()
    df   = lake.fetch("AAPL", timeframe="1d", lookback="6mo")
    feat = lake.engineer_features(df)
    norm = lake.normalise(feat)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("hfp.data_feed")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Timeframe = Literal["1m", "5m", "15m", "1h", "1d"]
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------------
# yfinance interval → max lookback mapping (API constraints)
# ---------------------------------------------------------------------------
_MAX_LOOKBACK: Dict[str, int] = {
    "1m":  7,        # 7 calendar days
    "5m":  60,
    "15m": 60,
    "1h":  730,
    "1d":  3650,
}

# ---------------------------------------------------------------------------
# Whale-Tracker configuration
# ---------------------------------------------------------------------------
WHALE_WINDOW: int = 20          # rolling window (days / bars)
WHALE_Z_THRESHOLD: float = 2.0  # z-score trigger


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DataLake — Primary Interface                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@dataclass
class DataLake:
    """
    Central data-ingestion and feature-engineering hub.

    Parameters
    ----------
    sentiment_backend : str
        ``"finbert"`` (default, GPU-accelerated) or ``"vader"`` (lightweight).
    whale_z_threshold : float
        Z-score above which a bar is flagged as a whale event.
    scaler : MinMaxScaler | None
        Reusable scaler instance (persisted across retrains).
    """

    sentiment_backend: Literal["finbert", "vader"] = "finbert"
    whale_z_threshold: float = WHALE_Z_THRESHOLD
    scaler: Optional[MinMaxScaler] = None
    _sentiment_engine: Optional[object] = field(default=None, repr=False, init=False)

    # ── public ────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker: str,
        timeframe: Timeframe = "1d",
        lookback: str = "6mo",
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch raw OHLCV data from yfinance.

        Parameters
        ----------
        ticker : str
            Equity / ETF symbol (e.g. ``"AAPL"``).
        timeframe : Timeframe
            Bar interval — ``1m | 5m | 15m | 1h | 1d``.
        lookback : str
            yfinance period string when *start/end* are not given.
        start, end : str | None
            ISO date strings for explicit date-range fetch.

        Returns
        -------
        pd.DataFrame
            Clean OHLCV DataFrame indexed by datetime.
        """
        logger.info("Fetching %s  tf=%s  lookback=%s", ticker, timeframe, lookback)

        tkr = yf.Ticker(ticker)

        if start and end:
            df = tkr.history(interval=timeframe, start=start, end=end, auto_adjust=True)
        else:
            df = tkr.history(interval=timeframe, period=lookback, auto_adjust=True)

        if df.empty:
            logger.warning("No data returned for %s (%s / %s)", ticker, timeframe, lookback)
            return pd.DataFrame(columns=OHLCV_COLS)

        # Normalise column names (yfinance sometimes changes capitalisation)
        df.columns = [c.strip().title() for c in df.columns]
        for col in OHLCV_COLS:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' missing after fetch — got {list(df.columns)}")

        df = df[OHLCV_COLS].copy()
        df.dropna(subset=["Close", "Volume"], inplace=True)

        logger.info("Fetched %d bars for %s", len(df), ticker)
        return df

    # ------------------------------------------------------------------
    # Async wrapper — allows `await lake.afetch(...)` from event loops
    # ------------------------------------------------------------------
    async def afetch(
        self,
        ticker: str,
        timeframe: Timeframe = "1d",
        lookback: str = "6mo",
        **kwargs,
    ) -> pd.DataFrame:
        """Non-blocking wrapper around :meth:`fetch` for async pipelines."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.fetch(ticker, timeframe, lookback, **kwargs))

    async def afetch_batch(
        self,
        tickers: List[str],
        timeframe: Timeframe = "1d",
        lookback: str = "6mo",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple tickers concurrently."""
        tasks = [self.afetch(t, timeframe, lookback) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, pd.DataFrame] = {}
        for t, r in zip(tickers, results):
            if isinstance(r, Exception):
                logger.error("Failed to fetch %s: %s", t, r)
            else:
                out[t] = r
        return out

    # ══════════════════════════════════════════════════════════════════════
    #  Feature Engineering
    # ══════════════════════════════════════════════════════════════════════

    def engineer_features(
        self,
        df: pd.DataFrame,
        *,
        ticker: Optional[str] = None,
        include_sentiment: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the full HFP feature set and append columns in-place.

        Categories
        ----------
        * **Trend** —  ADX, Ichimoku cloud spans, MACD (line / signal / histogram)
        * **Volatility** — ATR, Bollinger Band width
        * **Volume** — OBV, VWAP
        * **Custom** — Whale Tracker (volume anomaly z-score + boolean flag)
        * **Sentiment** — FinBERT or VADER score for *ticker* (optional)

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV bars from :meth:`fetch`.
        ticker : str | None
            If given *and* ``include_sentiment=True``, attach sentiment.
        include_sentiment : bool
            Toggle sentiment column generation.

        Returns
        -------
        pd.DataFrame
            Original OHLCV + all derived feature columns.
        """
        if df.empty:
            logger.warning("engineer_features received empty DataFrame — skipping")
            return df

        out = df.copy()

        # ── Trend ─────────────────────────────────────────────────────
        out = self._add_trend(out)

        # ── Volatility ────────────────────────────────────────────────
        out = self._add_volatility(out)

        # ── Volume ────────────────────────────────────────────────────
        out = self._add_volume(out)

        # ── Whale Tracker ─────────────────────────────────────────────
        out = self._add_whale_tracker(out)

        # ── Sentiment ─────────────────────────────────────────────────
        if include_sentiment and ticker:
            out = self._add_sentiment(out, ticker)

        logger.info(
            "Engineered %d features  (%d rows, %d cols)",
            len(out.columns) - len(OHLCV_COLS),
            len(out),
            len(out.columns),
        )
        return out

    # ══════════════════════════════════════════════════════════════════════
    #  Normalisation
    # ══════════════════════════════════════════════════════════════════════

    def normalise(
        self,
        df: pd.DataFrame,
        *,
        drop_na: bool = True,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        MinMax-scale all numeric columns to [0, 1].

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (output of :meth:`engineer_features`).
        drop_na : bool
            If True, forward-fill then drop remaining NaN rows.
        fit : bool
            If True, fit a *new* scaler.  Set False at inference time
            to reuse the scaler fitted during training.

        Returns
        -------
        pd.DataFrame
            Scaled copy of *df* with the same index and column names.
        """
        out = df.copy()

        if drop_na:
            out.ffill(inplace=True)
            out.bfill(inplace=True)
            out.dropna(inplace=True)

        if out.empty:
            logger.warning("normalise: DataFrame is empty after NaN handling")
            return out

        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return out

        if fit or self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            out[numeric_cols] = self.scaler.fit_transform(out[numeric_cols])
        else:
            out[numeric_cols] = self.scaler.transform(out[numeric_cols])

        logger.info("Normalised %d columns  (%d rows)", len(numeric_cols), len(out))
        return out

    # ══════════════════════════════════════════════════════════════════════
    #  Pipeline convenience
    # ══════════════════════════════════════════════════════════════════════

    def pipeline(
        self,
        ticker: str,
        timeframe: Timeframe = "1d",
        lookback: str = "6mo",
        *,
        include_sentiment: bool = True,
        normalise: bool = True,
    ) -> pd.DataFrame:
        """
        End-to-end: fetch → engineer → normalise.

        Returns the ready-to-train (or ready-to-infer) feature matrix.
        """
        raw = self.fetch(ticker, timeframe, lookback)
        feat = self.engineer_features(raw, ticker=ticker, include_sentiment=include_sentiment)
        if normalise:
            return self.normalise(feat)
        return feat

    async def apipeline(
        self,
        ticker: str,
        timeframe: Timeframe = "1d",
        lookback: str = "6mo",
        **kwargs,
    ) -> pd.DataFrame:
        """Async version of :meth:`pipeline`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.pipeline(ticker, timeframe, lookback, **kwargs))

    # ══════════════════════════════════════════════════════════════════════
    #  PRIVATE — feature helpers
    # ══════════════════════════════════════════════════════════════════════

    # ── Trend ──────────────────────────────────────────────────────────

    @staticmethod
    def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
        """ADX, Ichimoku Cloud, MACD."""
        # ADX — Average Directional Index (trend strength)
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        if adx is not None and not adx.empty:
            df = pd.concat([df, adx], axis=1)

        # Ichimoku Cloud
        ichi = ta.ichimoku(df["High"], df["Low"], df["Close"])
        if ichi is not None:
            # ichimoku returns a tuple of (lines_df, span_df)
            if isinstance(ichi, tuple):
                for part in ichi:
                    if part is not None and isinstance(part, pd.DataFrame) and not part.empty:
                        # Avoid duplicate column names
                        overlap = part.columns.intersection(df.columns)
                        if len(overlap):
                            part = part.drop(columns=overlap)
                        df = pd.concat([df, part], axis=1)
            elif isinstance(ichi, pd.DataFrame) and not ichi.empty:
                overlap = ichi.columns.intersection(df.columns)
                if len(overlap):
                    ichi = ichi.drop(columns=overlap)
                df = pd.concat([df, ichi], axis=1)

        # MACD (12, 26, 9)
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)

        return df

    # ── Volatility ─────────────────────────────────────────────────────

    @staticmethod
    def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Band Width."""
        # ATR — Average True Range
        atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        if atr is not None:
            df["ATR_14"] = atr

        # Bollinger Bands → derive Band Width
        bbands = ta.bbands(df["Close"], length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            upper_col = [c for c in bbands.columns if "BBU" in c]
            lower_col = [c for c in bbands.columns if "BBL" in c]
            mid_col = [c for c in bbands.columns if "BBM" in c]

            if upper_col and lower_col and mid_col:
                bbu = bbands[upper_col[0]]
                bbl = bbands[lower_col[0]]
                bbm = bbands[mid_col[0]]
                df["BB_Width"] = (bbu - bbl) / bbm  # normalised width
                df["BB_Upper"] = bbu
                df["BB_Lower"] = bbl
                df["BB_Mid"] = bbm

        return df

    # ── Volume ─────────────────────────────────────────────────────────

    @staticmethod
    def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
        """OBV, VWAP."""
        # OBV — On-Balance Volume
        obv = ta.obv(df["Close"], df["Volume"])
        if obv is not None:
            df["OBV"] = obv

        # VWAP — Volume Weighted Average Price
        vwap = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        if vwap is not None:
            df["VWAP"] = vwap

        return df

    # ── Whale Tracker (custom) ─────────────────────────────────────────

    def _add_whale_tracker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag anomalous volume spikes relative to a 20-bar rolling window.

        Columns added
        -------------
        * ``Whale_Vol_Mean``  — 20-bar rolling mean volume
        * ``Whale_Vol_Std``   — 20-bar rolling std volume
        * ``Whale_Z``         — z-score of current volume vs rolling stats
        * ``Whale_Flag``      — boolean, True when z > threshold
        """
        vol = df["Volume"].astype(float)
        roll_mean = vol.rolling(window=WHALE_WINDOW, min_periods=1).mean()
        roll_std = vol.rolling(window=WHALE_WINDOW, min_periods=1).std().fillna(1.0)

        # Prevent division by zero
        roll_std = roll_std.replace(0, 1.0)

        z_score = (vol - roll_mean) / roll_std

        df["Whale_Vol_Mean"] = roll_mean
        df["Whale_Vol_Std"] = roll_std
        df["Whale_Z"] = z_score
        df["Whale_Flag"] = (z_score > self.whale_z_threshold).astype(int)

        whale_count = df["Whale_Flag"].sum()
        if whale_count > 0:
            logger.info("🐋 Whale Tracker: %d anomalous volume bars detected", whale_count)

        return df

    # ── Sentiment ──────────────────────────────────────────────────────

    def _add_sentiment(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Attach a ``Sentiment`` column (float, -1.0 to +1.0) to every row.

        Backend precedence:
            1. FinBERT (GPU-accelerated) via ``transformers``
            2. VADER (rule-based fallback)
        """
        try:
            score = self._get_sentiment_score(ticker)
        except Exception as exc:
            logger.warning("Sentiment fetch failed for %s: %s — defaulting to 0.0", ticker, exc)
            score = 0.0

        df["Sentiment"] = score  # broadcast scalar to every row
        return df

    def _get_sentiment_score(self, ticker: str) -> float:
        """
        Return a single aggregated sentiment score for *ticker*.

        Tries FinBERT first; falls back to VADER.
        """
        headlines = self._scrape_headlines(ticker)
        if not headlines:
            logger.debug("No headlines found for %s", ticker)
            return 0.0

        if self.sentiment_backend == "finbert":
            return self._score_finbert(headlines)
        return self._score_vader(headlines)

    # ── headline scraping ──────────────────────────────────────────────

    @staticmethod
    def _scrape_headlines(ticker: str, max_items: int = 20) -> List[str]:
        """
        Pull recent news headlines for *ticker* via yfinance.

        Returns up to *max_items* headline strings (no API key needed).
        """
        try:
            tkr = yf.Ticker(ticker)
            news = tkr.news
            if not news:
                return []
            headlines = [
                item.get("title", "")
                for item in news[:max_items]
                if item.get("title")
            ]
            return headlines
        except Exception as exc:
            logger.warning("Headline scrape error for %s: %s", ticker, exc)
            return []

    # ── FinBERT scoring ────────────────────────────────────────────────

    def _score_finbert(self, headlines: List[str]) -> float:
        """
        Score headlines with ProsusAI/finbert on GPU (if available).

        Returns mean sentiment in [-1, 1]:
            positive → +1, neutral → 0, negative → -1.
        """
        if self._sentiment_engine is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                model_name = "ProsusAI/finbert"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading FinBERT on %s …", device)

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.to(device)
                model.eval()

                self._sentiment_engine = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "device": device,
                    "labels": ["positive", "negative", "neutral"],
                }
                logger.info("FinBERT loaded ✓  (device=%s)", device)

            except ImportError:
                logger.warning("transformers/torch not available — falling back to VADER")
                self.sentiment_backend = "vader"
                return self._score_vader(headlines)

        engine = self._sentiment_engine
        tokenizer = engine["tokenizer"]
        model = engine["model"]
        device = engine["device"]
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

        import torch

        scores: List[float] = []
        # Batch headlines for efficiency
        batch_size = 16
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            for j, pred_idx in enumerate(preds):
                label = engine["labels"][pred_idx.item()]
                confidence = probs[j, pred_idx].item()
                # Weight by confidence
                scores.append(label_map[label] * confidence)

        return float(np.clip(np.mean(scores), -1.0, 1.0)) if scores else 0.0

    # ── VADER scoring (fallback) ───────────────────────────────────────

    @staticmethod
    def _score_vader(headlines: List[str]) -> float:
        """
        Rule-based sentiment via VADER (no GPU needed).

        Returns mean compound score in [-1, 1].
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            logger.error("vaderSentiment not installed — returning 0.0")
            return 0.0

        analyzer = SentimentIntensityAnalyzer()
        compounds: List[float] = []

        for headline in headlines:
            vs = analyzer.polarity_scores(headline)
            compounds.append(vs["compound"])

        return float(np.clip(np.mean(compounds), -1.0, 1.0)) if compounds else 0.0

    # ══════════════════════════════════════════════════════════════════════
    #  Utility helpers
    # ══════════════════════════════════════════════════════════════════════

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return only the derived feature columns (excludes raw OHLCV)."""
        return [c for c in df.columns if c not in OHLCV_COLS]

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological train/test split (no shuffling — time-series safe).
        """
        split_idx = int(len(df) * (1 - test_ratio))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    @staticmethod
    def rolling_window_indices(
        total_len: int,
        train_window: int,
        test_window: int,
        step: int = 1,
    ) -> List[Tuple[range, range]]:
        """
        Generate (train_range, test_range) index pairs for walk-forward
        rolling window retraining.

        Parameters
        ----------
        total_len : int
            Total number of bars.
        train_window : int
            Number of bars in each training fold.
        test_window : int
            Number of bars in each test fold.
        step : int
            Stride between folds.

        Returns
        -------
        list[tuple[range, range]]
        """
        folds: List[Tuple[range, range]] = []
        start = 0
        while start + train_window + test_window <= total_len:
            train_r = range(start, start + train_window)
            test_r = range(start + train_window, start + train_window + test_window)
            folds.append((train_r, test_r))
            start += step
        return folds

    def __repr__(self) -> str:
        return (
            f"DataLake(sentiment={self.sentiment_backend!r}, "
            f"whale_z={self.whale_z_threshold}, "
            f"scaler={'fitted' if self.scaler else 'None'})"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Quick smoke-test (python -m src.data.data_feed)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    lake = DataLake(sentiment_backend="vader")

    print("=" * 60)
    print("  HFP Data Lake — Smoke Test")
    print("=" * 60)

    # Fetch daily bars
    raw = lake.fetch("AAPL", timeframe="1d", lookback="6mo")
    print(f"\n📊 Raw bars:  {raw.shape}")
    print(raw.tail(3))

    # Engineer features (skip sentiment for speed)
    feat = lake.engineer_features(raw, ticker="AAPL", include_sentiment=False)
    print(f"\n🔧 Feature cols: {lake.get_feature_columns(feat)}")
    print(f"   Shape: {feat.shape}")

    # Whale flags
    whale_bars = feat[feat["Whale_Flag"] == 1]
    print(f"\n🐋 Whale bars:  {len(whale_bars)}")

    # Normalise
    norm = lake.normalise(feat)
    print(f"\n📐 Normalised shape: {norm.shape}")
    print(norm.tail(3))

    # Rolling window demo
    folds = lake.rolling_window_indices(len(norm), train_window=60, test_window=10, step=10)
    print(f"\n🔁 Rolling folds: {len(folds)}")
    print(f"   First fold — train {folds[0][0].start}:{folds[0][0].stop}, test {folds[0][1].start}:{folds[0][1].stop}")

    print("\n✅ Smoke test passed.")
