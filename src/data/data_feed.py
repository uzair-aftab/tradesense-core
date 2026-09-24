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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.cache import DataCache
from src.utils.logger import get_logger

logger = get_logger("hfp.data_feed")

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
    use_cache : bool
        If True, cache fetch() results in SQLite for TTL-based reuse.
    cache_ttl : int
        Cache time-to-live in seconds (default 1 hour).
    max_retries : int
        Number of retries on fetch failure (exponential backoff).
    """

    sentiment_backend: Literal["finbert", "vader"] = "finbert"
    whale_z_threshold: float = WHALE_Z_THRESHOLD
    scaler: Optional[MinMaxScaler] = None
    use_cache: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    _sentiment_engine: Optional[object] = field(default=None, repr=False, init=False)
    _cache: DataCache = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self.use_cache:
            self._cache = DataCache()

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
        # ── Check cache first ──────────────────────────────────────
        cache_key = f"fetch:{ticker}:{timeframe}:{lookback}"
        if self.use_cache and self._cache and not (start or end):
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache HIT for %s (%d bars)", cache_key, len(cached))
                return cached

        logger.info("Fetching %s  tf=%s  lookback=%s", ticker, timeframe, lookback)

        # ── Fetch with retry logic ────────────────────────────────
        df = pd.DataFrame()
        for attempt in range(1, self.max_retries + 1):
            try:
                tkr = yf.Ticker(ticker)
                if start and end:
                    df = tkr.history(interval=timeframe, start=start, end=end, auto_adjust=True)
                else:
                    df = tkr.history(interval=timeframe, period=lookback, auto_adjust=True)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Fetch attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt, self.max_retries, ticker, exc, wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    logger.error("All %d fetch attempts failed for %s", self.max_retries, ticker)
                    return pd.DataFrame(columns=OHLCV_COLS)

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

        # ── Store in cache ────────────────────────────────────────
        if self.use_cache and self._cache and not (start or end):
            self._cache.set(cache_key, df, ttl=self.cache_ttl)
            logger.info("Cached %d bars for %s (TTL=%ds)", len(df), ticker, self.cache_ttl)

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

    def scan_watchlist(
        self,
        tickers: List[str],
        timeframe: Timeframe = "1d",
        lookback: str = "3mo",
    ) -> pd.DataFrame:
        """
        Scan multiple tickers to produce an institutional radar matrix:
        Price, 1-Bar Return %, Volatility, Whale Z-Score, Whale Alert,
        Trend Regime, and Sentiment Score.
        """
        results: List[Dict[str, Any]] = []

        for ticker in tickers:
            try:
                raw = self.fetch(ticker, timeframe, lookback)
                if raw.empty or len(raw) < 5:
                    continue

                feat = self.engineer_features(raw, ticker=ticker, include_sentiment=False)
                last_bar = feat.iloc[-1]
                prev_bar = feat.iloc[-2]

                price = float(last_bar["Close"])
                prev_price = float(prev_bar["Close"])
                change_pct = ((price - prev_price) / prev_price) * 100.0

                # 20-period volatility / ATR %
                atr = float(last_bar.get("ATR_14", 0.0))
                vol_pct = (atr / price * 100.0) if price > 0 else 0.0

                # Whale activity: maximum Z-score in the last 3 bars
                recent_z = float(feat["Whale_Z"].tail(3).max()) if "Whale_Z" in feat.columns else 0.0
                whale_alert = bool(feat["Whale_Flag"].tail(3).any()) if "Whale_Flag" in feat.columns else False

                # Trend Regime: Fast vs Slow EMA
                ema20 = float(feat["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                ema50 = float(feat["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                if price > ema20 > ema50:
                    regime = "Strong Bullish"
                elif price > ema20:
                    regime = "Bullish"
                elif price < ema20 < ema50:
                    regime = "Strong Bearish"
                elif price < ema20:
                    regime = "Bearish"
                else:
                    regime = "Neutral"

                # Sentiment score
                sentiment_score = self._get_sentiment_score(ticker)

                results.append({
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "Change %": round(change_pct, 2),
                    "Vol %": round(vol_pct, 2),
                    "Whale Z": round(recent_z, 2),
                    "Whale Alert": "🚨 WHALE" if whale_alert else "Normal",
                    "Regime": regime,
                    "Sentiment": round(sentiment_score, 2),
                })
            except Exception as exc:
                logger.warning("Error scanning ticker %s: %s", ticker, exc)

        if not results:
            return pd.DataFrame()

        scan_df = pd.DataFrame(results)
        scan_df = scan_df.sort_values(by="Whale Z", ascending=False).reset_index(drop=True)
        return scan_df

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
            # ichimoku returns a tuple of (lines_df, span_df).
            # span_df projects 26 periods into the future. Reindex strictly to df.index
            # so no future phantom rows are added to df.
            if isinstance(ichi, tuple):
                for part in ichi:
                    if part is not None and isinstance(part, pd.DataFrame) and not part.empty:
                        aligned = part.reindex(df.index)
                        overlap = aligned.columns.intersection(df.columns)
                        if len(overlap):
                            aligned = aligned.drop(columns=overlap)
                        df = pd.concat([df, aligned], axis=1)
            elif isinstance(ichi, pd.DataFrame) and not ichi.empty:
                aligned = ichi.reindex(df.index)
                overlap = aligned.columns.intersection(df.columns)
                if len(overlap):
                    aligned = aligned.drop(columns=overlap)
                df = pd.concat([df, aligned], axis=1)

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

    @staticmethod
    def _scrape_headlines(ticker: str, max_items: int = 20) -> List[str]:
        """
        Pull recent news headlines for *ticker* via Finnhub (if key present)
        or yfinance fallback.

        Returns up to *max_items* headline strings.
        """
        headlines: List[str] = []

        # 1. Try Finnhub first if API key is configured
        try:
            from src.config import config
            if config.has_finnhub_key():
                import finnhub
                fc = finnhub.Client(api_key=config.FINNHUB_API_KEY)
                today = datetime.now(timezone.utc).date()
                from_date = (today - timedelta(days=7)).isoformat()
                to_date = today.isoformat()
                fh_news = fc.company_news(ticker, _from=from_date, to=to_date)
                if fh_news:
                    for item in fh_news[:max_items]:
                        hl = item.get("headline")
                        if hl:
                            headlines.append(hl)
                    if headlines:
                        logger.debug("Fetched %d headlines from Finnhub for %s", len(headlines), ticker)
                        return headlines
        except Exception as exc:
            logger.debug("Finnhub headline fetch error for %s: %s", ticker, exc)

        # 2. Fallback to yfinance
        try:
            tkr = yf.Ticker(ticker)
            news = tkr.news
            if not news:
                return []
            for item in news[:max_items]:
                # Handle both yfinance 1.0 (item['content']['title']) and legacy (item['title'])
                title = item.get("title")
                if not title and isinstance(item.get("content"), dict):
                    title = item["content"].get("title")
                if title:
                    headlines.append(title)
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
    import sys
    from pathlib import Path
    _root = str(Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
