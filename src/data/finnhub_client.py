"""
TradeSense v8 — Finnhub Free-Tier Client & Optional FRED Macro Hook
===================================================================
Provides rate-limited (50 req/min guard), cached access to Finnhub REST v1
free-tier endpoints (/news, /company-news, /calendar/earnings) — explicitly
avoiding paid/403 /calendar/economic — plus an optional zero-dependency httpx
hook for St. Louis Fed FRED series when FRED_API_KEY is configured.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta, timezone
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.config import config
from src.utils.logger import get_logger

logger = get_logger("hfp.finnhub")

# Normalize Canadian TSX / NEO CDR tickers so Finnhub company-news finds rich headlines
FINNHUB_SYMBOL_MAP: Dict[str, Tuple[str, str]] = {
    "SHOP.TO": ("SHOP", "equity"),
    "RY.TO": ("RY", "equity"),
    "TD.TO": ("TD", "equity"),
    "ENB.TO": ("ENB", "equity"),
    "CNR.TO": ("CNI", "equity"),
    "CCO.TO": ("CCJ", "equity"),
    "CLS.TO": ("CLS", "equity"),
    "VFV.TO": ("SPY", "equity"),
    "XIU.TO": ("EWC", "equity"),
    "XQQ.TO": ("QQQ", "equity"),
    "BTCC.TO": ("BITO", "crypto"),
    "BTCX-B.TO": ("BITO", "crypto"),
    "ETHX-B.TO": ("ETHA", "crypto"),
    "AAPL.NE": ("AAPL", "equity"),
    "NVDA.NE": ("NVDA", "equity"),
    "TSLA.NE": ("TSLA", "equity"),
    "MSFT.NE": ("MSFT", "equity"),
    "AMZN.NE": ("AMZN", "equity"),
    "GOOGL.NE": ("GOOGL", "equity"),
    "AMD.NE": ("AMD", "equity"),
    "META.NE": ("META", "equity"),
}


class FinnhubClient:
    """Thread-safe, rate-limited Finnhub v1 API client with TTL caching."""

    BASE_URL = "https://finnhub.io/api/v1"
    MAX_CALLS_PER_MIN = 50

    def __init__(self, cache_ttl_sec: int = 240):
        self.cache_ttl_sec = cache_ttl_sec
        self._lock = threading.Lock()
        self._call_times: deque[float] = deque()
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._client = httpx.Client(timeout=6.0)

    def is_configured(self) -> bool:
        return config.has_finnhub_key()

    def _allow_call(self) -> bool:
        now = time.time()
        with self._lock:
            while self._call_times and (now - self._call_times[0]) > 60.0:
                self._call_times.popleft()
            if len(self._call_times) >= self.MAX_CALLS_PER_MIN:
                return False
            self._call_times.append(now)
            return True

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> Any:
        if not self.is_configured():
            return None

        req_params = dict(params or {})
        cache_key = f"{path}:{sorted(req_params.items())}"
        use_ttl = self.cache_ttl_sec if ttl is None else ttl
        now = time.time()

        with self._lock:
            hit = self._cache.get(cache_key)
            if hit and (now - hit[0]) < use_ttl:
                return hit[1]

        if not self._allow_call():
            logger.debug("Finnhub local rate guard triggered for %s", path)
            return hit[1] if hit else None

        try:
            req_params["token"] = config.FINNHUB_API_KEY.strip()
            resp = self._client.get(f"{self.BASE_URL}{path}", params=req_params)
            if resp.status_code == 429:
                logger.warning("Finnhub rate limit (429) on %s", path)
                return hit[1] if hit else None
            if resp.status_code >= 400:
                logger.debug("Finnhub HTTP %d on %s", resp.status_code, path)
                return None
            payload = resp.json()
            with self._lock:
                self._cache[cache_key] = (now, payload)
            return payload
        except Exception as exc:
            logger.debug("Finnhub request error (%s): %s", path, exc)
            return None

    @staticmethod
    def _normalize_news_items(raw_items: Any, max_items: int) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("headline") or "").strip()
            if not headline:
                continue
            pub_ts = float(item.get("datetime") or 0)
            if pub_ts <= 0:
                pub_ts = time.time() - 1800
            out.append({
                "title": headline,
                "summary": str(item.get("summary") or "").strip(),
                "publisher": str(item.get("source") or "Finnhub Wire").strip(),
                "url": str(item.get("url") or "").strip() or "https://finnhub.io",
                "pub_ts": pub_ts,
                "category": str(item.get("category") or "general").strip(),
            })
            if len(out) >= max_items:
                break
        return out

    def get_market_news(self, category: str = "general", max_items: int = 30) -> List[Dict[str, Any]]:
        """Fetch live general, forex, or crypto headlines via GET /news."""
        cat = category.lower().strip()
        if cat not in ("general", "forex", "crypto", "merger"):
            cat = "general"
        raw = self._get("/news", {"category": cat}, ttl=240)
        return self._normalize_news_items(raw, max_items=max_items)

    def get_company_news(self, symbol: str, days: int = 7, max_items: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent headlines for a symbol (routes crypto/Canadian proxies smartly)."""
        sym = symbol.upper().strip()
        if "-CAD" in sym or "-USD" in sym:
            coin = sym.split("-")[0]
            crypto_news = self.get_market_news("crypto", max_items=40)
            matched = [a for a in crypto_news if coin.lower() in f"{a['title']} {a['summary']}".lower()]
            return (matched or crypto_news)[:max_items]

        mapped_sym, kind = FINNHUB_SYMBOL_MAP.get(sym, (sym.split(".")[0], "equity"))
        if kind == "crypto":
            return self.get_market_news("crypto", max_items=max_items)

        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=max(1, days))
        raw = self._get(
            "/company-news",
            {"symbol": mapped_sym, "from": start.isoformat(), "to": today.isoformat()},
            ttl=240,
        )
        return self._normalize_news_items(raw, max_items=max_items)

    def get_earnings_calendar(self, days_ahead: int = 14) -> List[Dict[str, Any]]:
        """Fetch upcoming corporate earnings releases via GET /calendar/earnings."""
        today = date.today()
        end = today + timedelta(days=max(1, days_ahead))
        raw = self._get(
            "/calendar/earnings",
            {"from": today.isoformat(), "to": end.isoformat()},
            ttl=900,
        )
        if not isinstance(raw, dict):
            return []
        events = raw.get("earningsCalendar") or []
        return [e for e in events if isinstance(e, dict) and e.get("symbol")]


def fetch_fred_latest(series_id: str) -> Optional[Dict[str, Any]]:
    """
    Optional FRED hook: returns latest observation for CPIAUCSL, FEDFUNDS, PCEPI,
    UNRATE, or GDPC1 if FRED_API_KEY is configured in .env; otherwise returns None.
    """
    if not config.has_fred_key():
        return None
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": config.FRED_API_KEY.strip(),
                "file_type": "json",
                "sort_order": "desc",
                "limit": 2,
            },
            timeout=5.0,
        )
        if resp.status_code != 200:
            return None
        obs = resp.json().get("observations") or []
        if not obs:
            return None
        latest = obs[0]
        prev = obs[1] if len(obs) > 1 else None
        val = float(latest["value"]) if latest.get("value") not in (None, ".") else None
        prev_val = float(prev["value"]) if prev and prev.get("value") not in (None, ".") else None
        return {
            "series_id": series_id,
            "date": latest.get("date"),
            "value": val,
            "prev_value": prev_val,
            "delta": round(val - prev_val, 4) if (val is not None and prev_val is not None) else None,
        }
    except Exception as exc:
        logger.debug("Optional FRED fetch skipped for %s: %s", series_id, exc)
        return None


finnhub_client = FinnhubClient()
