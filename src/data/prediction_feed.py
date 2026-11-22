"""
TradeSense v8 — Live Kalshi / Wealthsimple Predict Feed & Macro Calendar
=========================================================================
Connects to Kalshi's unauthenticated public REST v2 API (which powers
Wealthsimple Predict in Canada) using the 2026 fixed-point (*_dollars / *_fp)
schema, enforces Canadian CIRO category filters (Economics, Financials,
Climate & Weather — zero sports/politics), tracks 30+ day term eligibility,
parses binary YES/NO orderbooks and probability candlesticks, and builds a live
Macro Event Calendar without hitting Finnhub's paid /calendar/economic route.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.config import config
from src.utils.logger import get_logger

logger = get_logger("hfp.prediction_feed")

KALSHI_PRIMARY_BASE = (config.KALSHI_API_BASE or "https://api.elections.kalshi.com/trade-api/v2").rstrip("/")
KALSHI_FALLBACK_BASE = "https://external-api.kalshi.com/trade-api/v2"

# Canadian CIRO-permitted prediction market categories on Wealthsimple Predict
ALLOWED_WS_CATEGORIES = {"Economics", "Financials", "Climate and Weather"}
BLOCKED_CATEGORIES = {"Sports", "Politics", "Elections", "Entertainment", "Mentions"}

FLAGSHIP_SERIES: List[Dict[str, Any]] = [
    # Central Banks
    {
        "series_ticker": "KXFEDDECISION",
        "label": "Fed FOMC Decision",
        "ui_category": "CENTRAL_BANKS",
        "domain": "FED_RATES",
        "underlying_yf_symbol": "^IRX",
        "keywords": ["fed", "fomc", "powell", "federal reserve", "interest rate", "rate cut", "rate hike"],
    },
    {
        "series_ticker": "KXFED",
        "label": "Fed Funds Rate",
        "ui_category": "CENTRAL_BANKS",
        "domain": "FED_RATES",
        "underlying_yf_symbol": "^IRX",
        "keywords": ["fed funds", "fomc", "interest rate", "powell", "federal reserve"],
    },
    {
        "series_ticker": "KXRATECUTCOUNT",
        "label": "Fed Rate Cut Count",
        "ui_category": "CENTRAL_BANKS",
        "domain": "FED_RATES",
        "underlying_yf_symbol": "^IRX",
        "keywords": ["rate cut", "rate cuts", "fomc", "fed", "powell"],
    },
    {
        "series_ticker": "KXCBDECISIONCANADA",
        "label": "Bank of Canada Decision",
        "ui_category": "CENTRAL_BANKS",
        "domain": "BOC_RATES",
        "underlying_yf_symbol": "CAD=X",
        "keywords": ["bank of canada", "boc", "macklem", "canada rate", "canadian dollar", "canada inflation"],
    },
    # Macro Economic Releases
    {
        "series_ticker": "KXCPI",
        "label": "US CPI MoM Inflation",
        "ui_category": "MACRO_DATA",
        "domain": "CPI_INFLATION",
        "underlying_yf_symbol": "^TNX",
        "keywords": ["cpi", "inflation", "consumer price", "core cpi", "prices"],
    },
    {
        "series_ticker": "KXCPIYOY",
        "label": "US CPI YoY Inflation",
        "ui_category": "MACRO_DATA",
        "domain": "CPI_INFLATION",
        "underlying_yf_symbol": "^TNX",
        "keywords": ["cpi", "annual inflation", "inflation rate", "consumer price"],
    },
    {
        "series_ticker": "KXPCEHEAD",
        "label": "US PCE Inflation",
        "ui_category": "MACRO_DATA",
        "domain": "CPI_INFLATION",
        "underlying_yf_symbol": "^TNX",
        "keywords": ["pce", "inflation", "fed inflation", "personal consumption"],
    },
    {
        "series_ticker": "KXPAYROLLS",
        "label": "US Non-Farm Payrolls",
        "ui_category": "MACRO_DATA",
        "domain": "GROWTH_JOBS",
        "underlying_yf_symbol": "SPY",
        "keywords": ["payroll", "jobs", "nonfarm", "employment", "hiring", "labor market"],
    },
    {
        "series_ticker": "KXU3",
        "label": "US Unemployment Rate",
        "ui_category": "MACRO_DATA",
        "domain": "GROWTH_JOBS",
        "underlying_yf_symbol": "SPY",
        "keywords": ["unemployment", "jobless", "labor market", "layoffs"],
    },
    {
        "series_ticker": "KXGDP",
        "label": "US Real GDP Growth",
        "ui_category": "MACRO_DATA",
        "domain": "GROWTH_JOBS",
        "underlying_yf_symbol": "SPY",
        "keywords": ["gdp", "economic growth", "recession", "economy"],
    },
    # Financials & Equities / Crypto Strikes
    {
        "series_ticker": "KXINX",
        "label": "S&P 500 Range/Target",
        "ui_category": "FINANCIALS",
        "domain": "FINANCIALS",
        "underlying_yf_symbol": "SPY",
        "keywords": ["s&p 500", "sp500", "wall street", "stocks", "equities"],
    },
    {
        "series_ticker": "KXNASDAQ100",
        "label": "Nasdaq 100 Range",
        "ui_category": "FINANCIALS",
        "domain": "FINANCIALS",
        "underlying_yf_symbol": "QQQ",
        "keywords": ["nasdaq", "tech stocks", "ai", "qqq"],
    },
    {
        "series_ticker": "KXBTC",
        "label": "Bitcoin Price Range",
        "ui_category": "FINANCIALS",
        "domain": "FINANCIALS",
        "underlying_yf_symbol": "BTC-CAD",
        "keywords": ["bitcoin", "btc", "crypto", "etf"],
    },
    {
        "series_ticker": "KXETH",
        "label": "Ethereum Price Range",
        "ui_category": "FINANCIALS",
        "domain": "FINANCIALS",
        "underlying_yf_symbol": "ETH-CAD",
        "keywords": ["ethereum", "eth", "crypto"],
    },
    # Climate & Weather
    {
        "series_ticker": "KXHURCLAND",
        "label": "Hurricane Landfall Climate",
        "ui_category": "CLIMATE",
        "domain": "CLIMATE",
        "underlying_yf_symbol": None,
        "keywords": ["hurricane", "storm", "climate", "weather", "noaa"],
    },
    {
        "series_ticker": "KXHIGHNY",
        "label": "NYC Daily High Temp",
        "ui_category": "CLIMATE",
        "domain": "CLIMATE",
        "underlying_yf_symbol": None,
        "keywords": ["temperature", "weather", "forecast", "climate"],
    },
]

SERIES_META_MAP: Dict[str, Dict[str, Any]] = {s["series_ticker"]: s for s in FLAGSHIP_SERIES}


def _parse_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except Exception:
        return default


def _parse_iso_dt(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        clean = str(ts_str).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        return None


@dataclass
class PredictionMarket:
    """Normalized binary prediction contract supporting Kalshi fixed-point v2 schema."""

    market_ticker: str
    event_ticker: str
    series_ticker: str
    event_title: str
    outcome_label: str
    category: str
    ui_category: str
    domain: str
    status: str
    result: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    prev_price: float
    mid_prob: float
    volume: float
    open_interest: float
    open_time: Optional[str]
    close_time: Optional[str]
    contract_term_days: int
    days_to_close: float
    term_days_30plus: bool
    ws_predict_eligible: bool
    rules_primary: str
    underlying_yf_symbol: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_ticker": self.market_ticker,
            "event_ticker": self.event_ticker,
            "series_ticker": self.series_ticker,
            "event_title": self.event_title,
            "outcome_label": self.outcome_label,
            "category": self.category,
            "ui_category": self.ui_category,
            "domain": self.domain,
            "status": self.status,
            "result": self.result,
            "yes_bid": round(self.yes_bid, 4),
            "yes_ask": round(self.yes_ask, 4),
            "no_bid": round(self.no_bid, 4),
            "no_ask": round(self.no_ask, 4),
            "last_price": round(self.last_price, 4),
            "prev_price": round(self.prev_price, 4),
            "mid_prob": round(self.mid_prob, 4),
            "yes_cents": round(self.yes_ask * 100.0, 1),
            "no_cents": round(self.no_ask * 100.0, 1),
            "volume": round(self.volume, 2),
            "open_interest": round(self.open_interest, 2),
            "open_time": self.open_time,
            "close_time": self.close_time,
            "contract_term_days": self.contract_term_days,
            "days_to_close": round(self.days_to_close, 1),
            "term_days_30plus": self.term_days_30plus,
            "ws_predict_eligible": self.ws_predict_eligible,
            "rules_primary": self.rules_primary,
            "underlying_yf_symbol": self.underlying_yf_symbol,
        }


def parse_kalshi_market(
    m: Dict[str, Any],
    event_meta: Optional[Dict[str, Any]] = None,
    series_cfg: Optional[Dict[str, Any]] = None,
) -> PredictionMarket:
    """Safely parse Kalshi v2 2026 fixed-point (*_dollars / *_fp) market dict."""
    ev = event_meta or {}
    event_ticker = str(m.get("event_ticker") or ev.get("event_ticker") or "").strip()
    series_ticker = str(ev.get("series_ticker") or m.get("series_ticker") or event_ticker.split("-")[0]).strip()
    cfg = series_cfg or SERIES_META_MAP.get(series_ticker, {})

    category = str(ev.get("category") or cfg.get("category") or "Economics").strip()
    ui_category = str(cfg.get("ui_category") or ("FINANCIALS" if category == "Financials" else ("CLIMATE" if "Climate" in category else "MACRO_DATA")))
    domain = str(cfg.get("domain") or ("FINANCIALS" if category == "Financials" else ("CLIMATE" if "Climate" in category else "FED_RATES")))

    # Fixed-point dollar strings preferred; fallback to legacy cents/100 if present
    def _price_field(fp_key: str, legacy_key: str) -> float:
        if m.get(fp_key) not in (None, ""):
            return max(0.0, min(1.0, _parse_float(m.get(fp_key))))
        if m.get(legacy_key) not in (None, ""):
            return max(0.0, min(1.0, _parse_float(m.get(legacy_key)) / 100.0))
        return 0.0

    yes_bid = _price_field("yes_bid_dollars", "yes_bid")
    yes_ask = _price_field("yes_ask_dollars", "yes_ask")
    no_bid = _price_field("no_bid_dollars", "no_bid")
    no_ask = _price_field("no_ask_dollars", "no_ask")
    last_price = _price_field("last_price_dollars", "last_price")
    prev_price = _price_field("previous_price_dollars", "previous_price")

    # Binary symmetry repair if one side of the book is blank
    if no_ask <= 0.0 and yes_bid > 0.0:
        no_ask = round(max(0.01, min(0.99, 1.0 - yes_bid)), 4)
    if no_bid <= 0.0 and yes_ask > 0.0:
        no_bid = round(max(0.01, min(0.99, 1.0 - yes_ask)), 4)
    if yes_ask <= 0.0 and last_price > 0.0:
        yes_ask = min(0.99, max(0.01, last_price))
    if no_ask <= 0.0 and last_price > 0.0:
        no_ask = round(max(0.01, min(0.99, 1.0 - last_price)), 4)

    if yes_bid > 0.0 and yes_ask > 0.0:
        mid_prob = (yes_bid + yes_ask) / 2.0
    elif last_price > 0.0:
        mid_prob = last_price
    elif yes_ask > 0.0:
        mid_prob = yes_ask
    else:
        mid_prob = 0.50

    volume = _parse_float(m.get("volume_fp"), _parse_float(m.get("volume"), 0.0))
    open_interest = _parse_float(m.get("open_interest_fp"), _parse_float(m.get("open_interest"), 0.0))

    open_str = m.get("open_time")
    close_str = m.get("expected_expiration_time") or m.get("close_time") or m.get("latest_expiration_time")
    now_dt = datetime.now(timezone.utc)
    open_dt = _parse_iso_dt(open_str)
    close_dt = _parse_iso_dt(close_str)

    term_days = max(0, (close_dt - open_dt).days) if (open_dt and close_dt) else 0
    days_to_close = max(0.0, (close_dt - now_dt).total_seconds() / 86400.0) if close_dt else 0.0
    term_30plus = term_days >= 30

    is_allowed_cat = (category in ALLOWED_WS_CATEGORIES) and (category not in BLOCKED_CATEGORIES)
    # Core macro/financial series in WS Predict often launch >30d ahead
    ws_eligible = bool(is_allowed_cat and (term_30plus or term_days >= 28 or series_ticker in ("KXFEDDECISION", "KXFED", "KXRATECUTCOUNT", "KXCBDECISIONCANADA", "KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXGDP")))

    outcome_label = str(
        m.get("yes_sub_title")
        or m.get("subtitle")
        or m.get("title")
        or m.get("ticker")
        or ""
    ).strip()

    event_title = str(ev.get("title") or cfg.get("label") or m.get("title") or event_ticker).strip()
    rules = str(m.get("rules_primary") or ev.get("sub_title") or "").strip()

    return PredictionMarket(
        market_ticker=str(m.get("ticker") or ""),
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        event_title=event_title,
        outcome_label=outcome_label,
        category=category,
        ui_category=ui_category,
        domain=domain,
        status=str(m.get("status") or "active"),
        result=str(m.get("result") or ""),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        last_price=last_price,
        prev_price=prev_price,
        mid_prob=mid_prob,
        volume=volume,
        open_interest=open_interest,
        open_time=open_str,
        close_time=close_str,
        contract_term_days=term_days,
        days_to_close=days_to_close,
        term_days_30plus=term_30plus,
        ws_predict_eligible=ws_eligible,
        rules_primary=rules[:450],
        underlying_yf_symbol=cfg.get("underlying_yf_symbol"),
        keywords=list(cfg.get("keywords") or []),
    )


class KalshiPredictionFeed:
    """Thread-safe client & cache for Kalshi public prediction markets."""

    def __init__(self, event_ttl_sec: int = 90, quote_ttl_sec: int = 30):
        self.event_ttl_sec = event_ttl_sec
        self.quote_ttl_sec = quote_ttl_sec
        self._lock = threading.Lock()
        self._flagship_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_req_ts = 0.0
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._client = httpx.Client(timeout=8.0)

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> Optional[Dict[str, Any]]:
        use_ttl = self.event_ttl_sec if ttl is None else ttl
        ckey = f"{path}:{sorted((params or {}).items())}"
        now = time.time()

        with self._lock:
            hit = self._cache.get(ckey)
            if hit and (now - hit[0]) < use_ttl:
                return hit[1]

        # Gentle pacing guard to stay well below Kalshi public read rate limits
        with self._rate_lock:
            elapsed = time.time() - self._last_req_ts
            if elapsed < 0.08:
                time.sleep(0.08 - elapsed)
            self._last_req_ts = time.time()

        for base in (KALSHI_PRIMARY_BASE, KALSHI_FALLBACK_BASE):
            try:
                resp = self._client.get(f"{base}{path}", params=params or {})
                if resp.status_code == 200:
                    data = resp.json()
                    with self._lock:
                        self._cache[ckey] = (time.time(), data)
                    return data
                if resp.status_code == 429:
                    logger.debug("Kalshi 429 backoff on %s", path)
                    time.sleep(0.6)
                    retry = self._client.get(f"{base}{path}", params=params or {})
                    if retry.status_code == 200:
                        data = retry.json()
                        with self._lock:
                            self._cache[ckey] = (time.time(), data)
                        return data
                    return hit[1] if hit else None
            except Exception as exc:
                logger.debug("Kalshi request failed on %s (%s): %s", base, path, exc)
                continue

        return hit[1] if hit else None

    def _fetch_series_events(self, series_cfg: Dict[str, Any], force_refresh: bool = False) -> List[Dict[str, Any]]:
        st = series_cfg["series_ticker"]
        ttl = 0 if force_refresh else self.event_ttl_sec
        data = self._get_json(
            "/events",
            params={"series_ticker": st, "status": "open", "with_nested_markets": "true", "limit": 4},
            ttl=ttl,
        )
        if not data:
            return []

        raw_events = data.get("events") or []
        parsed_events: List[Dict[str, Any]] = []
        for ev in raw_events:
            cat = str(ev.get("category") or "Economics")
            if cat in BLOCKED_CATEGORIES:
                continue
            raw_markets = ev.get("markets") or []
            if not raw_markets:
                continue

            mkts = [parse_kalshi_market(m, event_meta=ev, series_cfg=series_cfg) for m in raw_markets]
            total_vol = sum(m.volume for m in mkts)
            total_oi = sum(m.open_interest for m in mkts)

            # Pick lead market with highest realistic probability or volume
            active_mkts = [m for m in mkts if m.yes_ask > 0.01 and m.yes_ask < 0.99] or mkts
            lead = max(active_mkts, key=lambda m: (m.mid_prob, m.volume))

            parsed_events.append({
                "event_ticker": ev.get("event_ticker"),
                "series_ticker": st,
                "title": ev.get("title") or series_cfg.get("label"),
                "category": cat,
                "ui_category": series_cfg.get("ui_category"),
                "domain": series_cfg.get("domain"),
                "keywords": series_cfg.get("keywords", []),
                "underlying_yf_symbol": series_cfg.get("underlying_yf_symbol"),
                "total_volume": round(total_vol, 2),
                "total_open_interest": round(total_oi, 2),
                "close_time": lead.close_time,
                "days_to_close": round(lead.days_to_close, 1),
                "ws_predict_eligible": any(m.ws_predict_eligible for m in mkts),
                "term_days_30plus": any(m.term_days_30plus for m in mkts),
                "lead_market": lead,
                "markets": mkts,
            })
        return parsed_events

    def _get_all_flagship_raw(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Single-flight fetch of all flagship series events so concurrent requests never double-hit Kalshi."""
        ckey = "flagship_raw_all_v8"
        with self._flagship_lock:
            now = time.time()
            if not force_refresh:
                with self._lock:
                    hit = self._cache.get(ckey)
                    if hit and (now - hit[0]) < self.event_ttl_sec:
                        return list(hit[1])

            all_events: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = [pool.submit(self._fetch_series_events, s, force_refresh) for s in FLAGSHIP_SERIES]
                for fut in as_completed(futs):
                    try:
                        all_events.extend(fut.result())
                    except Exception as exc:
                        logger.debug("Series event batch error: %s", exc)

            with self._lock:
                self._cache[ckey] = (time.time(), all_events)
            return list(all_events)

    def fetch_flagship_events(
        self,
        category: str = "ALL",
        ws_eligible_only: bool = True,
        search_q: str = "",
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch live Wealthsimple-aligned Kalshi flagship events with instant local filtering."""
        cat_filter = (category or "ALL").upper().strip()
        q = (search_q or "").strip().lower()

        all_events = self._get_all_flagship_raw(force_refresh=force_refresh)

        filtered: List[Dict[str, Any]] = []
        for ev in all_events:
            if cat_filter not in ("ALL", "") and ev.get("ui_category") != cat_filter:
                continue
            if ws_eligible_only and not ev.get("ws_predict_eligible"):
                continue
            if q:
                hay = f"{ev['title']} {ev['series_ticker']} {ev['event_ticker']}".lower()
                if q not in hay:
                    continue
            filtered.append(ev)

        # Prioritize active liquidity & near-term macro catalysts
        filtered.sort(key=lambda e: (e["total_volume"] + e["total_open_interest"]), reverse=True)
        return filtered

    def fetch_event_detail(self, event_ticker: str) -> Dict[str, Any]:
        """Fetch an event and all of its child YES/NO outcome contracts."""
        data = self._get_json(f"/events/{event_ticker}", params={"with_nested_markets": "true"}, ttl=30)
        if not data:
            return {"error": f"Event {event_ticker} not found"}

        ev = data.get("event") or data
        raw_markets = data.get("markets") or ev.get("markets") or []
        series_ticker = str(ev.get("series_ticker") or event_ticker.split("-")[0])
        cfg = SERIES_META_MAP.get(series_ticker, {})

        mkts = [parse_kalshi_market(m, event_meta=ev, series_cfg=cfg) for m in raw_markets]
        # Sort by mid_prob descending, placing tradable buckets first
        mkts.sort(key=lambda m: (m.mid_prob, m.volume), reverse=True)
        overround = round(sum(m.mid_prob for m in mkts), 4)

        return {
            "event_ticker": event_ticker,
            "series_ticker": series_ticker,
            "title": ev.get("title") or event_ticker,
            "sub_title": ev.get("sub_title") or "",
            "category": ev.get("category") or "Economics",
            "ui_category": cfg.get("ui_category", "MACRO_DATA"),
            "domain": cfg.get("domain", "FED_RATES"),
            "keywords": cfg.get("keywords", [ev.get("title", "")]),
            "underlying_yf_symbol": cfg.get("underlying_yf_symbol"),
            "mutually_exclusive": bool(ev.get("mutually_exclusive", True)),
            "overround_sum": overround,
            "markets": mkts,
        }

    def fetch_market(self, market_ticker: str, bypass_cache: bool = False) -> Optional[PredictionMarket]:
        """Fetch live fixed-point quote & settlement state for a single market ticker."""
        ttl = 0 if bypass_cache else self.quote_ttl_sec
        data = self._get_json(f"/markets/{market_ticker}", ttl=ttl)
        if not data:
            return None
        m = data.get("market") or data
        if not isinstance(m, dict) or not m.get("ticker"):
            return None
        ev_tkr = str(m.get("event_ticker") or "")
        st = str(m.get("series_ticker") or ev_tkr.split("-")[0])
        cfg = SERIES_META_MAP.get(st, {})
        return parse_kalshi_market(m, event_meta={"event_ticker": ev_tkr, "series_ticker": st, "category": cfg.get("category", "Economics")}, series_cfg=cfg)

    def fetch_orderbook(self, market_ticker: str, depth: int = 8) -> Dict[str, Any]:
        """
        Parse Kalshi v2 orderbook_fp.{yes_dollars, no_dollars} into bid ladders,
        dollar liquidity depth, imbalance skew, and effective spread.
        """
        data = self._get_json(f"/markets/{market_ticker}/orderbook", params={"depth": max(depth, 10)}, ttl=20)
        ob = (data or {}).get("orderbook_fp") or (data or {}).get("orderbook") or {}
        yes_raw = ob.get("yes_dollars") or ob.get("yes") or []
        no_raw = ob.get("no_dollars") or ob.get("no") or []

        def _norm_levels(levels: Any) -> List[Dict[str, float]]:
            parsed = []
            for row in levels:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                p = _parse_float(row[0])
                if p > 1.0:
                    p = p / 100.0
                q = _parse_float(row[1])
                if p > 0 and q > 0:
                    parsed.append({"price": round(p, 4), "qty": round(q, 2), "notional_usd": round(p * q, 2)})
            parsed.sort(key=lambda x: x["price"], reverse=True)
            return parsed[:depth]

        yes_bids = _norm_levels(yes_raw)
        no_bids = _norm_levels(no_raw)

        yes_depth_usd = round(sum(x["notional_usd"] for x in yes_bids), 2)
        no_depth_usd = round(sum(x["notional_usd"] for x in no_bids), 2)
        yes_qty = sum(x["qty"] for x in yes_bids)
        no_qty = sum(x["qty"] for x in no_bids)
        # Contract-quantity imbalance removes price-asymmetry bias on low/high probability contracts
        imbalance = (yes_qty - no_qty) / max(yes_qty + no_qty, 1.0)

        best_yes_bid = yes_bids[0]["price"] if yes_bids else 0.0
        best_no_bid = no_bids[0]["price"] if no_bids else 0.0
        best_yes_ask = round(1.0 - best_no_bid, 4) if best_no_bid > 0 else 0.0
        spread_cents = round((best_yes_ask - best_yes_bid) * 100.0, 2) if (best_yes_bid > 0 and best_yes_ask > best_yes_bid) else 0.0

        return {
            "yes_bids": yes_bids,
            "no_bids": no_bids,
            "yes_depth_usd": yes_depth_usd,
            "no_depth_usd": no_depth_usd,
            "imbalance_ratio": round(max(-1.0, min(1.0, imbalance)), 4),
            "best_yes_bid": best_yes_bid,
            "best_yes_ask": best_yes_ask,
            "spread_cents": spread_cents,
        }

    def fetch_candlesticks(self, series_ticker: str, market_ticker: str, lookback_days: int = 14) -> List[Dict[str, Any]]:
        """
        Fetch OHLC implied probability candlesticks in cents (0-100) formatted
        directly for TradingView Lightweight Charts v4.
        """
        end_ts = int(time.time())
        start_ts = end_ts - max(1, lookback_days) * 86400
        interval = 60 if lookback_days <= 21 else 1440

        data = self._get_json(
            f"/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": interval},
            ttl=60,
        )
        candles_raw = (data or {}).get("candlesticks") or []
        bars: List[Dict[str, Any]] = []
        last_seen_close: Optional[float] = None
        seen_ts = set()

        for c in candles_raw:
            ts = int(c.get("end_period_ts") or 0)
            if ts <= 0 or ts in seen_ts:
                continue
            price_obj = c.get("price") or {}
            ask_obj = c.get("yes_ask") or {}

            close_v = _parse_float(price_obj.get("close_dollars"), -1.0)
            if close_v < 0:
                close_v = _parse_float(ask_obj.get("close_dollars"), -1.0)
            if close_v < 0:
                close_v = _parse_float(price_obj.get("previous_dollars"), -1.0)
            if close_v < 0:
                if last_seen_close is None:
                    continue
                close_v = last_seen_close

            open_v = _parse_float(price_obj.get("open_dollars"), -1.0)
            high_v = _parse_float(price_obj.get("high_dollars"), -1.0)
            low_v = _parse_float(price_obj.get("low_dollars"), -1.0)

            if open_v < 0:
                open_v = last_seen_close if last_seen_close is not None else close_v
            if high_v < 0:
                high_v = max(open_v, close_v)
            if low_v < 0:
                low_v = min(open_v, close_v)

            last_seen_close = close_v
            seen_ts.add(ts)
            bars.append({
                "time": ts,
                "open": round(open_v * 100.0, 2),
                "high": round(high_v * 100.0, 2),
                "low": round(low_v * 100.0, 2),
                "close": round(close_v * 100.0, 2),
                "volume": _parse_float(c.get("volume_fp"), 0.0),
            })

        bars.sort(key=lambda b: b["time"])
        return bars

    def get_macro_calendar(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Build chronological countdown timeline of major Central Bank & Macro
        catalysts directly from live Kalshi event schedules + Finnhub earnings.
        (Avoids Finnhub's paid /calendar/economic 403 endpoint completely.)
        """
        ckey = "macro_calendar_v8"
        now = time.time()
        if not force_refresh:
            with self._lock:
                hit = self._cache.get(ckey)
                if hit and (now - hit[0]) < 180:
                    return hit[1]

        macro_events = self.fetch_flagship_events(category="ALL", ws_eligible_only=False, force_refresh=force_refresh)
        items: List[Dict[str, Any]] = []
        seen_series = set()

        for ev in macro_events:
            st = ev["series_ticker"]
            if ev["ui_category"] not in ("CENTRAL_BANKS", "MACRO_DATA"):
                continue
            # Keep nearest event per flagship macro series to build a crisp countdown ribbon
            if st in seen_series:
                continue
            seen_series.add(st)
            lead: PredictionMarket = ev["lead_market"]
            dt = _parse_iso_dt(ev.get("close_time"))
            date_label = dt.strftime("%b %d") if dt else "Upcoming"
            items.append({
                "kind": "MACRO_EVENT",
                "event_ticker": ev["event_ticker"],
                "series_ticker": st,
                "title": ev["title"],
                "date_label": date_label,
                "days_away": ev["days_to_close"],
                "domain": ev["domain"],
                "lead_outcome": lead.outcome_label,
                "lead_prob_pct": round(lead.mid_prob * 100.0, 1),
                "volume": ev["total_volume"],
                "ws_eligible": ev["ws_predict_eligible"],
            })

        # Add top upcoming corporate earnings from Finnhub free tier
        try:
            from src.data.finnhub_client import finnhub_client
            if finnhub_client.is_configured():
                major_set = {"NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "CRM", "AVGO", "COST", "WMT", "JPM", "GS"}
                earn = finnhub_client.get_earnings_calendar(days_ahead=7)
                picked = [e for e in earn if str(e.get("symbol", "")).upper() in major_set] or earn[:3]
                for e in picked[:3]:
                    sym = str(e.get("symbol")).upper()
                    d_str = str(e.get("date") or "")
                    dt_e = _parse_iso_dt(d_str)
                    if dt_e and dt_e.tzinfo is None:
                        dt_e = dt_e.replace(tzinfo=timezone.utc)
                    days_d = round(max(0.0, (dt_e.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 86400.0), 1) if dt_e else 3.0
                    items.append({
                        "kind": "EARNINGS",
                        "event_ticker": None,
                        "series_ticker": sym,
                        "title": f"{sym} Quarterly Earnings (Est EPS {e.get('epsEstimate', '—')})",
                        "date_label": d_str,
                        "days_away": days_d,
                        "domain": "FINANCIALS",
                        "lead_outcome": f"Hour: {e.get('hour', 'AMC').upper()}",
                        "lead_prob_pct": None,
                        "volume": 0.0,
                        "ws_eligible": False,
                    })
        except Exception:
            pass

        items.sort(key=lambda x: float(x.get("days_away", 999)))
        with self._lock:
            self._cache[ckey] = (now, items)
        return items


prediction_feed = KalshiPredictionFeed()
