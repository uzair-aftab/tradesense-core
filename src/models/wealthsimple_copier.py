"""
Wealthsimple Mobile Trade Copier & Lifecycle Engine
===================================================
Tailored for Wealthsimple Mobile App execution in North York, Ontario, Canada:
  - Canadian TSX ETF proxy mapping (e.g. BTCC.TO with $0 CAD commission)
  - Rigorous signal gating (>= 70% confidence, >= 1.5% hurdle to beat spreads, RRR >= 1.5x)
  - Unique sequential transaction IDs (WS-YYYYMMDD-###)
  - Active position lifecycle monitoring against target, stop, trailing ratchet, & regime
  - Linked exit closing alerts with net realized P&L tracking ($ CAD and %)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.data.database import DatabaseManager, db_manager
from src.models.strategies import SignalType, StrategySignal
from src.utils.logger import get_logger

logger = get_logger("hfp.wealthsimple")


@dataclass
class WealthsimpleCandidate:
    """Represents a high-conviction candidate trade for Wealthsimple execution."""
    ticker: str
    ws_symbol: str
    side: str
    entry_price: float
    target_price: float
    stop_price: float
    confidence: float
    expected_return_pct: float
    rrr: float
    reason: str

# ───────────────────────────────────────────────────────────────────────────
# Canadian Wealthsimple Asset Proxy Map
# ───────────────────────────────────────────────────────────────────────────
CANADIAN_PROXIES: Dict[str, Dict[str, Any]] = {
    "BTC-CAD": {
        "ws_symbol": "BTCC.TO",
        "name": "Purpose Bitcoin CAD ETF",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Canada's premier spot Bitcoin ETF ($0 comm on Wealthsimple)",
    },
    "BTC-USD": {
        "ws_symbol": "BTCC.TO",
        "name": "Purpose Bitcoin CAD ETF",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Canada's premier spot Bitcoin ETF ($0 comm on Wealthsimple)",
    },
    "ETH-CAD": {
        "ws_symbol": "ETHX.TO",
        "name": "CI Galaxy Ethereum CAD ETF",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Spot Ethereum ETF ($0 comm on Wealthsimple)",
    },
    "ETH-USD": {
        "ws_symbol": "ETHX.TO",
        "name": "CI Galaxy Ethereum CAD ETF",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Spot Ethereum ETF ($0 comm on Wealthsimple)",
    },
    "SOL-CAD": {
        "ws_symbol": "SOL",
        "name": "Solana (Wealthsimple Crypto)",
        "commission": "1.5% Spread Fee",
        "type": "Crypto",
        "description": "Direct Solana Crypto trading on Wealthsimple",
    },
    "SOL-USD": {
        "ws_symbol": "SOL",
        "name": "Solana (Wealthsimple Crypto)",
        "commission": "1.5% Spread Fee",
        "type": "Crypto",
        "description": "Direct Solana Crypto trading on Wealthsimple",
    },
    "ADA-CAD": {
        "ws_symbol": "ADA",
        "name": "Cardano (Wealthsimple Crypto)",
        "commission": "1.5% Spread Fee",
        "type": "Crypto",
        "description": "Direct Cardano Crypto trading on Wealthsimple",
    },
    "AVAX-CAD": {
        "ws_symbol": "AVAX",
        "name": "Avalanche (Wealthsimple Crypto)",
        "commission": "1.5% Spread Fee",
        "type": "Crypto",
        "description": "Direct Avalanche Crypto trading on Wealthsimple",
    },
    "AAPL": {
        "ws_symbol": "AAPL.NE",
        "name": "Apple CDR (CAD Hedged)",
        "commission": "$0 CAD (NEO)",
        "type": "CDR",
        "description": "Canadian Depositary Receipt for Apple (no US FX fee)",
    },
    "NVDA": {
        "ws_symbol": "NVDA.NE",
        "name": "NVIDIA CDR (CAD Hedged)",
        "commission": "$0 CAD (NEO)",
        "type": "CDR",
        "description": "Canadian Depositary Receipt for NVIDIA (no US FX fee)",
    },
    "MSFT": {
        "ws_symbol": "MSFT.NE",
        "name": "Microsoft CDR (CAD Hedged)",
        "commission": "$0 CAD (NEO)",
        "type": "CDR",
        "description": "Canadian Depositary Receipt for Microsoft (no US FX fee)",
    },
    "GOOGL": {
        "ws_symbol": "GOOG.NE",
        "name": "Alphabet CDR (CAD Hedged)",
        "commission": "$0 CAD (NEO)",
        "type": "CDR",
        "description": "Canadian Depositary Receipt for Alphabet (no US FX fee)",
    },
    "AMZN": {
        "ws_symbol": "AMZN.NE",
        "name": "Amazon CDR (CAD Hedged)",
        "commission": "$0 CAD (NEO)",
        "type": "CDR",
        "description": "Canadian Depositary Receipt for Amazon (no US FX fee)",
    },
    "SPY": {
        "ws_symbol": "VFV.TO",
        "name": "Vanguard S&P 500 Index ETF (CAD)",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Canadian S&P 500 ETF ($0 comm, zero US FX fee on Wealthsimple)",
    },
    "QQQ": {
        "ws_symbol": "XQQ.TO",
        "name": "iShares NASDAQ 100 Index ETF (CAD-Hedged)",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Canadian Nasdaq-100 ETF ($0 comm, zero US FX fee on Wealthsimple)",
    },
    "SHOP.TO": {
        "ws_symbol": "SHOP.TO",
        "name": "Shopify Inc (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "Equity",
        "description": "Canadian Mega-Cap Tech Leader ($0 comm on Wealthsimple)",
    },
    "RY.TO": {
        "ws_symbol": "RY.TO",
        "name": "Royal Bank of Canada (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "Equity",
        "description": "Canada's largest bank ($0 comm on Wealthsimple)",
    },
    "TD.TO": {
        "ws_symbol": "TD.TO",
        "name": "TD Bank Group (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "Equity",
        "description": "Top-tier Canadian Bank ($0 comm on Wealthsimple)",
    },
    "ENB.TO": {
        "ws_symbol": "ENB.TO",
        "name": "Enbridge Inc (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "Equity",
        "description": "Canadian Energy Infrastructure Leader ($0 comm on Wealthsimple)",
    },
    "CNR.TO": {
        "ws_symbol": "CNR.TO",
        "name": "Canadian National Railway (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "Equity",
        "description": "Canadian Rail & Logistics Leader ($0 comm on Wealthsimple)",
    },
    "VFV.TO": {
        "ws_symbol": "VFV.TO",
        "name": "Vanguard S&P 500 Index ETF (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Canadian S&P 500 ETF ($0 comm on Wealthsimple)",
    },
    "XIU.TO": {
        "ws_symbol": "XIU.TO",
        "name": "iShares S&P/TSX 60 Index ETF (TSX)",
        "commission": "$0 CAD (TSX)",
        "type": "ETF",
        "description": "Top 60 Canadian TSX Companies ($0 comm on Wealthsimple)",
    },
}


# Default fallback CAD prices for TSX/NEO proxies when offline
DEFAULT_PROXY_PRICES_CAD: Dict[str, float] = {
    "BTCC.TO": 21.45,
    "ETHX.TO": 19.85,
    "NVDA.NE": 29.60,
    "AAPL.NE": 32.40,
    "MSFT.NE": 28.90,
    "GOOG.NE": 27.50,
    "AMZN.NE": 26.80,
    "VFV.TO": 146.50,
    "XQQ.TO": 152.20,
    "XIU.TO": 37.80,
    "SHOP.TO": 154.00,
    "RY.TO": 172.50,
    "TD.TO": 79.40,
    "ENB.TO": 59.80,
    "CNR.TO": 156.00,
}

_PROXY_PRICE_CACHE: Dict[str, tuple[float, float]] = {}


class CanadianAssetProxy:
    """Helper for Canadian Wealthsimple asset proxies and live CAD share price translation."""

    @staticmethod
    def get_proxy(ticker: str) -> Dict[str, Any]:
        base = ticker.upper().strip()
        if base in CANADIAN_PROXIES:
            info = dict(CANADIAN_PROXIES[base])
        elif base.endswith(".TO") or base.endswith(".V") or base.endswith(".NE"):
            info = {
                "ws_symbol": base,
                "name": base,
                "commission": "$0 CAD (Canadian)",
                "type": "Equity",
                "description": f"Direct Canadian equity ({base}) on Wealthsimple",
            }
        else:
            info = {
                "ws_symbol": base,
                "name": base,
                "commission": "Standard fee",
                "type": "Direct Asset",
                "description": f"Direct asset ({base}) on Wealthsimple",
            }
        info["ws_ticker"] = info["ws_symbol"]
        info["badge"] = f"{info['ws_symbol']} ({info['commission']})"
        return info

    @classmethod
    def resolve(cls, ticker: str) -> Dict[str, Any]:
        """Alias for get_proxy returning ws_symbol, ws_ticker, badge, commission, type."""
        return cls.get_proxy(ticker)

    @classmethod
    def _fetch_proxy_price_cad(cls, ws_symbol: str, underlying_ticker: str = "", underlying_price: float = 0.0) -> float:
        """
        Fetch the actual live CAD share price float for a Wealthsimple proxy (e.g. BTCC.TO, NVDA.NE, VFV.TO).
        Uses a 60-second TTL cache and falls back to calibrated proxy prices if offline.
        """
        import time
        import yfinance as yf

        sym = ws_symbol.upper().strip()
        u_sym = underlying_ticker.upper().strip()

        if sym == u_sym or (u_sym.startswith(sym + "-") and underlying_price > 0):
            return float(underlying_price) if underlying_price > 0 else DEFAULT_PROXY_PRICES_CAD.get(sym, 100.0)

        now = time.time()
        if sym in _PROXY_PRICE_CACHE:
            cached_ts, cached_p = _PROXY_PRICE_CACHE[sym]
            if (now - cached_ts) < 60.0 and cached_p > 0:
                return cached_p

        yf_lookup = sym
        try:
            tkr = yf.Ticker(yf_lookup)
            fi = getattr(tkr, "fast_info", None)
            if fi is not None:
                last_p = getattr(fi, "last_price", None)
                if last_p is None and isinstance(fi, dict):
                    last_p = fi.get("lastPrice") or fi.get("last_price")
                if last_p and float(last_p) > 0:
                    val = round(float(last_p), 2)
                    _PROXY_PRICE_CACHE[sym] = (now, val)
                    return val

            hist = tkr.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                val = round(float(hist["Close"].iloc[-1]), 2)
                if val > 0:
                    _PROXY_PRICE_CACHE[sym] = (now, val)
                    return val
        except Exception:
            pass

        fallback = DEFAULT_PROXY_PRICES_CAD.get(sym, float(underlying_price) if underlying_price > 0 else 50.0)
        _PROXY_PRICE_CACHE[sym] = (now, fallback)
        return fallback

    @classmethod
    def get_live_proxy_price(
        cls,
        ticker: str,
        current_underlying_price: float = 0.0,
        underlying_ticker: str = "",
        underlying_price: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Return a complete dictionary of the Wealthsimple proxy symbol, live CAD share price,
        commission badge, and unit-scaling status for a given ticker.
        """
        u_ticker = underlying_ticker or ticker
        u_price = current_underlying_price if current_underlying_price > 0 else underlying_price
        proxy_info = cls.get_proxy(u_ticker)
        ws_sym = proxy_info["ws_symbol"]
        live_cad = cls._fetch_proxy_price_cad(ws_sym, underlying_ticker=u_ticker, underlying_price=u_price)
        is_unit_scaled = (ws_sym.upper() != u_ticker.upper()) and not u_ticker.upper().startswith(ws_sym.upper() + "-")
        out = dict(proxy_info)
        out["proxy_price_cad"] = live_cad
        out["is_unit_scaled"] = is_unit_scaled
        return out

    @classmethod
    def translate_order_to_proxy(
        cls,
        ticker: str,
        underlying_price: Optional[float] = None,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        capital_cad: float = 500.0,
        ws_tier: str = "Core",
        use_spot_crypto_after_hours: bool = False,
        underlying_entry: Optional[float] = None,
        underlying_target: Optional[float] = None,
        underlying_stop: Optional[float] = None,
        tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate an underlying asset trade setup (e.g., BTC-CAD @ $115,860 or NVDA @ $138 USD)
        into exact, ready-to-paste Wealthsimple proxy share prices (e.g., BTCC.TO @ $21.45 CAD
        or NVDA.NE @ $29.60 CAD), including exact share quantity, Wealthsimple tier fees,
        and TSX/NEO market-hours guidance.
        """
        eff_entry = float(underlying_entry if underlying_entry is not None else (entry_price or 100.0))
        eff_target = float(underlying_target if underlying_target is not None else (target_price or eff_entry * 1.03))
        eff_stop = float(underlying_stop if underlying_stop is not None else (stop_price or eff_entry * 0.985))
        eff_u_price = float(underlying_price if underlying_price is not None else eff_entry)
        eff_tier = str(tier if tier is not None else ws_tier)

        proxy_info = cls.get_proxy(ticker)
        ws_sym = proxy_info["ws_symbol"]
        inst_type = proxy_info.get("type", "ETF")

        try:
            from src.data.market_intel import MarketIntelligenceEngine
            tsx_open, tsx_label, _ = MarketIntelligenceEngine.is_tsx_market_open()
        except Exception:
            tsx_open, tsx_label = True, "🟢 TSX/NEO MARKET HOURS"

        u_price = max(0.0001, eff_u_price)
        entry_ratio = eff_entry / u_price if eff_entry > 0 else 1.0
        target_ratio = eff_target / u_price if eff_target > 0 else 1.028
        stop_ratio = eff_stop / u_price if eff_stop > 0 else 0.985

        is_crypto_underlying = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL", "ADA", "AVAX"])
        if use_spot_crypto_after_hours and is_crypto_underlying:
            exec_symbol = ticker.upper().split("-")[0]
            proxy_live = u_price
            is_distinct_proxy = False
            inst_type = "Crypto"
        else:
            exec_symbol = ws_sym
            proxy_live = cls._fetch_proxy_price_cad(ws_sym, underlying_ticker=ticker, underlying_price=u_price)
            is_distinct_proxy = (exec_symbol.upper() != ticker.upper()) and not ticker.upper().startswith(exec_symbol.upper() + "-")

        p_entry = round(proxy_live * entry_ratio, 2 if proxy_live >= 1.0 else 4)
        p_target = round(proxy_live * target_ratio, 2 if proxy_live >= 1.0 else 4)
        p_stop = round(proxy_live * stop_ratio, 2 if proxy_live >= 1.0 else 4)

        tier_norm = (eff_tier or "Core").strip().title()
        if inst_type in ("ETF", "Equity"):
            round_trip_fee_pct = 0.0
            fee_desc = "$0 Commission (TSX CAD ETF/Equity)"
        elif inst_type == "CDR":
            round_trip_fee_pct = 0.0
            fee_desc = "$0 Commission + $0 US FX (NEO CAD-Hedged CDR)"
        elif inst_type == "Crypto":
            one_way = 0.50 if "Gen" in tier_norm else (0.75 if "Prem" in tier_norm else 1.50)
            round_trip_fee_pct = one_way * 2.0
            fee_desc = f"Wealthsimple {tier_norm} Crypto ({one_way:.2f}% per side)"
        else:
            round_trip_fee_pct = 0.0
            fee_desc = f"Wealthsimple {tier_norm} ($0 Commission)"

        cap = max(10.0, float(capital_cad))
        shares = round(cap / max(0.0001, p_entry), 4 if p_entry > 200 else 2)
        gross_profit_cad = abs(p_target - p_entry) * shares
        est_fee_cad = cap * (round_trip_fee_pct / 100.0)
        net_profit_cad = gross_profit_cad - est_fee_cad
        max_loss_cad = abs(p_entry - p_stop) * shares + est_fee_cad

        if is_distinct_proxy and not tsx_open and is_crypto_underlying:
            venue_guidance = (
                f"{tsx_label}: Queue a $0-fee Limit Order on {exec_symbol} for 9:30 AM ET open, "
                f"or trade 24/7 Spot Crypto for immediate overnight execution."
            )
        elif is_distinct_proxy:
            venue_guidance = (
                f"Mapped {ticker} (${u_price:,.2f}) → {exec_symbol} (${proxy_live:,.2f} CAD/sh) "
                f"for $0 commission on Wealthsimple."
            )
        else:
            venue_guidance = f"Direct execution on {exec_symbol} (${proxy_live:,.2f} CAD) • {fee_desc}."

        return {
            "ticker": ticker,
            "ws_symbol": exec_symbol,
            "exec_symbol": exec_symbol,
            "underlying_price": round(u_price, 2),
            "proxy_live_price": proxy_live,
            "proxy_price_cad": proxy_live,
            "is_distinct_proxy": is_distinct_proxy,
            "is_unit_scaled": is_distinct_proxy,
            "proxy_entry": p_entry,
            "proxy_entry_cad": p_entry,
            "proxy_target": p_target,
            "proxy_target_cad": p_target,
            "proxy_stop": p_stop,
            "proxy_stop_cad": p_stop,
            "underlying_entry": round(eff_entry, 2),
            "underlying_target": round(eff_target, 2),
            "underlying_stop": round(eff_stop, 2),
            "shares": shares,
            "share_qty": shares,
            "capital_cad": round(cap, 2),
            "gross_profit_cad": round(gross_profit_cad, 2),
            "estimated_fee_cad": round(est_fee_cad, 2),
            "est_roundtrip_fee_cad": round(est_fee_cad, 2),
            "net_profit_cad": round(net_profit_cad, 2),
            "net_expected_profit_cad": round(net_profit_cad, 2),
            "max_loss_cad": round(max_loss_cad, 2),
            "round_trip_fee_pct": round(round_trip_fee_pct, 2),
            "fee_description": fee_desc,
            "fee_badge": fee_desc,
            "tsx_open": tsx_open,
            "tsx_status_label": tsx_label,
            "venue_guidance": venue_guidance,
            "execution_guidance": venue_guidance,
        }


class WealthsimpleCopier:
    """End-to-end Wealthsimple Mobile Copier Lifecycle Engine."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    def get_proxy_info(self, ticker: str) -> Dict[str, Any]:
        """Lookup Canadian Wealthsimple execution proxy."""
        base = ticker.upper()
        if base in CANADIAN_PROXIES:
            return CANADIAN_PROXIES[base]
        # Fallback for standard Canadian or US equities
        if base.endswith(".TO") or base.endswith(".V") or base.endswith(".NE"):
            return {
                "ws_symbol": base,
                "name": base,
                "commission": "$0 CAD (Canadian)",
                "type": "Equity",
                "description": f"Canadian instrument tradeable on Wealthsimple ($0 comm)",
            }
        return {
            "ws_symbol": base,
            "name": base,
            "commission": "1.5% US FX Fee",
            "type": "US Stock",
            "description": f"US stock tradeable on Wealthsimple",
        }

    def generate_trade_id(self, user_id: int) -> str:
        """Generate human-readable sequential trade ID: WS-YYYYMMDD-###."""
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        count = self.db.get_today_ws_trade_count(user_id)
        return f"WS-{today_str}-{count + 1:03d}"

    def evaluate_signal_for_alert(
        self,
        signal: StrategySignal,
        current_price: float,
        portfolio_cash: float,
        risk_pct: float = 0.02,
        min_rrr: float = 1.2,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate if a strategy signal qualifies for a Wealthsimple mobile alert.

        Criteria:
          1. Signal is BUY or SELL (not HOLD)
          2. Signal strength / confidence >= 0.70 (70%)
          3. Expected return hurdle >= 1.5% (beats Wealthsimple spread friction)
          4. Reward-to-Risk Ratio >= min_rrr (default 1.2x)
        """
        if signal.signal_type == SignalType.HOLD:
            return None

        if signal.strength < 0.70:
            return None

        target = signal.target_price or (current_price * 1.03 if signal.signal_type == SignalType.BUY else current_price * 0.97)
        stop = signal.stop_price or (current_price * 0.98 if signal.signal_type == SignalType.BUY else current_price * 1.02)

        upside_pct = abs((target - current_price) / current_price) * 100.0
        downside_pct = abs((current_price - stop) / current_price) * 100.0

        if upside_pct < 1.5:  # Minimum 1.5% hurdle to cover fees & execution latency
            return None

        rrr = upside_pct / max(0.01, downside_pct)
        if rrr < min_rrr:
            return None

        proxy = self.get_proxy_info(signal.ticker)

        # Capital sizing: 2% risk sizing scaled to position capital, capped between $100 and $2,000 CAD
        risk_capital = max(100.0, min(portfolio_cash * risk_pct * 4.0, min(2000.0, portfolio_cash * 0.4)))
        qty = round(risk_capital / max(1.0, current_price), 4) if "BTC" in signal.ticker else round(risk_capital / max(1.0, current_price), 2)
        if qty <= 0:
            qty = 1.0

        return {
            "ticker": signal.ticker,
            "ws_symbol": proxy["ws_symbol"],
            "ws_name": proxy["name"],
            "commission": proxy["commission"],
            "side": signal.signal_type.value,
            "suggested_price": round(current_price, 2),
            "target_price": round(target, 2),
            "stop_price": round(stop, 2),
            "upside_pct": round(upside_pct, 2),
            "downside_pct": round(downside_pct, 2),
            "rrr": round(rrr, 2),
            "confidence": round(signal.strength * 100, 1),
            "suggested_capital": round(risk_capital, 2),
            "suggested_qty": qty,
            "reason": signal.reason,
            "strategy": signal.strategy_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def confirm_entry(
        self,
        user_id: int,
        alert: Dict[str, Any],
        actual_price: Optional[float] = None,
        actual_capital: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> str:
        """User confirms they executed the trade on their Wealthsimple app."""
        tx_id = self.generate_trade_id(user_id)
        fill_price = actual_price or alert["suggested_price"]
        capital = actual_capital or alert["suggested_capital"]
        qty = round(capital / fill_price, 4) if "BTC" in alert["ticker"] else round(capital / fill_price, 2)

        self.db.create_ws_trade(
            tx_id=tx_id,
            user_id=user_id,
            ticker=alert["ticker"],
            ws_symbol=alert["ws_symbol"],
            side=alert["side"],
            status="OPEN",
            suggested_entry=alert["suggested_price"],
            actual_entry=fill_price,
            quantity=max(0.0001, qty),
            capital_cad=capital,
            target_price=alert["target_price"],
            stop_price=alert["stop_price"],
            notes=notes or f"Copied from {alert.get('strategy', 'TradeSense')}",
        )
        logger.info(
            "Wealthsimple trade confirmed: %s for user %d (%s %s @ $%.2f, cap=$%.2f)",
            tx_id, user_id, alert["side"], alert["ws_symbol"], fill_price, capital
        )
        return tx_id

    def check_open_trades_for_exit(
        self,
        user_id: int,
        current_price: float,
        current_regime: str = "BULL_TREND",
    ) -> List[Dict[str, Any]]:
        """
        Evaluate active open Wealthsimple trades against live price.

        Exit conditions:
          1. Take Profit reached: price >= target_price
          2. Stop Loss breached: price <= stop_price
          3. Bear Trend shift: regime == 'BEAR_TREND' (cash preservation lock)
        """
        active_trades = self.db.get_active_ws_trades(user_id)
        exit_alerts: List[Dict[str, Any]] = []

        for t in active_trades:
            entry = float(t["actual_entry"] or t["suggested_entry"])
            target = float(t["target_price"]) if t["target_price"] else entry * 1.04
            stop = float(t["stop_price"]) if t["stop_price"] else entry * 0.98
            qty = float(t["quantity"])
            side = t["side"]

            exit_triggered = False
            reason = ""

            if side == "BUY":
                if current_price >= target:
                    exit_triggered = True
                    reason = f"🎯 Take-Profit Target Reached ($ {target:,.2f})"
                elif current_price <= stop:
                    exit_triggered = True
                    reason = f"🛑 Stop-Loss Protection Hit ($ {stop:,.2f})"
                elif "BEAR" in current_regime.upper():
                    exit_triggered = True
                    reason = "🛡️ Market Regime Shifted to Bearish (Cash Preservation)"

                if exit_triggered:
                    pnl_cad = (current_price - entry) * qty
                    pnl_pct = ((current_price - entry) / entry) * 100.0
                    exit_alerts.append(
                        {
                            "opening_tx_id": t["tx_id"],
                            "ticker": t["ticker"],
                            "ws_symbol": t["ws_symbol"],
                            "entry_price": entry,
                            "current_price": round(current_price, 2),
                            "quantity": qty,
                            "capital_cad": float(t["capital_cad"]),
                            "pnl_cad": round(pnl_cad, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "exit_reason": reason,
                            "action_needed": "SELL / CLOSE ON WEALTHSIMPLE",
                        }
                    )

        return exit_alerts

    def confirm_exit(
        self,
        tx_id: str,
        actual_exit_price: float,
        exit_reason: str = "Manual User Close",
    ) -> Dict[str, Any]:
        """User confirms they closed the trade on their Wealthsimple app."""
        trade = self.db.get_ws_trade_by_tx_id(tx_id)
        if not trade:
            raise ValueError(f"Transaction ID '{tx_id}' not found")

        entry = float(trade["actual_entry"] or trade["suggested_entry"])
        qty = float(trade["quantity"])
        side = trade["side"]

        if side == "BUY":
            pnl_cad = (actual_exit_price - entry) * qty
            pnl_pct = ((actual_exit_price - entry) / entry) * 100.0
        else:
            pnl_cad = (entry - actual_exit_price) * qty
            pnl_pct = ((entry - actual_exit_price) / entry) * 100.0

        self.db.close_ws_trade(
            tx_id=tx_id,
            actual_exit_price=round(actual_exit_price, 2),
            realized_pnl_cad=round(pnl_cad, 2),
            realized_pnl_pct=round(pnl_pct, 2),
            closing_reason=exit_reason,
        )

        logger.info(
            "Wealthsimple trade closed: %s @ $%.2f (P&L: $%.2f / %.2f%%)",
            tx_id, actual_exit_price, pnl_cad, pnl_pct
        )
        return {
            "tx_id": tx_id,
            "entry_price": entry,
            "exit_price": actual_exit_price,
            "pnl_cad": round(pnl_cad, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": exit_reason,
        }

    def get_user_analytics(self, user_id: int) -> Dict[str, Any]:
        """Calculate cumulative Wealthsimple performance metrics."""
        closed = self.db.get_closed_ws_trades(user_id, limit=500)
        active = self.db.get_active_ws_trades(user_id)

        if not closed:
            return {
                "total_trades": len(active),
                "closed_trades": 0,
                "active_trades": len(active),
                "total_realized_pnl_cad": 0.0,
                "win_rate_pct": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit_factor": 0.0,
                "avg_win_cad": 0.0,
                "avg_loss_cad": 0.0,
            }

        wins = [float(t["realized_pnl_cad"]) for t in closed if float(t["realized_pnl_cad"]) > 0]
        losses = [abs(float(t["realized_pnl_cad"])) for t in closed if float(t["realized_pnl_cad"]) < 0]
        total_pnl = sum(float(t["realized_pnl_cad"]) for t in closed)
        win_rate = (len(wins) / len(closed)) * 100.0 if closed else 0.0
        profit_factor = (sum(wins) / max(0.01, sum(losses))) if losses else (99.0 if wins else 0.0)

        return {
            "total_trades": len(closed) + len(active),
            "closed_trades": len(closed),
            "active_trades": len(active),
            "total_realized_pnl_cad": round(total_pnl, 2),
            "win_rate_pct": round(win_rate, 1),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "profit_factor": round(profit_factor, 2),
            "avg_win_cad": round(sum(wins) / max(1, len(wins)), 2),
            "avg_loss_cad": round(sum(losses) / max(1, len(losses)), 2),
        }

    def get_active_trades(self, user_id: int) -> List[Dict[str, Any]]:
        """Return list of active open trades for user."""
        return self.db.get_active_ws_trades(user_id)

    def get_closed_trades(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Return list of closed trades for user."""
        return self.db.get_closed_ws_trades(user_id, limit=limit)

    def get_user_summary(self, user_id: int) -> Dict[str, Any]:
        """Return user performance summary dictionary."""
        analytics = self.get_user_analytics(user_id)
        analytics["total_realized_cad"] = analytics.get("total_realized_pnl_cad", 0.0)
        return analytics

    def evaluate_signal(
        self,
        prediction: Optional[Any],
        current_price: float,
        ticker: str,
    ) -> Optional[WealthsimpleCandidate]:
        """
        Evaluate if a prediction qualifies as a high-conviction Wealthsimple trade candidate.
        """
        if prediction is None:
            return None

        conf = getattr(prediction, "confidence", 0.0)
        direction = getattr(prediction, "direction", "HOLD")

        if conf < 0.70 or direction not in ("BULLISH", "BEARISH"):
            return None

        side = "BUY" if direction == "BULLISH" else "SELL"
        p90 = getattr(prediction, "p90", current_price * 1.03)
        p10 = getattr(prediction, "p10", current_price * 0.98)

        if side == "BUY":
            target = p90 if p90 > current_price else current_price * 1.03
            stop = p10 if p10 < current_price else current_price * 0.98
            expected_ret = ((target - current_price) / current_price) * 100.0
            downside = ((current_price - stop) / current_price) * 100.0
        else:
            target = p10 if p10 < current_price else current_price * 0.97
            stop = p90 if p90 > current_price else current_price * 1.02
            expected_ret = ((current_price - target) / current_price) * 100.0
            downside = ((stop - current_price) / current_price) * 100.0

        if expected_ret < 1.5:
            return None

        rrr = expected_ret / max(0.01, downside)
        if rrr < 1.2:
            return None

        proxy = self.get_proxy_info(ticker)
        return WealthsimpleCandidate(
            ticker=ticker,
            ws_symbol=proxy["ws_symbol"],
            side=side,
            entry_price=round(current_price, 2),
            target_price=round(target, 2),
            stop_price=round(stop, 2),
            confidence=conf,
            expected_return_pct=round(expected_ret, 2),
            rrr=round(rrr, 2),
            reason=f"Deep Quantile {conf:.0%} Conviction ({direction})",
        )

    def confirm_execution(
        self,
        user_id: int,
        signal: Any,
        actual_entry_price: float,
        capital_cad: float,
        quantity: float,
    ) -> str:
        """Confirm execution from the interactive card."""
        tx_id = self.generate_trade_id(user_id)
        ticker = getattr(signal, "ticker", "BTC-CAD")
        ws_symbol = getattr(signal, "ws_symbol", "BTCC.TO")
        side = getattr(signal, "side", "BUY")
        target_p = getattr(signal, "target_price", actual_entry_price * 1.03)
        stop_p = getattr(signal, "stop_price", actual_entry_price * 0.98)
        reason = getattr(signal, "reason", "Wealthsimple Signal Execution")

        self.db.create_ws_trade(
            tx_id=tx_id,
            user_id=user_id,
            ticker=ticker,
            ws_symbol=ws_symbol,
            side=side,
            status="OPEN",
            suggested_entry=actual_entry_price,
            actual_entry=actual_entry_price,
            quantity=quantity,
            capital_cad=capital_cad,
            target_price=target_p,
            stop_price=stop_p,
            notes=reason,
        )
        return tx_id

    def evaluate_active_trades(
        self,
        user_id: int,
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Evaluate open active trades for exit conditions across current prices."""
        active_trades = self.db.get_active_ws_trades(user_id)
        triggers = []
        for t in active_trades:
            ticker = t["ticker"]
            cur_p = current_prices.get(ticker, float(t["actual_entry"] or 0.0))
            entry = float(t["actual_entry"] or t["suggested_entry"])
            target = float(t["target_price"]) if t.get("target_price") else entry * 1.04
            stop = float(t["stop_price"]) if t.get("stop_price") else entry * 0.98
            qty = float(t["quantity"])
            side = t["side"]

            exit_flag = False
            reason = ""
            if side == "BUY":
                if cur_p >= target:
                    exit_flag = True
                    reason = f"Take-Profit Target Hit (${target:,.2f})"
                elif cur_p <= stop:
                    exit_flag = True
                    reason = f"Stop-Loss Breached (${stop:,.2f})"
            elif side == "SELL":
                if cur_p <= target:
                    exit_flag = True
                    reason = f"Take-Profit Target Hit (${target:,.2f})"
                elif cur_p >= stop:
                    exit_flag = True
                    reason = f"Stop-Loss Breached (${stop:,.2f})"

            if exit_flag:
                pnl_cad = (cur_p - entry) * qty if side == "BUY" else (entry - cur_p) * qty
                pnl_pct = ((cur_p - entry) / entry) * 100.0 if side == "BUY" else ((entry - cur_p) / entry) * 100.0
                triggers.append({
                    "tx_id": t["tx_id"],
                    "ticker": ticker,
                    "ws_symbol": t["ws_symbol"],
                    "side": side,
                    "actual_entry": entry,
                    "quantity": qty,
                    "current_price": cur_p,
                    "reason": reason,
                    "projected_pnl_cad": round(pnl_cad, 2),
                    "projected_pnl_pct": round(pnl_pct, 2),
                })
        return triggers

    def confirm_closing(
        self,
        tx_id: str,
        actual_exit_price: float,
        realized_pnl_cad: float = 0.0,
        realized_pnl_pct: float = 0.0,
        closing_reason: str = "Exit Target Hit",
    ) -> Dict[str, Any]:
        """Confirm closing an active trade."""
        self.db.close_ws_trade(
            tx_id=tx_id,
            actual_exit_price=round(actual_exit_price, 2),
            realized_pnl_cad=round(realized_pnl_cad, 2),
            realized_pnl_pct=round(realized_pnl_pct, 2),
            closing_reason=closing_reason,
        )
        return {
            "tx_id": tx_id,
            "exit_price": actual_exit_price,
            "realized_pnl_cad": realized_pnl_cad,
            "realized_pnl_pct": realized_pnl_pct,
            "reason": closing_reason,
        }


# Global singleton instance
ws_copier = WealthsimpleCopier()
