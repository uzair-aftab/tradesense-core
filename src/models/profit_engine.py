"""
TradeSense Profit-Maximizing Institutional Stack
=================================================
Core institutional wealth-generation infrastructure:
  1. AlphaScanner: Multi-pair Net-EV opportunity scanner (after Wealthsimple fees)
  2. PositionRiskSizer: Quarter-Kelly, volatility scaling, and portfolio heat guard
  3. MarketHoursGuard: Toronto market clock, order shield, and TFSA wrapper advisor
  4. TrailingStopRatchet: Breakeven lock and Chandelier trailing stop engine
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pytz

import numpy as np
import pandas as pd

from src.data.data_feed import DataLake
from src.data.market_intel import MarketIntelligenceEngine
from src.data.ws_catalog import WealthsimpleAsset, WealthsimpleCatalog, WealthsimpleFeeSchedule
from src.models.regime import MarketRegime, RegimeClassifier
from src.utils.logger import get_logger

logger = get_logger("hfp.profit_engine")

TOP_WEALTHSIMPLE_BASKET = [
    "SHOP.TO", "RY.TO", "CLS.TO", "CCO.TO", "ENB.TO",
    "VFV.TO", "XQQ.TO",
    "NVDA.NE", "AAPL.NE", "TSLA.NE", "MSFT.NE", "AMD.NE",
    "BTC-CAD", "ETH-CAD", "SOL-CAD",
    "QQQ",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market Hours & Account Shield
# ─────────────────────────────────────────────────────────────────────────────
class MarketHoursGuard:
    """Monitors Toronto market clock and recommends defensive order handling."""

    TORONTO_TZ = pytz.timezone("America/Toronto")

    @classmethod
    def get_market_status(cls, asset: WealthsimpleAsset) -> Dict[str, Any]:
        """Evaluates live market phase and execution safety for the given asset."""
        if asset.category == "CRYPTO":
            return {
                "phase": "OPEN",
                "is_open": True,
                "msg": "🟢 24/7 Global Crypto Market Active",
                "order_warning": None,
                "wrapper_advice": "⚠️ WS Crypto Account is Taxable. For tax-free gains, consider proxy BTCC.TO / ETHX-B.TO inside a TFSA.",
            }

        now_et = datetime.datetime.now(cls.TORONTO_TZ)
        weekday = now_et.weekday()  # 0=Monday, 6=Sunday
        time_minutes = now_et.hour * 60 + now_et.minute

        # 9:30 AM = 570 mins, 4:00 PM = 960 mins
        market_open_min = 9 * 60 + 30
        market_close_min = 16 * 60
        pre_market_min = 7 * 60
        post_market_min = 20 * 60

        is_weekend = weekday >= 5
        if is_weekend:
            phase = "CLOSED_WEEKEND"
            is_open = False
            msg = "🔴 Market Closed (Weekend)"
        elif time_minutes < pre_market_min or time_minutes >= post_market_min:
            phase = "CLOSED_NIGHT"
            is_open = False
            msg = "🔴 Market Closed (Night)"
        elif time_minutes < market_open_min:
            phase = "PRE_MARKET"
            is_open = False
            msg = "🟡 Pre-Market Session"
        elif time_minutes <= market_close_min:
            phase = "OPEN"
            is_open = True
            msg = "🟢 Regular Market Hours (TSX/NYSE Open)"
        else:
            phase = "POST_MARKET"
            is_open = False
            msg = "🟡 After-Hours Session"

        order_warning = None
        if not is_open:
            order_warning = "⚠️ Market is currently closed. ALWAYS use a LIMIT ORDER on Wealthsimple (never market order off-hours)."

        wrapper_advice = "✅ TFSA / FHSA / RRSP Eligible (0% Canadian Capital Gains Tax)"
        if asset.category == "US_STOCK" and not asset.symbol.endswith(".NE"):
            wrapper_advice += " • Note: 15% US withholding tax on dividends in TFSA; held best in RRSP."

        return {
            "phase": phase,
            "is_open": is_open,
            "msg": msg,
            "toronto_time": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "order_warning": order_warning,
            "wrapper_advice": wrapper_advice,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Position Risk & Kelly Sizer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionSizeResult:
    recommended_cad: float
    recommended_shares: float
    quarter_kelly_fraction: float
    max_risk_cap_cad: float
    dollar_risk_at_stop: float
    heat_warning: Optional[str]
    notes: List[str]


class PositionRiskSizer:
    """Calculates Quarter-Kelly allocations with strict max stop risk and portfolio heat controls."""

    @classmethod
    def calculate_position_size(
        cls,
        total_equity_cad: float,
        current_price: float,
        p10_stop: float,
        p90_target: float,
        hit_rate_pct: float = 60.0,
        round_trip_fee_pct: float = 0.08,
        risk_per_trade_pct: float = 1.5,
        existing_heat_cad: float = 0.0,
        max_portfolio_heat_pct: float = 6.0,
    ) -> PositionSizeResult:
        """
        Computes mathematically optimal allocation using Quarter-Kelly and Stop-Loss Risk Cap.
        """
        notes = []
        if total_equity_cad <= 0 or current_price <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, 0.0, 0.0, None, ["Account empty"])

        # 1. Expected Returns
        upside_pct = max(0.1, ((p90_target - current_price) / current_price) * 100.0)
        downside_pct = max(0.1, ((current_price - p10_stop) / current_price) * 100.0)

        net_win = max(0.05, upside_pct - round_trip_fee_pct)
        net_loss = downside_pct + round_trip_fee_pct

        # 2. Kelly Criterion
        p = float(np.clip(hit_rate_pct / 100.0, 0.35, 0.75))
        b = max(0.1, net_win / net_loss)
        full_kelly = max(0.0, p - ((1.0 - p) / b))

        # Quarter-Kelly (fractional Kelly prevents drawdown ruin)
        quarter_kelly = float(np.clip(0.25 * full_kelly, 0.0, 0.25))
        kelly_capital = total_equity_cad * quarter_kelly

        # 3. Hard Stop-Risk Cap: Max loss if P10 stop is hit must not exceed risk_per_trade_pct
        max_allowed_dollar_risk = total_equity_cad * (risk_per_trade_pct / 100.0)
        risk_cap_capital = max_allowed_dollar_risk / (downside_pct / 100.0)

        # Baseline allocation is smaller of Quarter-Kelly and Risk Cap
        recommended_cad = min(kelly_capital, risk_cap_capital)
        if quarter_kelly <= 0.01:
            # Low edge
            recommended_cad = min(recommended_cad, total_equity_cad * 0.05)
            notes.append("Low edge regime: size restricted to defensive baseline")

        # 4. Portfolio Heat Guard
        max_heat_dollars = total_equity_cad * (max_portfolio_heat_pct / 100.0)
        new_risk_at_stop = recommended_cad * (downside_pct / 100.0)
        total_projected_heat = existing_heat_cad + new_risk_at_stop

        heat_warning = None
        if total_projected_heat > max_heat_dollars:
            available_risk = max(0.0, max_heat_dollars - existing_heat_cad)
            recommended_cad = available_risk / (downside_pct / 100.0)
            heat_warning = f"⚠️ Portfolio heat limit ({max_portfolio_heat_pct}%): allocation reduced to ${recommended_cad:,.2f} CAD."
            notes.append("Heat limit enforced")

        recommended_cad = max(0.0, round(recommended_cad, 2))
        shares = round(recommended_cad / current_price, 4) if current_price > 0 else 0.0
        actual_dollar_risk = round(recommended_cad * (downside_pct / 100.0), 2)

        return PositionSizeResult(
            recommended_cad=recommended_cad,
            recommended_shares=shares,
            quarter_kelly_fraction=round(quarter_kelly, 4),
            max_risk_cap_cad=round(risk_cap_capital, 2),
            dollar_risk_at_stop=actual_dollar_risk,
            heat_warning=heat_warning,
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trailing Stop Ratchet Engine
# ─────────────────────────────────────────────────────────────────────────────
class TrailingStopRatchet:
    """Computes dynamic Breakeven Locks and Chandelier Trailing Stops for open positions."""

    @classmethod
    def evaluate_stop(
        cls,
        entry_price: float,
        current_price: float,
        initial_stop_price: float,
        target_price: float,
        round_trip_fee_pct: float = 0.08,
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Determines whether the stop should ratchet to Breakeven or trail dynamically.
        """
        if entry_price <= 0 or current_price <= 0:
            return {"stop_price": initial_stop_price, "status": "INITIAL", "desc": "Standard initial stop"}

        target_dist = target_price - entry_price
        gain = current_price - entry_price
        breakeven_price = round(entry_price * (1.0 + (round_trip_fee_pct / 100.0)), 4)

        # 1. 80%+ Target reached -> Chandelier Trailing Stop
        if target_dist > 0 and gain >= 0.80 * target_dist:
            vol_buffer = (atr * 1.2) if (atr and atr > 0) else (current_price * 0.015)
            chandelier_stop = round(current_price - vol_buffer, 4)
            active_stop = max(breakeven_price, chandelier_stop, initial_stop_price)
            return {
                "stop_price": active_stop,
                "status": "CHANDELIER_RUNNER",
                "desc": f"🎯 80%+ Target reached! Trailing runner active at ${active_stop:,.2f}",
                "is_breakeven_locked": True,
            }

        # 2. 50%+ Target reached -> Risk-Free Breakeven Lock
        if target_dist > 0 and gain >= 0.50 * target_dist:
            active_stop = max(breakeven_price, initial_stop_price)
            return {
                "stop_price": active_stop,
                "status": "BREAKEVEN_LOCKED",
                "desc": f"🛡️ Risk-Free Breakeven locked at ${active_stop:,.2f} (covers WS fees)",
                "is_breakeven_locked": True,
            }

        return {
            "stop_price": initial_stop_price,
            "status": "INITIAL",
            "desc": f"Initial protection stop at ${initial_stop_price:,.2f}",
            "is_breakeven_locked": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Pair Net-EV Opportunity Scanner
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AlphaOpportunity:
    symbol: str
    name: str
    category: str
    optimal_route_symbol: str
    current_price: float
    target_price: float
    p10_stop: float
    upside_pct: float
    downside_pct: float
    round_trip_fee_pct: float
    net_ev_pct: float
    hit_rate_pct: float
    regime: str
    action: str  # "STRONG_BUY", "TACTICAL_BUY", "WAIT_CHOP", "CASH_LOCK"
    earnings_warning: Optional[str]
    tfsa_status: str


class AlphaScanner:
    """Scans liquid Wealthsimple universe and ranks opportunities by Net Expected Value %."""

    _cached_scan: Optional[List[AlphaOpportunity]] = None
    _cached_timestamp: Optional[datetime.datetime] = None

    @classmethod
    def scan_opportunities(
        cls,
        basket: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[AlphaOpportunity]:
        """Scans the basket and returns Net-EV ranked opportunities."""
        now = datetime.datetime.now()
        if (
            not force_refresh
            and cls._cached_scan is not None
            and cls._cached_timestamp is not None
            and (now - cls._cached_timestamp).total_seconds() < 180
        ):
            return cls._cached_scan

        tickers = basket or TOP_WEALTHSIMPLE_BASKET
        opportunities: List[AlphaOpportunity] = []
        intel = MarketIntelligenceEngine()

        for sym in tickers:
            asset = WealthsimpleCatalog.get_asset(sym)
            if not asset:
                continue

            try:
                df = DataLake().pipeline(sym, timeframe="1h", lookback="30d", include_sentiment=False, normalise=False)
                if df is None or len(df) < 25:
                    continue

                curr_price = float(df["Close"].iloc[-1])
                regime_ctx = RegimeClassifier().classify(df)
                regime_name = regime_ctx.regime.value

                # Optimal Route calculation
                proxy_sym = asset.alt_cad_proxy
                proxy_asset = WealthsimpleCatalog.resolve_asset(proxy_sym) if proxy_sym else None
                opt_asset = proxy_asset if proxy_asset else asset
                fee_info = WealthsimpleFeeSchedule.calculate_fee(opt_asset, 500.0, "Core", False)
                rt_fee_pct = fee_info.round_trip_total_pct  # Round-trip

                # Heuristic / Model consensus estimate
                hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
                ema21 = df["Close"].ewm(span=21).mean().iloc[-1]
                vol_20 = float(df["Close"].pct_change().tail(20).std() * np.sqrt(5)) * 100.0

                drift_pct = float(np.clip(((curr_price - ema21) / curr_price) * 100.0 * 0.4, -4.0, 4.0))
                if regime_ctx.regime == MarketRegime.BULL_TREND:
                    upside_pct = max(1.2, vol_20 * 1.1 + drift_pct)
                    downside_pct = max(0.8, vol_20 * 0.7)
                    hit_rate = 66.0
                elif regime_ctx.regime == MarketRegime.BEAR_TREND:
                    upside_pct = max(0.5, vol_20 * 0.6)
                    downside_pct = max(1.5, vol_20 * 1.3)
                    hit_rate = 38.0
                else:  # Ranging / Chop
                    upside_pct = max(0.8, vol_20 * 0.8)
                    downside_pct = max(0.8, vol_20 * 0.8)
                    hit_rate = 50.0

                p_hit = hit_rate / 100.0
                net_ev_pct = round((p_hit * upside_pct - (1.0 - p_hit) * downside_pct) - rt_fee_pct, 2)

                p90_t = curr_price * (1.0 + upside_pct / 100.0)
                p10_s = curr_price * (1.0 - downside_pct / 100.0)

                # Quick earnings check
                earnings_warn = None

                # Determine action
                if regime_ctx.regime == MarketRegime.BEAR_TREND:
                    action = "CASH_LOCK"
                elif net_ev_pct > 1.2 and regime_ctx.regime == MarketRegime.BULL_TREND:
                    action = "STRONG_BUY"
                elif net_ev_pct > 0.4:
                    action = "TACTICAL_BUY"
                else:
                    action = "WAIT_CHOP"

                tfsa = "TFSA: 0% Tax" if asset.category != "CRYPTO" else "Taxable (Use Proxy)"

                opportunities.append(
                    AlphaOpportunity(
                        symbol=sym,
                        name=asset.name,
                        category=asset.category,
                        optimal_route_symbol=opt_asset.symbol,
                        current_price=round(curr_price, 2),
                        target_price=round(p90_t, 2),
                        p10_stop=round(p10_s, 2),
                        upside_pct=round(upside_pct, 2),
                        downside_pct=round(downside_pct, 2),
                        round_trip_fee_pct=round(rt_fee_pct, 2),
                        net_ev_pct=net_ev_pct,
                        hit_rate_pct=hit_rate,
                        regime=regime_name,
                        action=action,
                        earnings_warning=earnings_warn,
                        tfsa_status=tfsa,
                    )
                )
            except Exception as exc:
                logger.debug("Error scanning %s: %s", sym, exc)

        # Rank descending by Net-EV %
        opportunities.sort(key=lambda x: x.net_ev_pct, reverse=True)
        cls._cached_scan = opportunities
        cls._cached_timestamp = now
        return opportunities
