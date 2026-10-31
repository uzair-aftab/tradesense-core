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

import concurrent.futures
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.data_feed import DataLake
from src.data.market_intel import MarketIntelligenceEngine
from src.data.ws_catalog import WealthsimpleAsset, WealthsimpleCatalog, WealthsimpleFeeSchedule
from src.models.regime import MarketRegime, RegimeClassifier
from src.utils.logger import get_logger

logger = get_logger("hfp.profit_engine")

TORONTO_TZ = ZoneInfo("America/Toronto")

TOP_WEALTHSIMPLE_BASKET = [
    "SHOP.TO", "RY.TO", "CLS.TO", "CCO.TO", "ENB.TO",
    "VFV.TO", "XQQ.TO",
    "NVDA.NE", "AAPL.NE", "TSLA.NE", "MSFT.NE", "AMD.NE",
    "BTC-CAD", "ETH-CAD", "SOL-CAD",
    "QQQ", "SPY",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market Hours & Account Shield
# ─────────────────────────────────────────────────────────────────────────────
class MarketHoursGuard:
    """Monitors Toronto market clock and recommends defensive order handling."""

    TORONTO_TZ = ZoneInfo("America/Toronto")

    @classmethod
    def get_market_status(cls, asset: WealthsimpleAsset) -> Dict[str, Any]:
        """Evaluates live market phase and execution safety for the given asset."""
        now_et = datetime.datetime.now(cls.TORONTO_TZ)
        time_str = now_et.strftime("%a %I:%M %p ET")

        if asset.category == "CRYPTO":
            return {
                "phase": "OPEN",
                "is_open": True,
                "msg": "🟢 24/7 Global Crypto Market Active",
                "toronto_time": time_str,
                "order_warning": None,
                "wrapper_advice": "⚠️ WS Crypto is Taxable. For tax-free gains, consider proxy BTCC.TO / ETHX-B.TO inside a TFSA.",
            }

        weekday = now_et.weekday()  # 0=Monday, 6=Sunday
        time_minutes = now_et.hour * 60 + now_et.minute

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
            order_warning = "⚠️ Market closed. ALWAYS use a LIMIT ORDER on Wealthsimple (never market order off-hours)."

        wrapper_advice = "✅ TFSA / FHSA / RRSP Eligible (0% Canadian Capital Gains Tax)"
        if asset.category in ("US_STOCK", "US_ETF") and not asset.symbol.endswith(".NE"):
            wrapper_advice += " • Note: 15% US dividend withholding in TFSA; held best in RRSP."

        return {
            "phase": phase,
            "is_open": is_open,
            "msg": msg,
            "toronto_time": time_str,
            "order_warning": order_warning,
            "wrapper_advice": wrapper_advice,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Position Risk & Kelly Sizer (Scale-Independent)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_cad": self.recommended_cad,
            "recommended_shares": self.recommended_shares,
            "quarter_kelly_fraction": self.quarter_kelly_fraction,
            "max_risk_cap_cad": self.max_risk_cap_cad,
            "dollar_risk_at_stop": self.dollar_risk_at_stop,
            "heat_warning": self.heat_warning,
            "notes": self.notes,
        }


class PositionRiskSizer:
    """Calculates Quarter-Kelly allocations with strict max stop risk and portfolio heat controls."""

    @classmethod
    def calculate_position_size(
        cls,
        total_equity_cad: float,
        current_price: float,
        p10_stop: Optional[float] = None,
        p90_target: Optional[float] = None,
        hit_rate_pct: float = 60.0,
        round_trip_fee_pct: float = 0.08,
        risk_per_trade_pct: float = 1.5,
        existing_heat_cad: float = 0.0,
        max_portfolio_heat_pct: float = 6.0,
        p10_ret_pct: Optional[float] = None,
        p90_ret_pct: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Computes mathematically optimal allocation using Quarter-Kelly and Stop-Loss Risk Cap.
        Supports both absolute price boundaries and percentage returns.
        """
        notes = []
        if total_equity_cad <= 0 or current_price <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, 0.0, 0.0, None, ["Account empty or price invalid"])

        # Resolve percentage returns safely
        if p90_ret_pct is not None and p10_ret_pct is not None:
            upside_pct = max(0.1, float(p90_ret_pct))
            downside_pct = max(0.1, abs(float(p10_ret_pct)))
        elif p90_target is not None and p10_stop is not None and p90_target > 0 and p10_stop > 0:
            upside_pct = max(0.1, ((p90_target - current_price) / current_price) * 100.0)
            downside_pct = max(0.1, ((current_price - p10_stop) / current_price) * 100.0)
        else:
            upside_pct = 2.0
            downside_pct = 1.0

        net_win = max(0.05, upside_pct - round_trip_fee_pct)
        net_loss = downside_pct + round_trip_fee_pct

        # Kelly Criterion
        p = float(np.clip(hit_rate_pct / 100.0, 0.35, 0.75))
        b = max(0.1, net_win / net_loss)
        full_kelly = max(0.0, p - ((1.0 - p) / b))

        # Quarter-Kelly (fractional Kelly prevents drawdown ruin)
        quarter_kelly = float(np.clip(0.25 * full_kelly, 0.0, 0.25))
        kelly_capital = total_equity_cad * quarter_kelly

        # Hard Stop-Risk Cap: Max loss if P10 stop is hit must not exceed risk_per_trade_pct
        max_allowed_dollar_risk = total_equity_cad * (risk_per_trade_pct / 100.0)
        risk_cap_capital = max_allowed_dollar_risk / max(downside_pct / 100.0, 0.005)

        recommended_cad = min(kelly_capital, risk_cap_capital)
        if quarter_kelly <= 0.01:
            recommended_cad = min(recommended_cad, total_equity_cad * 0.05)
            notes.append("Low statistical edge: size restricted to defensive baseline")

        # Portfolio Heat Guard
        max_heat_dollars = total_equity_cad * (max_portfolio_heat_pct / 100.0)
        new_risk_at_stop = recommended_cad * (downside_pct / 100.0)
        total_projected_heat = existing_heat_cad + new_risk_at_stop

        heat_warning = None
        if total_projected_heat > max_heat_dollars:
            available_risk = max(0.0, max_heat_dollars - existing_heat_cad)
            recommended_cad = available_risk / max(downside_pct / 100.0, 0.005)
            heat_warning = f"Portfolio heat limit ({max_portfolio_heat_pct}%): allocation reduced to ${recommended_cad:,.2f} CAD."
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
            return {"stop_price": initial_stop_price, "status": "INITIAL", "desc": "Standard initial stop", "is_breakeven_locked": False}

        target_dist = target_price - entry_price
        gain = current_price - entry_price
        breakeven_price = round(entry_price * (1.0 + (round_trip_fee_pct / 100.0)), 4)

        # 80%+ Target reached -> Chandelier Trailing Stop
        if target_dist > 0 and gain >= 0.80 * target_dist:
            vol_buffer = (atr * 1.2) if (atr and atr > 0) else (current_price * 0.015)
            chandelier_stop = round(current_price - vol_buffer, 4)
            active_stop = max(breakeven_price, chandelier_stop, initial_stop_price)
            return {
                "stop_price": active_stop,
                "status": "CHANDELIER_RUNNER",
                "desc": f"🎯 80%+ Target reached! Trailing runner locked at ${active_stop:,.2f}",
                "is_breakeven_locked": True,
            }

        # 50%+ Target reached -> Risk-Free Breakeven Lock
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
            "desc": f"Initial stop at ${initial_stop_price:,.2f}",
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
    action: str  # "STRONG_BUY", "TACTICAL_BUY", "HOLD_WAIT", "AVOID_FEES"
    earnings_warning: Optional[str]
    tfsa_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "category": self.category,
            "optimal_route_symbol": self.optimal_route_symbol,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "p10_stop": self.p10_stop,
            "upside_pct": self.upside_pct,
            "downside_pct": self.downside_pct,
            "round_trip_fee_pct": self.round_trip_fee_pct,
            "net_ev_pct": self.net_ev_pct,
            "hit_rate_pct": self.hit_rate_pct,
            "regime": self.regime,
            "action": self.action,
            "earnings_warning": self.earnings_warning,
            "tfsa_status": self.tfsa_status,
        }


class AlphaScanner:
    """Scans liquid Wealthsimple universe and ranks opportunities by Net Expected Value %."""

    _cached_scan: Optional[List[AlphaOpportunity]] = None
    _cached_timestamp: Optional[datetime.datetime] = None

    @classmethod
    def _evaluate_single_asset(
        cls,
        sym: str,
        ws_tier: str = "Core",
        usd_account_active: bool = False,
    ) -> Optional[AlphaOpportunity]:
        """Evaluates a single asset using live data and optimal fee route."""
        try:
            primary_asset, exec_asset, is_proxied = WealthsimpleCatalog.get_execution_route(sym, use_proxy=True)
            df = DataLake().pipeline(sym, timeframe="1h", lookback="30d", include_sentiment=False, normalise=False)
            if df is None or len(df) < 25:
                return None

            curr_price = float(df["Close"].iloc[-1])
            regime_ctx = RegimeClassifier().classify(df)
            regime_name = regime_ctx.regime.value

            fee_info = WealthsimpleFeeSchedule.calculate_fee(exec_asset, 500.0, ws_tier, usd_account_active)
            rt_fee_pct = fee_info.round_trip_total_pct

            # Calculate momentum and volatility metrics
            closes = df["Close"].astype(float)
            ema9 = closes.ewm(span=9).mean().iloc[-1]
            ema21 = closes.ewm(span=21).mean().iloc[-1]
            vol_20 = float(closes.pct_change().tail(20).std() * np.sqrt(5)) * 100.0
            drift_pct = float(np.clip(((curr_price - ema21) / curr_price) * 100.0 * 0.5, -4.0, 4.0))

            if regime_ctx.regime == MarketRegime.BULL_TREND:
                upside_pct = max(1.2, vol_20 * 1.1 + drift_pct)
                downside_pct = max(0.8, vol_20 * 0.65)
                hit_rate = 65.0
            elif regime_ctx.regime == MarketRegime.BEAR_TREND:
                upside_pct = max(0.4, vol_20 * 0.5)
                downside_pct = max(1.5, vol_20 * 1.2)
                hit_rate = 36.0
            else:  # Ranging / Chop
                upside_pct = max(0.8, vol_20 * 0.8)
                downside_pct = max(0.8, vol_20 * 0.8)
                hit_rate = 52.0

            p_hit = hit_rate / 100.0
            net_ev_pct = round((p_hit * upside_pct - (1.0 - p_hit) * downside_pct) - rt_fee_pct, 2)

            p90_t = curr_price * (1.0 + upside_pct / 100.0)
            p10_s = curr_price * (1.0 - downside_pct / 100.0)

            # Determine actionable label
            if net_ev_pct >= 1.0 and regime_ctx.regime == MarketRegime.BULL_TREND:
                action = "STRONG_BUY"
            elif net_ev_pct >= 0.2:
                action = "TACTICAL_BUY"
            elif net_ev_pct >= -0.5:
                action = "HOLD_WAIT"
            else:
                action = "AVOID_FEES"

            tfsa = "TFSA Eligible (0% Tax)" if primary_asset.category != "CRYPTO" else (
                f"Tax-Free via {exec_asset.symbol}" if is_proxied else "Taxable (Consider Proxy)"
            )

            return AlphaOpportunity(
                symbol=sym,
                name=primary_asset.name,
                category=primary_asset.category,
                optimal_route_symbol=exec_asset.symbol,
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
                earnings_warning=None,
                tfsa_status=tfsa,
            )
        except Exception as exc:
            logger.debug("Error scanning %s: %s", sym, exc)
            return None

    @classmethod
    def scan_opportunities(
        cls,
        basket: Optional[List[str]] = None,
        force_refresh: bool = False,
        ws_tier: str = "Core",
        usd_account_active: bool = False,
    ) -> List[AlphaOpportunity]:
        """Scans the basket in parallel and returns Net-EV ranked opportunities."""
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(cls._evaluate_single_asset, sym, ws_tier, usd_account_active): sym
                for sym in tickers
            }
            for fut in concurrent.futures.as_completed(futures):
                opp = fut.result()
                if opp is not None:
                    opportunities.append(opp)

        opportunities.sort(key=lambda x: x.net_ev_pct, reverse=True)
        cls._cached_scan = opportunities
        cls._cached_timestamp = now
        return opportunities
