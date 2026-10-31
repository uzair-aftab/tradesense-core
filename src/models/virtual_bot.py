"""
TradeSense Persistent Virtual-Money Proof-of-Edge Bot Engine
============================================================
Allows users to prove algorithmic trading edge on virtual money before risking real CAD:
  - SQLite WAL persistent virtual CAD wallet ($1,000 - $250,000 CAD)
  - Real-time automated virtual trading with exact Wealthsimple mobile fee/spread friction
  - 1-Click Zero-Lookahead 60-Bar Walk-Forward Pair Efficiency Evaluator
  - Unmistakable Readiness Verdict (🟢 Proven Edge | 🟡 Calibrating | 🔴 Unprofitable After Fees)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.database import DatabaseManager, db_manager
from src.data.ws_catalog import WealthsimpleAsset, WealthsimpleCatalog, WealthsimpleFeeSchedule
from src.models.adaptive_predictor import AdaptiveForecast, AdaptivePredictionEngine, adaptive_engine
from src.utils.logger import get_logger

logger = get_logger("hfp.virtual_bot")


@dataclass
class BotEfficiencySummary:
    """Quantitative performance and Wealthsimple-readiness scorecard for a pair."""
    ticker: str
    ws_symbol: str
    timeframe: str
    eval_bars: int
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    hit_rate_pct: float          # Directional forecast accuracy
    gross_return_pct: float
    total_ws_fees_cad: float     # Exact Wealthsimple commission + spread friction deducted
    net_return_cad: float
    net_return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    readiness_state: str         # "PROVEN_EDGE" | "CALIBRATING" | "AVOID"
    readiness_badge: str         # e.g., "🟢 Proven Edge — Ready for Wealthsimple"
    readiness_explanation: str
    optimization_tip: Optional[str] = None
    trades_history: List[Dict[str, Any]] = field(default_factory=list)


class VirtualBotEngine:
    """Virtual-money execution bot and walk-forward efficiency evaluator."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Live Tick Event Handler (Persistent Virtual CAD Trading)
    # ─────────────────────────────────────────────────────────────────────────
    def on_live_tick(
        self,
        user_id: str,
        asset: WealthsimpleAsset,
        current_price: float,
        forecast: AdaptiveForecast,
        fx_usd_cad: float = 1.39,
    ) -> Dict[str, Any]:
        """
        Processes a live tick for the active asset:
          1. Updates open virtual positions (stop-loss, take-profit, time decay).
          2. Evaluates fee-aware entry gates if armed and flat.
        """
        acct = self.db.get_or_create_virtual_account(user_id)
        if not acct.get("is_armed", 1):
            return {"status": "DISARMED", "message": "Virtual Bot is currently paused."}

        ws_tier = acct.get("ws_tier", "Core")
        usd_account = bool(acct.get("usd_account_active", 0))
        fee_est = WealthsimpleFeeSchedule.calculate_fee(asset, 1000.0, ws_tier, usd_account)

        # ── Step A: Check Existing Virtual Positions for this user & ticker ──
        positions = self.db.get_virtual_positions(user_id)
        pos = next((p for p in positions if p["ticker"] == asset.symbol), None)

        if pos is not None:
            entry_price = float(pos["entry_price"])
            quantity = float(pos["quantity"])
            target_p90 = float(pos["target_p90"])
            stop_p10 = float(pos["stop_p10"])
            bars_held = int(pos.get("bars_held", 0)) + 1
            horizon_bars = int(pos.get("horizon_bars", 5))

            # Mark to market in CAD
            curr_val_native = quantity * current_price
            val_cad = curr_val_native * (fx_usd_cad if asset.currency == "USD" else 1.0)

            close_reason = None
            if current_price >= target_p90:
                close_reason = f"TARGET_HIT (P90 {target_p90:.2f})"
            elif current_price <= stop_p10:
                close_reason = f"STOP_LOSS (P10 {stop_p10:.2f})"
            elif forecast.direction == "BEARISH" and forecast.confidence_pct >= 68.0:
                close_reason = f"FORECAST_REVERSAL (Conf {forecast.confidence_pct:.0f}%)"
            elif bars_held >= horizon_bars * 3:
                close_reason = f"TIME_EXPIRY ({bars_held} bars)"

            if close_reason:
                # Deduct exit fee
                exit_fee_cad = val_cad * (fee_est.one_way_fee_pct + fee_est.one_way_slippage_pct) / 100.0
                entry_fee_cad = float(pos.get("entry_fee_cad", 0.0))
                total_fees_cad = entry_fee_cad + exit_fee_cad

                entry_val_cad = (quantity * entry_price) * (fx_usd_cad if asset.currency == "USD" else 1.0)
                gross_pnl_cad = val_cad - entry_val_cad
                net_pnl_cad = gross_pnl_cad - total_fees_cad
                net_pnl_pct = (net_pnl_cad / max(entry_val_cad, 1e-4)) * 100.0

                prediction_hit = (current_price > entry_price) if pos["side"] == "BUY" else (current_price < entry_price)

                # Update wallet balance
                new_cash = float(acct["virtual_cash_cad"]) + val_cad - exit_fee_cad
                self.db.update_virtual_account(user_id, virtual_cash_cad=new_cash)

                # Record trade and remove position
                self.db.record_virtual_trade(
                    user_id=user_id,
                    ticker=asset.symbol,
                    ws_symbol=asset.ws_display_symbol,
                    entry_price=entry_price,
                    exit_price=current_price,
                    predicted_p50=float(pos["predicted_p50"]),
                    quantity=quantity,
                    gross_pnl_cad=gross_pnl_cad,
                    ws_fees_cad=total_fees_cad,
                    net_pnl_cad=net_pnl_cad,
                    net_pnl_pct=net_pnl_pct,
                    prediction_hit=prediction_hit,
                    exit_reason=close_reason,
                    mode="LIVE_VIRTUAL",
                    opened_at=pos["opened_at"],
                )
                self.db.delete_virtual_position(user_id, asset.symbol)

                return {
                    "status": "POSITION_CLOSED",
                    "reason": close_reason,
                    "net_pnl_cad": round(net_pnl_cad, 2),
                    "net_pnl_pct": round(net_pnl_pct, 2),
                }
            else:
                # Update current position tracking
                self.db.save_virtual_position(
                    user_id=user_id,
                    ticker=asset.symbol,
                    ws_symbol=asset.ws_display_symbol,
                    side=pos["side"],
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    predicted_p50=forecast.p50,
                    target_p90=forecast.p90,
                    stop_p10=forecast.p10,
                    entry_fee_cad=float(pos.get("entry_fee_cad", 0.0)),
                    dna_profile=forecast.dna_profile,
                    horizon_bars=forecast.horizon_bars,
                )
                return {
                    "status": "POSITION_HOLDING",
                    "unrealized_pnl_cad": round(val_cad - ((quantity * entry_price) * (fx_usd_cad if asset.currency == "USD" else 1.0)), 2),
                }

        # ── Step B: Fee-Aware Entry Gate (If Flat) ──
        if forecast.direction == "BULLISH" and forecast.confidence_pct >= 62.0:
            # Check fee hurdle: expected upside must clear round-trip friction by at least 1.6x
            fee_hurdle = max(1.0, fee_est.round_trip_total_pct * 1.60)
            if forecast.upside_pct >= fee_hurdle and forecast.rrr >= 1.35:
                cash_cad = float(acct["virtual_cash_cad"])
                if cash_cad >= 100.0:
                    # Allocate fixed risk (e.g. 2% of total equity risked, max 25% of cash on one trade)
                    risk_cad = cash_cad * 0.02
                    price_cad = current_price * (fx_usd_cad if asset.currency == "USD" else 1.0)
                    stop_dist_pct = max(0.015, (current_price - forecast.p10) / current_price)
                    target_capital_cad = min(cash_cad * 0.25, risk_cad / stop_dist_pct)
                    target_capital_cad = max(100.0, target_capital_cad)

                    qty = (target_capital_cad / price_cad)
                    if not asset.fractional_supported:
                        qty = math.floor(qty)

                    if qty > 0:
                        entry_fee_cad = target_capital_cad * (fee_est.one_way_fee_pct + fee_est.one_way_slippage_pct) / 100.0
                        total_deduct = target_capital_cad + entry_fee_cad

                        if cash_cad >= total_deduct:
                            self.db.update_virtual_account(user_id, virtual_cash_cad=cash_cad - total_deduct)
                            self.db.save_virtual_position(
                                user_id=user_id,
                                ticker=asset.symbol,
                                ws_symbol=asset.ws_display_symbol,
                                side="BUY",
                                quantity=qty,
                                entry_price=current_price,
                                current_price=current_price,
                                predicted_p50=forecast.p50,
                                target_p90=forecast.p90,
                                stop_p10=forecast.p10,
                                entry_fee_cad=entry_fee_cad,
                                dna_profile=forecast.dna_profile,
                                horizon_bars=forecast.horizon_bars,
                            )
                            return {
                                "status": "POSITION_OPENED",
                                "quantity": round(qty, 4),
                                "entry_price": current_price,
                                "target": forecast.p90,
                                "stop": forecast.p10,
                            }

        return {"status": "WAITING_SIGNAL", "reason": "Signal did not meet fee hurdle or confidence threshold."}

    def manual_open_position(
        self,
        user_id: str,
        asset: WealthsimpleAsset,
        current_price: float,
        p10_stop: float,
        p50_target: float,
        p90_target: float,
        capital_cad: float = 500.0,
        fx_usd_cad: float = 1.39,
        dna_profile: str = "MANUAL",
        horizon_bars: int = 5,
    ) -> Dict[str, Any]:
        """Manually opens a virtual position for testing or user-initiated trades."""
        acct = self.db.get_or_create_virtual_account(user_id)
        cash_cad = float(acct["virtual_cash_cad"])
        ws_tier = acct.get("ws_tier", "Core")
        usd_account = bool(acct.get("usd_account_active", 0))

        if capital_cad > cash_cad:
            return {"status": "INSUFFICIENT_FUNDS", "message": f"Requires ${capital_cad:,.2f} CAD, but cash is ${cash_cad:,.2f} CAD."}

        fee_est = WealthsimpleFeeSchedule.calculate_fee(asset, capital_cad, ws_tier, usd_account)
        entry_fee_cad = capital_cad * (fee_est.one_way_fee_pct + fee_est.one_way_slippage_pct) / 100.0
        total_deduct = capital_cad + entry_fee_cad

        if total_deduct > cash_cad:
            return {"status": "INSUFFICIENT_FUNDS", "message": f"Total cost with fees (${total_deduct:,.2f}) exceeds cash (${cash_cad:,.2f})."}

        price_cad = current_price * (fx_usd_cad if asset.currency == "USD" else 1.0)
        qty = capital_cad / price_cad
        if not asset.fractional_supported:
            qty = math.floor(qty)
            if qty <= 0:
                return {"status": "INVALID_QUANTITY", "message": "Capital insufficient for 1 whole share."}

        self.db.update_virtual_account(user_id, virtual_cash_cad=cash_cad - total_deduct)
        self.db.save_virtual_position(
            user_id=user_id,
            ticker=asset.symbol,
            ws_symbol=asset.ws_display_symbol,
            side="BUY",
            quantity=qty,
            entry_price=current_price,
            current_price=current_price,
            predicted_p50=p50_target,
            target_p90=p90_target,
            stop_p10=p10_stop,
            entry_fee_cad=entry_fee_cad,
            dna_profile=dna_profile,
            horizon_bars=horizon_bars,
        )
        return {
            "status": "POSITION_OPENED",
            "quantity": round(qty, 4),
            "entry_price": current_price,
            "target": p90_target,
            "stop": p10_stop,
            "capital_cad": capital_cad,
            "fee_cad": entry_fee_cad,
        }

    def manual_close_position(
        self,
        user_id: str,
        asset: WealthsimpleAsset,
        current_price: float,
        exit_reason: str = "MANUAL_CLOSE",
        fx_usd_cad: float = 1.39,
    ) -> Dict[str, Any]:
        """Manually closes an existing virtual position and realizes P&L."""
        acct = self.db.get_or_create_virtual_account(user_id)
        positions = self.db.get_virtual_positions(user_id)
        pos = next((p for p in positions if p["ticker"] == asset.symbol), None)
        if not pos:
            return {"status": "NOT_FOUND", "message": f"No open position found for {asset.symbol}."}

        ws_tier = acct.get("ws_tier", "Core")
        usd_account = bool(acct.get("usd_account_active", 0))
        quantity = float(pos["quantity"])
        entry_price = float(pos["entry_price"])
        val_cad = (quantity * current_price) * (fx_usd_cad if asset.currency == "USD" else 1.0)

        fee_est = WealthsimpleFeeSchedule.calculate_fee(asset, val_cad, ws_tier, usd_account)
        exit_fee_cad = val_cad * (fee_est.one_way_fee_pct + fee_est.one_way_slippage_pct) / 100.0
        entry_fee_cad = float(pos.get("entry_fee_cad", 0.0))
        total_fees_cad = entry_fee_cad + exit_fee_cad

        entry_val_cad = (quantity * entry_price) * (fx_usd_cad if asset.currency == "USD" else 1.0)
        gross_pnl_cad = val_cad - entry_val_cad
        net_pnl_cad = gross_pnl_cad - total_fees_cad
        net_pnl_pct = (net_pnl_cad / max(entry_val_cad, 1e-4)) * 100.0
        prediction_hit = current_price > entry_price

        new_cash = float(acct["virtual_cash_cad"]) + val_cad - exit_fee_cad
        self.db.update_virtual_account(user_id, virtual_cash_cad=new_cash)
        self.db.record_virtual_trade(
            user_id=user_id,
            ticker=asset.symbol,
            ws_symbol=asset.ws_display_symbol,
            entry_price=entry_price,
            exit_price=current_price,
            predicted_p50=float(pos.get("predicted_p50", entry_price)),
            quantity=quantity,
            gross_pnl_cad=gross_pnl_cad,
            ws_fees_cad=total_fees_cad,
            net_pnl_cad=net_pnl_cad,
            net_pnl_pct=net_pnl_pct,
            prediction_hit=prediction_hit,
            exit_reason=exit_reason,
            mode="LIVE_VIRTUAL",
            opened_at=pos.get("opened_at"),
        )
        self.db.delete_virtual_position(user_id, asset.symbol)

        return {
            "status": "POSITION_CLOSED",
            "reason": exit_reason,
            "net_pnl_cad": round(net_pnl_cad, 2),
            "net_pnl_pct": round(net_pnl_pct, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Instant Zero-Lookahead Walk-Forward Efficiency Evaluator
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_pair_efficiency(
        self,
        df: pd.DataFrame,
        asset: WealthsimpleAsset,
        timeframe: str = "1h",
        horizon_bars: int = 5,
        eval_bars: int = 60,
        starting_cad: float = 10000.0,
        ws_tier: str = "Core",
        usd_account: bool = False,
        fx_usd_cad: float = 1.39,
    ) -> BotEfficiencySummary:
        """
        Runs an instant, zero-lookahead walk-forward test of the Virtual Bot on
        the trailing eval_bars of this specific pair using exact Wealthsimple fees.
        """
        if df is None or len(df) < 35:
            return BotEfficiencySummary(
                ticker=asset.symbol,
                ws_symbol=asset.ws_display_symbol,
                timeframe=timeframe,
                eval_bars=0,
                trades_count=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                hit_rate_pct=0.0,
                gross_return_pct=0.0,
                total_ws_fees_cad=0.0,
                net_return_cad=0.0,
                net_return_pct=0.0,
                profit_factor=0.0,
                max_drawdown_pct=0.0,
                readiness_state="CALIBRATING",
                readiness_badge="🟡 CALIBRATING — INSUFFICIENT DATA",
                readiness_explanation=f"Awaiting at least 35 bars of history for {asset.symbol}.",
            )

        fee_est = WealthsimpleFeeSchedule.calculate_fee(asset, 1000.0, ws_tier, usd_account)
        round_trip_fee_pct = fee_est.round_trip_total_pct
        one_way_fee_pct = fee_est.one_way_fee_pct + fee_est.one_way_slippage_pct

        total_bars = len(df)
        start_idx = max(25, total_bars - eval_bars)
        actual_eval_bars = total_bars - start_idx

        # Simulation state
        cash_cad = starting_cad
        equity_curve = [starting_cad]
        trades: List[Dict[str, Any]] = []

        in_pos = False
        pos_entry_price = 0.0
        pos_qty = 0.0
        pos_entry_idx = 0
        pos_target = 0.0
        pos_stop = 0.0
        pos_p50 = 0.0
        pos_capital_cad = 0.0
        pos_entry_fee_cad = 0.0

        step_jump = max(2, horizon_bars)

        for i in range(start_idx, total_bars - 1):
            sub_df = df.iloc[:i + 1]
            curr_bar = df.iloc[i]
            curr_price = float(curr_bar["Close"])
            high_price = float(curr_bar["High"]) if "High" in curr_bar else curr_price
            low_price = float(curr_bar["Low"]) if "Low" in curr_bar else curr_price

            # Check open position exit
            if in_pos:
                bars_held = i - pos_entry_idx
                exit_price = None
                exit_reason = None

                if high_price >= pos_target:
                    exit_price = pos_target
                    exit_reason = "TAKE_PROFIT_P90"
                elif low_price <= pos_stop:
                    exit_price = pos_stop
                    exit_reason = "STOP_LOSS_P10"
                elif bars_held >= horizon_bars:
                    exit_price = curr_price
                    exit_reason = f"HORIZON_EXPIRY ({horizon_bars} bars)"

                if exit_price is not None:
                    # Calculate realized P&L with Wealthsimple fees
                    exit_val_cad = pos_qty * exit_price * (fx_usd_cad if asset.currency == "USD" else 1.0)
                    exit_fee_cad = exit_val_cad * (one_way_fee_pct / 100.0)
                    total_fee_cad = pos_entry_fee_cad + exit_fee_cad

                    gross_pnl_cad = exit_val_cad - pos_capital_cad
                    net_pnl_cad = gross_pnl_cad - total_fee_cad
                    net_pnl_pct = (net_pnl_cad / max(pos_capital_cad, 1e-4)) * 100.0
                    pred_hit = (exit_price > pos_entry_price)

                    cash_cad += (exit_val_cad - exit_fee_cad)
                    equity_curve.append(cash_cad)

                    trades.append({
                        "ticker": asset.symbol,
                        "entry_idx": pos_entry_idx,
                        "exit_idx": i,
                        "entry_price": pos_entry_price,
                        "exit_price": exit_price,
                        "predicted_p50": pos_p50,
                        "gross_pnl_cad": round(gross_pnl_cad, 2),
                        "ws_fees_cad": round(total_fee_cad, 2),
                        "net_pnl_cad": round(net_pnl_cad, 2),
                        "net_pnl_pct": round(net_pnl_pct, 2),
                        "prediction_hit": pred_hit,
                        "exit_reason": exit_reason,
                    })
                    in_pos = False

            # Check new entry if flat (evaluated every step_jump bars)
            if not in_pos and (i % step_jump == 0):
                forecast = adaptive_engine.compute_adaptive_forecast(
                    sub_df, asset, timeframe, horizon_bars=horizon_bars
                )

                fee_hurdle = max(0.9, round_trip_fee_pct * 1.50)
                if (
                    forecast.direction == "BULLISH"
                    and forecast.confidence_pct >= 60.0
                    and forecast.upside_pct >= fee_hurdle
                    and forecast.rrr >= 1.30
                ):
                    pos_capital_cad = min(cash_cad * 0.25, 2500.0)
                    if pos_capital_cad >= 100.0:
                        price_cad = curr_price * (fx_usd_cad if asset.currency == "USD" else 1.0)
                        pos_qty = pos_capital_cad / price_cad
                        pos_entry_fee_cad = pos_capital_cad * (one_way_fee_pct / 100.0)

                        cash_cad -= (pos_capital_cad + pos_entry_fee_cad)
                        in_pos = True
                        pos_entry_price = curr_price
                        pos_entry_idx = i
                        pos_target = forecast.p90
                        pos_stop = forecast.p10
                        pos_p50 = forecast.p50

        # Close open position at end of simulation if any
        if in_pos:
            last_price = float(df["Close"].iloc[-1])
            exit_val_cad = pos_qty * last_price * (fx_usd_cad if asset.currency == "USD" else 1.0)
            exit_fee_cad = exit_val_cad * (one_way_fee_pct / 100.0)
            total_fee_cad = pos_entry_fee_cad + exit_fee_cad

            gross_pnl_cad = exit_val_cad - pos_capital_cad
            net_pnl_cad = gross_pnl_cad - total_fee_cad
            net_pnl_pct = (net_pnl_cad / max(pos_capital_cad, 1e-4)) * 100.0

            cash_cad += (exit_val_cad - exit_fee_cad)
            trades.append({
                "ticker": asset.symbol,
                "entry_idx": pos_entry_idx,
                "exit_idx": total_bars - 1,
                "entry_price": pos_entry_price,
                "exit_price": last_price,
                "predicted_p50": pos_p50,
                "gross_pnl_cad": round(gross_pnl_cad, 2),
                "ws_fees_cad": round(total_fee_cad, 2),
                "net_pnl_cad": round(net_pnl_cad, 2),
                "net_pnl_pct": round(net_pnl_pct, 2),
                "prediction_hit": (last_price > pos_entry_price),
                "exit_reason": "SIMULATION_END",
            })

        # Summary calculations
        trades_count = len(trades)
        winning_trades = sum(1 for t in trades if t["net_pnl_cad"] > 0)
        losing_trades = sum(1 for t in trades if t["net_pnl_cad"] <= 0)
        win_rate_pct = round((winning_trades / trades_count * 100.0), 1) if trades_count > 0 else 0.0

        hits_count = sum(1 for t in trades if t["prediction_hit"])
        hit_rate_pct = round((hits_count / trades_count * 100.0), 1) if trades_count > 0 else 0.0

        total_fees = sum(t["ws_fees_cad"] for t in trades)
        gross_pnl = sum(t["gross_pnl_cad"] for t in trades)
        net_return_cad = round(cash_cad - starting_cad, 2)
        net_return_pct = round((net_return_cad / starting_cad) * 100.0, 2)
        gross_return_pct = round((gross_pnl / starting_cad) * 100.0, 2)

        gains = sum(t["net_pnl_cad"] for t in trades if t["net_pnl_cad"] > 0)
        losses = abs(sum(t["net_pnl_cad"] for t in trades if t["net_pnl_cad"] < 0))
        profit_factor = round(gains / max(losses, 1e-4), 2) if losses > 0 else (2.5 if gains > 0 else 1.0)

        # Max drawdown
        peak = starting_cad
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / max(peak, 1.0) * 100.0
            if dd > max_dd:
                max_dd = dd

        # Readiness Verdict
        tip = fee_est.optimization_tip
        if net_return_pct > 0 and profit_factor >= 1.30 and hit_rate_pct >= 54.0 and trades_count >= 3:
            readiness_state = "PROVEN_EDGE"
            readiness_badge = "🟢 PROVEN EDGE — READY FOR WEALTHSIMPLE"
            explanation = (
                f"Generated {net_return_pct:+.2f}% net return (${net_return_cad:+.2f} CAD) across {trades_count} "
                f"virtual trades after deducting ${total_fees:.2f} CAD in Wealthsimple fees. Profit factor is {profit_factor:.2f}."
            )
        elif trades_count < 3 or (1.0 <= profit_factor < 1.30 and net_return_pct >= -0.5):
            readiness_state = "CALIBRATING"
            readiness_badge = "🟡 CALIBRATING — CONTINUE VIRTUAL TESTING"
            explanation = (
                f"Sample size ({trades_count} trades) or edge is currently calibrating. "
                f"Net return is {net_return_pct:+.2f}% with {win_rate_pct:.0f}% win rate. Keep testing on virtual cash."
            )
        else:
            readiness_state = "AVOID"
            readiness_badge = "🔴 STAY VIRTUAL — FEES/RISK EXCEED EDGE"
            if gross_return_pct > 0 and net_return_pct <= 0:
                explanation = (
                    f"Gross predictions were profitable (+{gross_return_pct:.2f}%), but Wealthsimple "
                    f"fees (${total_fees:.2f} CAD, ~{round_trip_fee_pct:.2f}% RT) eroded net returns into {net_return_pct:.2f}%. "
                    f"Do NOT execute real money on this direct asset."
                )
            else:
                explanation = (
                    f"Walk-forward test produced a net drawdown of {net_return_pct:.2f}% on virtual money. "
                    f"Model is protecting real capital by staying in virtual test mode."
                )

        return BotEfficiencySummary(
            ticker=asset.symbol,
            ws_symbol=asset.ws_display_symbol,
            timeframe=timeframe,
            eval_bars=actual_eval_bars,
            trades_count=trades_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate_pct,
            hit_rate_pct=hit_rate_pct,
            gross_return_pct=gross_return_pct,
            total_ws_fees_cad=round(total_fees, 2),
            net_return_cad=net_return_cad,
            net_return_pct=net_return_pct,
            profit_factor=profit_factor,
            max_drawdown_pct=round(max_dd, 2),
            readiness_state=readiness_state,
            readiness_badge=readiness_badge,
            readiness_explanation=explanation,
            optimization_tip=tip,
            trades_history=trades,
        )


# Global singleton instance
virtual_bot_engine = VirtualBotEngine()
