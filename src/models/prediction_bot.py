"""
TradeSense v8 — Autonomous Wealthsimple Predict Paper Trading Bot & Brier Auditor
================================================================================
Fills the missing Practice Account gap on Wealthsimple Predict:
  - Dedicated $500 USD virtual sandbox bankroll per user profile
  - Fractional contracts from $1.00 USD minimum with exact Kalshi fees
  - Early exits at live Kalshi bids & automatic $1.00/$0.00 resolution settlement
  - Optional 60s Autonomous Paper Auto-Pilot ($5 micro-stakes on >= +4% Net EV)
  - Strict Brier Score Proof-of-Edge vs Market Implied Crowd
  - Clean PredictionExecutionAdapter interface (zero unauthorized scraping)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import datetime
from typing import Any, Dict, List, Optional

from src.data.database import DatabaseManager, db_manager
from src.data.market_intel import market_intel
from src.data.prediction_feed import PredictionMarket, prediction_feed
from src.data.profiles import profile_manager
from src.models.prediction_engine import PredictionAlphaScanner, PredictionFeeSchedule
from src.utils.logger import get_logger

logger = get_logger("hfp.prediction_bot")


class PredictionExecutionAdapter(ABC):
    """Clean broker seam keeping execution decoupled from Wealthsimple UI."""

    @abstractmethod
    def submit_order(
        self,
        market: PredictionMarket,
        side: str,
        stake_usd: float,
        limit_price_usd: float,
    ) -> Dict[str, Any]:
        """Execute a prediction contract order."""


class PaperKalshiAdapter(PredictionExecutionAdapter):
    """Default zero-risk simulated execution using live Kalshi v2 orderbook prices."""

    def submit_order(
        self,
        market: PredictionMarket,
        side: str,
        stake_usd: float,
        limit_price_usd: float,
    ) -> Dict[str, Any]:
        price = min(max(float(limit_price_usd), 0.01), 0.99)
        contracts = stake_usd / price
        fee_usd = PredictionFeeSchedule.calculate_taker_fee_usd(contracts, price)
        return {
            "status": "FILLED",
            "adapter": "PAPER_KALSHI",
            "contracts": round(contracts, 4),
            "fill_price_usd": round(price, 4),
            "fee_usd": round(fee_usd, 2),
        }


class PredictionBotEngine:
    """Manages $500 USD virtual prediction sandboxes, auto-settlement, auto-pilot, and Brier scores."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        adapter: Optional[PredictionExecutionAdapter] = None,
    ):
        self.db = db or db_manager
        self.adapter = adapter or PaperKalshiAdapter()

    @staticmethod
    def _live_fx() -> float:
        try:
            snap = market_intel.get_macro_pulse(force_refresh=False)
            return float(getattr(snap, "usd_cad", 1.39) or 1.39)
        except Exception:
            return 1.39

    def execute_paper_buy(
        self,
        user_id: str,
        market_ticker: str,
        side: str,
        stake_usd: float = 5.0,
        ai_fair_prob: Optional[float] = None,
        market_override: Optional[PredictionMarket] = None,
    ) -> Dict[str, Any]:
        """Open a fractional virtual YES/NO prediction trade ($1.00 USD minimum)."""
        uid = str(user_id).lower().strip()
        side_up = str(side).upper().strip()
        if side_up not in ("YES", "NO"):
            return {"status": "ERROR", "message": "Side must be YES or NO."}

        stake = float(stake_usd)
        if stake < 1.0:
            return {"status": "ERROR", "message": "Minimum Wealthsimple Predict stake is $1.00 USD."}

        market = market_override or prediction_feed.fetch_market(market_ticker, bypass_cache=True)
        if not market:
            return {"status": "ERROR", "message": f"Unable to fetch live market quote for {market_ticker}."}

        entry_price = market.yes_ask if side_up == "YES" else market.no_ask
        if entry_price <= 0.0 or entry_price >= 1.0:
            entry_price = market.mid_prob if side_up == "YES" else (1.0 - market.mid_prob)
        entry_price = min(max(entry_price, 0.01), 0.99)

        fill = self.adapter.submit_order(market, side_up, stake, entry_price)
        fee_usd = float(fill["fee_usd"])
        contracts = float(fill["contracts"])
        total_cost_usd = round(stake + fee_usd, 2)

        acct = self.db.get_or_create_prediction_account(uid)
        cash_usd = float(acct["cash_usd"])
        if total_cost_usd > cash_usd:
            return {
                "status": "ERROR",
                "message": f"Insufficient Predict sandbox cash (${cash_usd:.2f} USD available vs ${total_cost_usd:.2f} needed).",
            }

        fx = self._live_fx()
        fair_yes = float(ai_fair_prob) if ai_fair_prob is not None else float(market.mid_prob)

        self.db.update_prediction_account(uid, cash_usd=round(cash_usd - total_cost_usd, 2))
        pos_id = self.db.save_prediction_position({
            "user_id": uid,
            "event_ticker": market.event_ticker,
            "market_ticker": market.market_ticker,
            "series_ticker": market.series_ticker,
            "event_title": market.event_title,
            "outcome_label": market.outcome_label,
            "side": side_up,
            "contracts": contracts,
            "entry_price_usd": entry_price,
            "current_price_usd": entry_price,
            "entry_fee_usd": fee_usd,
            "stake_usd": round(stake, 2),
            "stake_cad": round(stake * fx, 2),
            "fx_rate": fx,
            "ai_fair_prob": fair_yes,
            "market_prob_entry": market.mid_prob,
            "close_time": market.close_time,
        })

        msg = (
            f"Bought {contracts:.2f} {side_up} '{market.outcome_label}' ({market.market_ticker}) "
            f"@ {entry_price * 100.0:.1f}c (${stake:.2f} USD + ${fee_usd:.2f} fee)"
        )
        logger.info("[%s] PREDICT_BUY: %s", uid, msg)
        return {
            "status": "SUCCESS",
            "message": msg,
            "position_id": pos_id,
            "contracts": contracts,
            "entry_price_usd": entry_price,
            "fee_usd": fee_usd,
            "cash_usd": round(cash_usd - total_cost_usd, 2),
        }

    def execute_paper_close(
        self,
        user_id: str,
        position_id: int,
        exit_reason: str = "EARLY_EXIT",
        settled_result: Optional[str] = None,
        market_override: Optional[PredictionMarket] = None,
    ) -> Dict[str, Any]:
        """Early exit at live Kalshi bid OR settle resolved contract at $1.00 / $0.00."""
        uid = str(user_id).lower().strip()
        pos = self.db.get_prediction_position_by_id(uid, position_id)
        if not pos:
            return {"status": "ERROR", "message": f"Open position #{position_id} not found."}

        side_up = pos["side"].upper()
        contracts = float(pos["contracts"])
        stake_usd = float(pos["stake_usd"])
        entry_fee_usd = float(pos.get("entry_fee_usd", 0.0))
        entry_price = float(pos["entry_price_usd"])
        fx = float(pos.get("fx_rate", 1.39)) or self._live_fx()

        brier_ai = None
        brier_market = None

        if settled_result and settled_result.lower() in ("yes", "no"):
            res_low = settled_result.lower()
            won = res_low == side_up.lower()
            exit_price = 1.0 if won else 0.0
            exit_fee = 0.0
            actual_yes = 1.0 if res_low == "yes" else 0.0
            brier_ai = round((float(pos["ai_fair_prob"]) - actual_yes) ** 2, 5)
            brier_market = round((float(pos["market_prob_entry"]) - actual_yes) ** 2, 5)
        else:
            mkt = market_override or prediction_feed.fetch_market(pos["market_ticker"], bypass_cache=True)
            if mkt and mkt.status == "settled" and mkt.result in ("yes", "no"):
                return self.execute_paper_close(
                    uid,
                    position_id,
                    exit_reason="SETTLED_AUTO",
                    settled_result=mkt.result,
                    market_override=mkt,
                )
            if mkt:
                bid = mkt.yes_bid if side_up == "YES" else mkt.no_bid
                exit_price = bid if bid > 0 else float(pos["current_price_usd"])
            else:
                exit_price = float(pos["current_price_usd"])
            exit_price = min(max(exit_price, 0.01), 0.99)
            exit_fee = PredictionFeeSchedule.calculate_taker_fee_usd(contracts, exit_price)

        gross_proceeds = contracts * exit_price
        net_proceeds = max(0.0, gross_proceeds - exit_fee)
        gross_pnl_usd = gross_proceeds - stake_usd
        total_fees_usd = entry_fee_usd + exit_fee
        net_pnl_usd = net_proceeds - (stake_usd + entry_fee_usd)
        net_pnl_cad = net_pnl_usd * fx
        roi_pct = (net_pnl_usd / max(stake_usd + entry_fee_usd, 0.01)) * 100.0
        outcome_won = 1 if net_pnl_usd > 0 else 0

        acct = self.db.get_or_create_prediction_account(uid)
        new_cash = round(float(acct["cash_usd"]) + net_proceeds, 2)
        self.db.update_prediction_account(uid, cash_usd=new_cash)

        self.db.record_prediction_trade({
            "user_id": uid,
            "event_ticker": pos["event_ticker"],
            "market_ticker": pos["market_ticker"],
            "event_title": pos["event_title"],
            "outcome_label": pos["outcome_label"],
            "side": side_up,
            "contracts": contracts,
            "entry_price_usd": entry_price,
            "exit_price_usd": round(exit_price, 4),
            "stake_usd": stake_usd,
            "gross_pnl_usd": round(gross_pnl_usd, 2),
            "fees_usd": round(total_fees_usd, 2),
            "net_pnl_usd": round(net_pnl_usd, 2),
            "net_pnl_cad": round(net_pnl_cad, 2),
            "roi_pct": round(roi_pct, 2),
            "ai_fair_prob": float(pos["ai_fair_prob"]),
            "market_prob_entry": float(pos["market_prob_entry"]),
            "outcome_won": outcome_won,
            "brier_ai": brier_ai,
            "brier_market": brier_market,
            "exit_reason": exit_reason,
            "opened_at": pos.get("opened_at"),
        })
        self.db.delete_prediction_position(position_id)

        msg = (
            f"Closed {side_up} '{pos['outcome_label']}' ({exit_reason}) @ {exit_price * 100.0:.1f}c "
            f"-> Net PnL: ${net_pnl_usd:+,.2f} USD ({roi_pct:+.1f}%)"
        )
        logger.info("[%s] PREDICT_CLOSE: %s", uid, msg)
        return {
            "status": "SUCCESS",
            "message": msg,
            "net_pnl_usd": round(net_pnl_usd, 2),
            "net_pnl_cad": round(net_pnl_cad, 2),
            "roi_pct": round(roi_pct, 2),
            "cash_usd": new_cash,
        }

    def run_prediction_cycle(self) -> None:
        """
        Called every 60s by FastAPI background runner:
          1. Mark open Predict positions to live Kalshi bids & auto-settle resolved contracts
          2. For accounts with auto_pilot=1, execute at most 1 micro $5 paper trade
             per cycle when Net EV >= +4.0% on a liquid WS-eligible event not yet held
        """
        profiles = profile_manager.list_profiles()
        scan_cache: Optional[Dict[str, Any]] = None

        for prof in profiles:
            uid = prof.profile_id
            acct = self.db.get_or_create_prediction_account(uid)
            open_pos = self.db.get_prediction_positions(uid)

            held_events = set()
            for p in open_pos:
                held_events.add(p["event_ticker"])
                mkt = prediction_feed.fetch_market(p["market_ticker"])
                if not mkt:
                    continue
                if mkt.status == "settled" and mkt.result in ("yes", "no"):
                    self.execute_paper_close(uid, p["id"], exit_reason="SETTLED_AUTO", settled_result=mkt.result, market_override=mkt)
                else:
                    mark = mkt.yes_bid if p["side"] == "YES" else mkt.no_bid
                    if mark <= 0:
                        mark = mkt.mid_prob if p["side"] == "YES" else (1.0 - mkt.mid_prob)
                    if mark > 0:
                        self.db.update_prediction_position_price(p["id"], round(mark, 4))

            # Optional Autonomous $5 Auto-Pilot for high-EV opportunities (>= +4.0%)
            if int(acct.get("auto_pilot", 1)) == 1:
                open_after = self.db.get_prediction_positions(uid)
                max_events = int(acct.get("max_open_events", 5))
                max_stake = float(acct.get("max_stake_usd", 5.0))
                cash_usd = float(acct.get("cash_usd", 0.0))

                if len(open_after) < max_events and cash_usd >= (max_stake + 0.50):
                    if scan_cache is None:
                        try:
                            scan_cache = PredictionAlphaScanner.scan_markets(
                                category="ALL",
                                ws_eligible_only=True,
                                force_refresh=False,
                            )
                        except Exception as exc:
                            logger.debug("Auto-pilot scan skipped: %s", exc)
                            break

                    for opp in (scan_cache.get("opportunities") or []):
                        if opp["recommended_side"] in ("YES", "NO") and opp["best_ev_pct"] >= 4.0 and opp["volume"] >= 100:
                            if opp["event_ticker"] not in held_events:
                                res = self.execute_paper_buy(
                                    user_id=uid,
                                    market_ticker=opp["market_ticker"],
                                    side=opp["recommended_side"],
                                    stake_usd=min(max_stake, opp.get("recommended_stake_usd", max_stake) or max_stake),
                                    ai_fair_prob=opp["ai_fair_prob"],
                                )
                                if res.get("status") == "SUCCESS":
                                    logger.info("[%s] Auto-Pilot placed virtual prediction trade: %s", uid, res["message"])
                                break

    def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        uid = str(user_id).lower().strip()
        acct = self.db.get_or_create_prediction_account(uid)
        open_raw = self.db.get_prediction_positions(uid)
        trades = self.db.get_prediction_trades(uid, limit=100)
        real_fills = self.db.get_real_predict_fills(uid, limit=50)
        fx = self._live_fx()

        open_views: List[Dict[str, Any]] = []
        open_val_usd = 0.0
        for p in open_raw:
            qty = float(p["contracts"])
            entry_p = float(p["entry_price_usd"])
            curr_p = float(p["current_price_usd"])
            mkt_val = qty * curr_p
            open_val_usd += mkt_val
            stake = float(p["stake_usd"])
            unr_usd = mkt_val - stake
            unr_pct = (unr_usd / max(stake, 0.01)) * 100.0
            open_views.append({
                **p,
                "market_value_usd": round(mkt_val, 2),
                "max_payout_usd": round(qty, 2),
                "unrealized_pnl_usd": round(unr_usd, 2),
                "unrealized_pnl_cad": round(unr_usd * fx, 2),
                "unrealized_pnl_pct": round(unr_pct, 1),
            })

        starting = float(acct.get("starting_usd", 500.0))
        cash = float(acct.get("cash_usd", 500.0))
        equity = cash + open_val_usd
        net_pnl = equity - starting
        net_ret = (net_pnl / max(starting, 1.0)) * 100.0

        wins = sum(1 for t in trades if int(t.get("outcome_won", 0)) == 1)
        win_rate = (wins / len(trades) * 100.0) if trades else 0.0

        brier_rows = [t for t in trades if t.get("brier_ai") is not None and t.get("brier_market") is not None]
        avg_brier_ai = round(sum(float(t["brier_ai"]) for t in brier_rows) / len(brier_rows), 4) if brier_rows else None
        avg_brier_mkt = round(sum(float(t["brier_market"]) for t in brier_rows) / len(brier_rows), 4) if brier_rows else None
        brier_skill = round((1.0 - avg_brier_ai / max(avg_brier_mkt, 1e-4)) * 100.0, 1) if (avg_brier_ai is not None and avg_brier_mkt is not None) else None

        return {
            "account": {
                "user_id": uid,
                "starting_usd": round(starting, 2),
                "cash_usd": round(cash, 2),
                "positions_value_usd": round(open_val_usd, 2),
                "total_equity_usd": round(equity, 2),
                "total_equity_cad": round(equity * fx, 2),
                "net_pnl_usd": round(net_pnl, 2),
                "net_return_pct": round(net_ret, 2),
                "auto_pilot": bool(acct.get("auto_pilot", 1)),
                "max_stake_usd": float(acct.get("max_stake_usd", 5.0)),
                "fx_usd_cad": round(fx, 4),
            },
            "calibration": {
                "total_trades": len(trades),
                "settled_count": len(brier_rows),
                "win_rate_pct": round(win_rate, 1),
                "avg_brier_ai": avg_brier_ai,
                "avg_brier_market": avg_brier_mkt,
                "brier_skill_pct": brier_skill,
            },
            "open_positions": open_views,
            "settled_trades": trades,
            "real_fills": real_fills,
        }

    def get_predict_leaderboard(self) -> List[Dict[str, Any]]:
        board: List[Dict[str, Any]] = []
        for prof in profile_manager.list_profiles():
            summ = self.get_portfolio_summary(prof.profile_id)
            acct = summ["account"]
            cal = summ["calibration"]
            board.append({
                "profile_id": prof.profile_id,
                "display_name": prof.display_name,
                "starting_usd": acct["starting_usd"],
                "total_equity_usd": acct["total_equity_usd"],
                "net_return_pct": acct["net_return_pct"],
                "open_count": len(summ["open_positions"]),
                "trades_count": cal["total_trades"],
                "win_rate_pct": cal["win_rate_pct"],
                "avg_brier_ai": cal["avg_brier_ai"],
                "real_fills_count": len(summ["real_fills"]),
            })
        board.sort(key=lambda r: r["net_return_pct"], reverse=True)
        return board


prediction_bot_engine = PredictionBotEngine()
