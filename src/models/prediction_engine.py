"""
TradeSense v8 — Prediction Market Alpha Engine, Exact Fee Math & Binary Kelly
==============================================================================
Synthesizes market-implied Kalshi/Wealthsimple Predict probabilities with:
  1. Polarity-aware GPU FinBERT macro/news log-odds shifts
  2. Cross-market quantitative anchors (ModelZoo conformal strikes for Financial
     contracts & short/long Treasury yield signals for Central Bank events)
  3. Orderbook liquidity imbalance skew from live orderbook_fp depth
  4. Exact Kalshi variable taker fee formula: ceil(100 * 0.07 * C * P * (1 - P)) / 100
  5. Micro-stake Quarter-Kelly sizing honoring Wealthsimple Predict's $1.00 min
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.finbert_engine import finbert_engine
from src.data.market_intel import market_intel
from src.data.prediction_feed import PredictionMarket, prediction_feed
from src.utils.logger import get_logger

logger = get_logger("hfp.prediction_engine")


def _logit(p: float) -> float:
    pc = min(max(float(p), 0.015), 0.985)
    return math.log(pc / (1.0 - pc))


def _sigmoid(x: float) -> float:
    xc = min(max(float(x), -10.0), 10.0)
    return 1.0 / (1.0 + math.exp(-xc))


class PredictionFeeSchedule:
    """Exact Kalshi & Wealthsimple Predict variable taker fee & payout math."""

    @staticmethod
    def calculate_taker_fee_usd(
        contracts: float,
        price_usd: float,
        multiplier: float = 1.0,
    ) -> float:
        """
        Compute exact variable taker fee rounded up to the nearest cent:
            fee = ceil(100 * M * 0.07 * C * P * (1 - P)) / 100
        """
        c = max(0.0, float(contracts))
        p = min(max(float(price_usd), 0.01), 0.99)
        if c <= 0.0:
            return 0.0
        raw_fee = float(multiplier) * 0.07 * c * p * (1.0 - p)
        return math.ceil(raw_fee * 100.0 - 1e-12) / 100.0

    @classmethod
    def estimate_trade_economics(
        cls,
        stake_usd: float,
        price_usd: float,
        fair_prob: float,
        fx_usd_cad: float = 1.39,
    ) -> Dict[str, Any]:
        """
        Calculate fractional contracts ($1 min), fee in USD & CAD, max profit,
        ROI %, breakeven probability, and Net Expected Value after fees.
        """
        stake = max(0.0, float(stake_usd))
        p = min(max(float(price_usd), 0.01), 0.99)
        q = min(max(float(fair_prob), 0.01), 0.99)
        fx = max(1.0, float(fx_usd_cad))

        contracts = stake / p if stake > 0 else 0.0
        fee_usd = cls.calculate_taker_fee_usd(contracts, p)
        total_cost_usd = stake + fee_usd
        total_cost_cad = total_cost_usd * fx

        payout_if_win_usd = contracts * 1.00
        net_profit_if_win_usd = payout_if_win_usd - total_cost_usd
        net_profit_if_win_cad = net_profit_if_win_usd * fx
        roi_if_win_pct = (net_profit_if_win_usd / total_cost_usd * 100.0) if total_cost_usd > 0 else 0.0
        breakeven_prob = (total_cost_usd / payout_if_win_usd) if payout_if_win_usd > 0 else 1.0

        net_ev_usd = (q * payout_if_win_usd) - total_cost_usd
        net_ev_pct = (net_ev_usd / total_cost_usd * 100.0) if total_cost_usd > 0 else 0.0

        return {
            "stake_usd": round(stake, 2),
            "price_usd": round(p, 4),
            "contracts": round(contracts, 4),
            "fee_usd": round(fee_usd, 2),
            "fee_cad": round(fee_usd * fx, 2),
            "total_cost_usd": round(total_cost_usd, 2),
            "total_cost_cad": round(total_cost_cad, 2),
            "payout_if_win_usd": round(payout_if_win_usd, 2),
            "net_profit_if_win_usd": round(net_profit_if_win_usd, 2),
            "net_profit_if_win_cad": round(net_profit_if_win_cad, 2),
            "roi_if_win_pct": round(roi_if_win_pct, 1),
            "breakeven_prob_pct": round(min(99.9, breakeven_prob * 100.0), 1),
            "net_ev_usd": round(net_ev_usd, 3),
            "net_ev_pct": round(net_ev_pct, 2),
        }


class EventProbabilityModel:
    """Bayesian log-odds probability engine combining FinBERT, Quant Zoo & Orderbook skew."""

    _zoo_cache: Dict[str, Tuple[float, float, float]] = {}  # sym -> (p10, p50, p90)

    @classmethod
    def _get_quant_distribution(cls, yf_symbol: str) -> Optional[Tuple[float, float, float]]:
        """Retrieve or compute conformal (p10, p50, p90) for a Financial contract underlying."""
        if yf_symbol in cls._zoo_cache:
            return cls._zoo_cache[yf_symbol]
        try:
            from src.data.data_feed import DataLake
            from src.data.ws_catalog import WealthsimpleCatalog
            from src.models.model_zoo import ModelZooEngine

            lake = DataLake()
            df = lake.fetch(yf_symbol, timeframe="1h", lookback="30d")
            if df is None or len(df) < 40:
                return None
            asset = WealthsimpleCatalog.resolve_asset(yf_symbol)
            res = ModelZooEngine.train_and_evaluate(df, asset, horizon_bars=10, force_retrain=False)
            tup = (
                float(res["consensus_p10"]),
                float(res["consensus_p50"]),
                float(res["consensus_p90"]),
            )
            cls._zoo_cache[yf_symbol] = tup
            return tup
        except Exception as exc:
            logger.debug("Quant distribution fallback for %s: %s", yf_symbol, exc)
            return None

    @classmethod
    def compute_financial_strike_anchor(cls, underlying_symbol: str, strike_text: str) -> Optional[float]:
        """
        Estimate model probability for Financial strike ranges (S&P 500, QQQ, BTC, ETH)
        using our conformal (p10, p50, p90) prediction cone.
        """
        if not underlying_symbol or not strike_text:
            return None
        dist = cls._get_quant_distribution(underlying_symbol)
        if not dist:
            return None

        p10, p50, p90 = dist
        # Conformal Gaussian scale approximation (80% interval = +/- 1.2816 sigma)
        sigma = max((p90 - p10) / 2.563, p50 * 0.005, 1e-4)

        nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", strike_text)]
        if not nums:
            return None

        # Handle unit scale mismatches (e.g. BTC-CAD vs BTC-USD, SPY vs SPX ^GSPC x10)
        adj_nums = []
        for n in nums:
            if underlying_symbol == "SPY" and n > 2000:
                adj_nums.append(n / 10.0)
            elif underlying_symbol == "QQQ" and n > 5000:
                adj_nums.append(n / 41.0)
            elif "BTC-CAD" in underlying_symbol and n < 120000 and p50 > 120000:
                adj_nums.append(n * 1.39)
            elif "ETH-CAD" in underlying_symbol and n < 4200 and p50 > 4200:
                adj_nums.append(n * 1.39)
            else:
                adj_nums.append(n)

        def _cdf(k: float) -> float:
            z = (k - p50) / sigma
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        low = strike_text.lower()
        if ("to" in low or "between" in low or "-" in low) and len(adj_nums) >= 2:
            lo_k, hi_k = min(adj_nums[0], adj_nums[1]), max(adj_nums[0], adj_nums[1])
            prob = _cdf(hi_k) - _cdf(lo_k)
        elif any(w in low for w in ["below", "under", "less than", "<"]):
            prob = _cdf(adj_nums[0])
        elif any(w in low for w in ["above", "over", "greater", "+", ">"]):
            prob = 1.0 - _cdf(adj_nums[0])
        else:
            # Bucket centered around a value vs single strike
            prob = 1.0 - _cdf(adj_nums[0])

        return float(min(max(prob, 0.02), 0.98))

    @classmethod
    def evaluate_market(
        cls,
        market: PredictionMarket,
        orderbook: Optional[Dict[str, Any]] = None,
        macro_articles: Optional[List[Any]] = None,
        quant_anchor_prob: Optional[float] = None,
        macro_pulse: Optional[Any] = None,
        fx_usd_cad: float = 1.39,
        bankroll_usd: float = 500.0,
    ) -> Dict[str, Any]:
        """
        Compute AI Fair Probability, FinBERT directional attribution, exact
        after-fee YES vs NO Net EV %, and Quarter-Kelly micro-stake sizing.
        """
        p_mkt = min(max(market.mid_prob, 0.02), 0.98)
        L0 = _logit(p_mkt)

        # 1. FinBERT Polarity-Aligned Macro News Shift
        article_texts = [
            a if isinstance(a, str) else str(a.get("title", "") if isinstance(a, dict) else getattr(a, "title", a))
            for a in (macro_articles or [])
        ]
        recency_weights = []
        now_ts = max(
            (0.0 if isinstance(a, str) else float(a.get("published_ts", 0.0) if isinstance(a, dict) else getattr(a, "published_ts", 0.0)) for a in (macro_articles or [])),
            default=0.0,
        )
        for a in (macro_articles or []):
            pts = 0.0 if isinstance(a, str) else float(a.get("published_ts", now_ts) if isinstance(a, dict) else getattr(a, "published_ts", now_ts))
            age_h = max(0.2, (now_ts - pts) / 3600.0) if (now_ts > 0 and pts > 0) else 6.0
            recency_weights.append(2.0 if age_h <= 6.0 else (1.2 if age_h <= 24.0 else 0.7))

        pol = finbert_engine.score_macro_polarity(
            article_texts,
            domain=market.domain,
            outcome_hint=f"{market.event_title} {market.outcome_label}",
            recency_weights=recency_weights if recency_weights else None,
        )
        dL_nlp = float(np.clip(0.45 * float(pol["directional_shift_score"]), -0.55, 0.55))

        # 2. Quantitative / Macro Rate Anchor Shift
        dL_quant = 0.0
        anchor_source = None
        if quant_anchor_prob is not None:
            dL_quant = float(np.clip(0.35 * (_logit(quant_anchor_prob) - L0), -0.50, 0.50))
            anchor_source = f"Quant Zoo Strike Prior ({quant_anchor_prob * 100.0:.1f}%)"
        elif market.domain in ("FED_RATES", "BOC_RATES") and macro_pulse is not None:
            vix = float(getattr(macro_pulse, "vix_level", 18.0))
            hint_l = market.outcome_label.lower()
            if vix >= 24.0 and "cut" in hint_l:
                dL_quant = 0.14
                anchor_source = f"Elevated VIX ({vix:.1f}) Dovish Tilt"
            elif vix < 15.5 and ("maintain" in hint_l or "hold" in hint_l):
                dL_quant = 0.10
                anchor_source = f"Low Vol VIX ({vix:.1f}) Hold Stability"

        # 3. Orderbook Depth Skew Shift (dampened contract-quantity imbalance)
        ob_imb = float((orderbook or {}).get("imbalance_ratio", 0.0))
        dL_flow = float(np.clip(0.08 * ob_imb, -0.10, 0.10))

        # Combine into Bayesian posterior probability
        q_yes = min(max(_sigmoid(L0 + dL_nlp + dL_quant + dL_flow), 0.01), 0.99)
        q_no = 1.0 - q_yes

        # Evaluate YES and NO economics after exact Kalshi taker fees (benchmark on $5 micro stake)
        yes_ask = market.yes_ask
        no_ask = market.no_ask

        yes_tradable = 0.03 <= yes_ask <= 0.95
        no_tradable = 0.03 <= no_ask <= 0.95

        yes_econ = PredictionFeeSchedule.estimate_trade_economics(5.0, yes_ask if yes_ask > 0 else 0.5, q_yes, fx_usd_cad)
        no_econ = PredictionFeeSchedule.estimate_trade_economics(5.0, no_ask if no_ask > 0 else 0.5, q_no, fx_usd_cad)

        yes_ev = yes_econ["net_ev_pct"] if yes_tradable else -99.0
        no_ev = no_econ["net_ev_pct"] if no_tradable else -99.0

        if yes_ev >= no_ev and yes_ev >= 1.5 and yes_tradable:
            best_side = "YES"
            best_ev_pct = yes_ev
            entry_price = yes_ask
            fair_side = q_yes
            best_econ = yes_econ
        elif no_ev > yes_ev and no_ev >= 1.5 and no_tradable:
            best_side = "NO"
            best_ev_pct = no_ev
            entry_price = no_ask
            fair_side = q_no
            best_econ = no_econ
        else:
            best_side = "PASS"
            if yes_ev >= no_ev and yes_tradable:
                best_ev_pct = yes_ev
                entry_price = yes_ask
                fair_side = q_yes
                best_econ = yes_econ
            else:
                best_ev_pct = no_ev if no_tradable else yes_ev
                entry_price = no_ask if no_tradable else yes_ask
                fair_side = q_no if no_tradable else q_yes
                best_econ = no_econ if no_tradable else yes_econ

        # Micro-Stake Quarter-Kelly Sizing ($1 min, capped at $50 or 10% of sandbox)
        kelly_stake = 0.0
        if best_side != "PASS" and 0.02 < entry_price < 0.96:
            p_eff = entry_price + PredictionFeeSchedule.calculate_taker_fee_usd(1.0, entry_price)
            kelly_f = 0.25 * max(0.0, (fair_side - p_eff) / max(1.0 - p_eff, 0.05))
            raw_stake = bankroll_usd * kelly_f
            kelly_stake = round(min(max(1.0, raw_stake), min(50.0, bankroll_usd * 0.25)), 2)

        # Build plain-English institutional reasoning bullets
        reasons: List[str] = [
            f"Market mid implies {p_mkt * 100.0:.1f}% vs AI fair estimate {q_yes * 100.0:.1f}% ({(q_yes - p_mkt) * 100.0:+.1f} pts edge).",
            f"FinBERT ({pol['backend']}): directional macro shift {pol['directional_shift_score']:+.2f} (log-odds {dL_nlp:+.2f}) across {len(article_texts)} Finnhub/RSS headlines.",
        ]
        if anchor_source:
            reasons.append(f"{anchor_source} (log-odds {dL_quant:+.2f}).")
        if abs(ob_imb) >= 0.05:
            reasons.append(f"Orderbook liquidity imbalance {ob_imb:+.2f} towards {'YES' if ob_imb > 0 else 'NO'} bids.")
        if best_side != "PASS":
            reasons.append(f"Actionable {best_side} @ {entry_price * 100.0:.1f}c yields {best_ev_pct:+.2f}% Net EV after Kalshi fee (${best_econ['fee_usd']:.2f}); Quarter-Kelly: ${kelly_stake:.2f} USD.")
        else:
            reasons.append(f"Wait / No edge: spread & taker fee absorb implied alpha (best Net EV {best_ev_pct:+.2f}%).")

        return {
            **market.to_dict(),
            "ai_fair_prob": round(q_yes, 4),
            "ai_fair_pct": round(q_yes * 100.0, 1),
            "market_prob_pct": round(p_mkt * 100.0, 1),
            "edge_pts": round((q_yes - p_mkt) * 100.0, 1),
            "recommended_side": best_side,
            "best_ev_pct": round(best_ev_pct, 2),
            "recommended_stake_usd": kelly_stake,
            "yes_net_ev_pct": round(yes_ev, 2),
            "no_net_ev_pct": round(no_ev, 2),
            "sample_economics": best_econ,
            "finbert_backend": pol["backend"],
            "finbert_shift": pol["directional_shift_score"],
            "hawkish_dovish_index": pol["hawkish_dovish_index"],
            "reasoning_bullets": reasons,
        }


class PredictionAlphaScanner:
    """Scans flagship Kalshi/WS Predict events in parallel and ranks by Net EV after fees."""

    @staticmethod
    def scan_markets(
        category: str = "ALL",
        ws_eligible_only: bool = True,
        search_q: str = "",
        force_refresh: bool = False,
        bankroll_usd: float = 500.0,
    ) -> Dict[str, Any]:
        """Scan live markets, batch-score Finnhub macro news via FinBERT, and rank top opportunities."""
        events = prediction_feed.fetch_flagship_events(
            category=category,
            ws_eligible_only=ws_eligible_only,
            search_q=search_q,
            force_refresh=force_refresh,
        )

        macro_pulse = market_intel.get_macro_pulse(force_refresh=False)
        fx_rate = float(getattr(macro_pulse, "usd_cad", 1.39) or 1.39)

        # Fetch & FinBERT-score macro topic intel once per unique domain in parallel
        unique_domains: Dict[str, List[str]] = {}
        for ev in events:
            dom = ev.get("domain", "FED_RATES")
            if dom not in unique_domains:
                unique_domains[dom] = list(ev.get("keywords") or [ev.get("title", dom)])

        domain_intel: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {
                pool.submit(market_intel.get_macro_topic_intelligence, dom, kws, force_refresh): dom
                for dom, kws in unique_domains.items()
            }
            for fut in as_completed(futs):
                dom = futs[fut]
                try:
                    domain_intel[dom] = fut.result()
                except Exception as exc:
                    logger.debug("Domain intel fallback for %s: %s", dom, exc)

        ranked_opportunities: List[Dict[str, Any]] = []
        eval_ctx_by_ticker: Dict[str, Tuple[PredictionMarket, List[Any], Optional[float], float]] = {}

        for ev in events:
            mkts: List[PredictionMarket] = ev.get("markets") or []
            intel_rep = domain_intel.get(ev.get("domain", "FED_RATES"))
            articles = intel_rep.articles if intel_rep else []

            # Score up to top 4 most-liquid active contracts per event to keep scan crisp
            tradable = [m for m in mkts if 0.03 <= m.yes_ask <= 0.95 or 0.03 <= m.no_ask <= 0.95]
            if not tradable:
                tradable = [ev["lead_market"]] if ev.get("lead_market") else []
            tradable.sort(key=lambda m: (m.volume + m.open_interest, -abs(m.mid_prob - 0.5)), reverse=True)

            for m in tradable[:4]:
                q_anchor = None
                if m.domain == "FINANCIALS" and m.underlying_yf_symbol and len(ranked_opportunities) < 20:
                    q_anchor = EventProbabilityModel.compute_financial_strike_anchor(m.underlying_yf_symbol, m.outcome_label)

                ev_res = EventProbabilityModel.evaluate_market(
                    market=m,
                    orderbook=None,
                    macro_articles=articles,
                    quant_anchor_prob=q_anchor,
                    macro_pulse=macro_pulse,
                    fx_usd_cad=fx_rate,
                    bankroll_usd=bankroll_usd,
                )
                ev_res["event_total_volume"] = ev["total_volume"]
                ranked_opportunities.append(ev_res)
                eval_ctx_by_ticker[m.market_ticker] = (m, articles, q_anchor, ev["total_volume"])

        ranked_opportunities.sort(
            key=lambda x: (1 if x["recommended_side"] != "PASS" else 0, x["best_ev_pct"], x["volume"]),
            reverse=True,
        )

        # Refine top 3 actionable candidates with live orderbook depth skew so Scanner & Ticket match 100%
        top_actionable_idx = [
            idx for idx, op in enumerate(ranked_opportunities) if op.get("recommended_side") != "PASS"
        ][:3]
        for idx in top_actionable_idx:
            tkr = ranked_opportunities[idx]["market_ticker"]
            ctx = eval_ctx_by_ticker.get(tkr)
            if not ctx:
                continue
            m_obj, arts, q_anc, ev_vol = ctx
            try:
                ob = prediction_feed.fetch_orderbook(tkr)
                refined = EventProbabilityModel.evaluate_market(
                    market=m_obj,
                    orderbook=ob,
                    macro_articles=arts,
                    quant_anchor_prob=q_anc,
                    macro_pulse=macro_pulse,
                    fx_usd_cad=fx_rate,
                    bankroll_usd=bankroll_usd,
                )
                refined["event_total_volume"] = ev_vol
                ranked_opportunities[idx] = refined
            except Exception:
                pass

        ranked_opportunities.sort(
            key=lambda x: (1 if x["recommended_side"] != "PASS" else 0, x["best_ev_pct"], x["volume"]),
            reverse=True,
        )

        return {
            "opportunities": ranked_opportunities,
            "events_scanned": len(events),
            "markets_scored": len(ranked_opportunities),
            "fx_usd_cad": round(fx_rate, 4),
            "macro_regime": getattr(macro_pulse, "macro_regime", "NEUTRAL MACRO"),
            "finbert_status": finbert_engine.get_status(),
        }
