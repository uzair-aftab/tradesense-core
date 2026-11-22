"""
TradeSense v8 — Prediction Markets, GPU FinBERT & Finnhub Verification Suite
=============================================================================
Covers:
  1. GPU FinBERT tri-class probability normalization, SHA-256 caching & macro polarity
  2. Finnhub free-tier news & earnings client (zero 403 /calendar/economic calls)
  3. Kalshi v2 fixed-point (*_dollars / *_fp) parser & Canadian CIRO filter
  4. Exact Kalshi fee formula ceil(100 * 0.07 * C * P * (1-P))/100 & binary Kelly
  5. Isolated $500 USD Predict Sandbox lifecycle: $1 min, early exit, $1/$0 settle, Brier Score
  6. FastAPI /api/predictions/* REST suite
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.database import DatabaseManager
from src.data.finbert_engine import finbert_engine
from src.data.finnhub_client import finnhub_client
from src.data.prediction_feed import PredictionMarket, parse_kalshi_market, prediction_feed
from src.models.prediction_bot import PredictionBotEngine
from src.models.prediction_engine import EventProbabilityModel, PredictionFeeSchedule
from src.web.auth_cookie import SESSION_COOKIE_NAME
from src.web.server import app


def test_finbert_engine_probs_and_macro_polarity():
    """Verify FinBERT tri-class probs, hash cache hits, and Fed hawkish/dovish polarity."""
    headlines = [
        "NVIDIA beats Q3 revenue expectations on strong data center AI demand",
        "Consumer prices surged unexpectedly keeping Federal Reserve policy restrictive",
    ]
    res1 = finbert_engine.score_texts(headlines)
    assert len(res1) == 2
    for r in res1:
        assert 0.0 <= r.positive <= 1.0
        assert 0.0 <= r.negative <= 1.0
        assert 0.0 <= r.neutral <= 1.0
        assert (r.positive + r.negative + r.neutral) == pytest.approx(1.0, abs=1e-3)
        assert -1.0 <= r.net_score <= 1.0

    # Cache check: second call must retain cache entries and return identical net scores
    cache_before = finbert_engine.get_status()["cache_size"]
    assert cache_before >= 2
    res2 = finbert_engine.score_texts(headlines)
    assert res1[0].net_score == pytest.approx(res2[0].net_score, abs=1e-6)

    # Polarity inversion check: hot inflation + hawkish Fed pushes Hike UP and Cut DOWN
    macro_txt = ["Hot CPI inflation keeps Fed hawkish and delays rate cuts"]
    pol_hike = finbert_engine.score_macro_polarity(macro_txt, domain="FED_RATES", outcome_hint="Hike 25bps")
    pol_cut = finbert_engine.score_macro_polarity(macro_txt, domain="FED_RATES", outcome_hint="Cut 25bps")
    assert pol_hike["directional_shift_score"] > 0.0, f"Expected positive shift for Hike: {pol_hike}"
    assert pol_cut["directional_shift_score"] < 0.0, f"Expected negative shift for Cut: {pol_cut}"


def test_finnhub_client_free_endpoints():
    """Verify Finnhub news & earnings helpers work cleanly and never touch paid endpoints."""
    if not finnhub_client.is_configured():
        pytest.skip("FINNHUB_API_KEY not configured in environment")

    news = finnhub_client.get_market_news("general", max_items=5)
    assert isinstance(news, list)
    if news:
        assert "title" in news[0]
        assert "publisher" in news[0]

    comp = finnhub_client.get_company_news("SHOP.TO", days=7, max_items=5)
    assert isinstance(comp, list)


def test_kalshi_fixed_point_parsing_and_ciro_filter():
    """Verify fixed-point (*_dollars / *_fp) parsing and exclusion of non-CIRO categories."""
    sample_raw = {
        "ticker": "KXFEDDECISION-26DEC-H0",
        "event_ticker": "KXFEDDECISION-26DEC",
        "yes_sub_title": "Fed maintains rate",
        "yes_bid": None,
        "yes_ask": None,
        "last_price": None,
        "yes_bid_dollars": "0.4200",
        "yes_ask_dollars": "0.4400",
        "last_price_dollars": "0.4300",
        "previous_price_dollars": "0.4100",
        "volume_fp": "7883151.28",
        "open_interest_fp": "4283519.72",
        "open_time": "2026-09-01T12:00:00Z",
        "close_time": "2026-12-09T18:59:00Z",
        "status": "active",
    }
    mkt = parse_kalshi_market(
        sample_raw,
        event_meta={"event_ticker": "KXFEDDECISION-26DEC", "series_ticker": "KXFEDDECISION", "category": "Economics", "title": "Fed decision in Dec 2026?"},
    )
    assert mkt.yes_bid == pytest.approx(0.42, abs=1e-4)
    assert mkt.yes_ask == pytest.approx(0.44, abs=1e-4)
    assert mkt.no_ask == pytest.approx(0.58, abs=1e-4)
    assert mkt.volume == pytest.approx(7883151.28, abs=1e-2)
    assert mkt.term_days_30plus is True
    assert mkt.ws_predict_eligible is True

    # Sports or Politics must never pass WS Predict CIRO eligibility
    sports_mkt = parse_kalshi_market(
        {**sample_raw, "ticker": "KXNBA-26"},
        event_meta={"event_ticker": "KXNBA-26", "series_ticker": "KXNBA", "category": "Sports"},
    )
    assert sports_mkt.ws_predict_eligible is False


def test_kalshi_fee_formula_and_binary_kelly():
    """Verify Kalshi ceil(100*0.07*C*P*(1-P))/100 fee rounding and EV/Kelly sizing."""
    # 1 contract @ $0.50 -> 0.0175 rounded up to $0.02
    assert PredictionFeeSchedule.calculate_taker_fee_usd(1.0, 0.50) == 0.02
    # 100 contracts @ $0.50 -> exact $1.75 (no floating-point overshoot to $1.76)
    assert PredictionFeeSchedule.calculate_taker_fee_usd(100.0, 0.50) == 1.75

    # Favorable YES contract: market asks 40c, dovish cut headlines boost AI fair prob > 45%
    mkt = PredictionMarket(
        market_ticker="KXFED-TEST-CUT25",
        event_ticker="KXFED-TEST",
        series_ticker="KXFEDDECISION",
        event_title="Fed decision in Dec 2026?",
        outcome_label="Cut 25bps",
        category="Economics",
        ui_category="CENTRAL_BANKS",
        domain="FED_RATES",
        status="active",
        result="",
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.59,
        no_ask=0.61,
        last_price=0.40,
        prev_price=0.38,
        mid_prob=0.395,
        volume=50000.0,
        open_interest=25000.0,
        open_time="2026-09-01T00:00:00Z",
        close_time="2026-12-10T18:59:00Z",
        contract_term_days=100,
        days_to_close=19.0,
        term_days_30plus=True,
        ws_predict_eligible=True,
        rules_primary="Resolves YES if FOMC cuts 25bps.",
    )
    ev = EventProbabilityModel.evaluate_market(
        market=mkt,
        orderbook={"imbalance_ratio": 0.45},
        macro_articles=["Cooling inflation and weak jobs trigger unanimous dovish rate cut expectations"],
        fx_usd_cad=1.39,
        bankroll_usd=500.0,
    )
    assert ev["ai_fair_prob"] > 0.42
    assert ev["recommended_side"] == "YES"
    assert ev["best_ev_pct"] > 0.0
    assert ev["recommended_stake_usd"] >= 1.0


def test_prediction_paper_lifecycle_and_brier_score(tmp_path):
    """Test $500 sandbox buy, $1 min guard, early exit, $1/$0 auto-settlement & Brier Score."""
    iso_db = DatabaseManager(db_path=tmp_path / "predict_test.db")
    bot = PredictionBotEngine(db=iso_db)
    uid = "test_uzair"

    acct0 = iso_db.get_or_create_prediction_account(uid, starting_usd=500.0)
    assert acct0["cash_usd"] == 500.0

    mock_mkt = PredictionMarket(
        market_ticker="KXCPI-TEST-ABOVE",
        event_ticker="KXCPI-TEST",
        series_ticker="KXCPI",
        event_title="CPI in Nov 2026?",
        outcome_label="Above 0.2%",
        category="Economics",
        ui_category="MACRO_DATA",
        domain="CPI_INFLATION",
        status="active",
        result="",
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.59,
        no_ask=0.61,
        last_price=0.40,
        prev_price=0.39,
        mid_prob=0.395,
        volume=10000.0,
        open_interest=5000.0,
        open_time="2026-10-01T00:00:00Z",
        close_time="2026-12-10T13:29:00Z",
        contract_term_days=70,
        days_to_close=19.0,
        term_days_30plus=True,
        ws_predict_eligible=True,
        rules_primary="BLS CPI release.",
    )

    # 1. Below $1 minimum must fail cleanly
    bad_buy = bot.execute_paper_buy(uid, mock_mkt.market_ticker, "YES", stake_usd=0.50, market_override=mock_mkt)
    assert bad_buy["status"] == "ERROR"

    # 2. Buy $10 YES @ $0.40 -> 25 contracts, fee = ceil(100*0.07*25*0.4*0.6)/100 = $0.42
    buy1 = bot.execute_paper_buy(uid, mock_mkt.market_ticker, "YES", stake_usd=10.0, ai_fair_prob=0.65, market_override=mock_mkt)
    assert buy1["status"] == "SUCCESS"
    assert buy1["contracts"] == pytest.approx(25.0, abs=1e-4)
    assert buy1["fee_usd"] == pytest.approx(0.42, abs=1e-4)
    assert buy1["cash_usd"] == pytest.approx(489.58, abs=1e-2)

    # 3. Early exit @ 0.60 YES bid
    mkt_up = PredictionMarket(**{**mock_mkt.__dict__, "yes_bid": 0.60, "yes_ask": 0.61})
    exit1 = bot.execute_paper_close(uid, buy1["position_id"], exit_reason="EARLY_EXIT", market_override=mkt_up)
    assert exit1["status"] == "SUCCESS"
    assert exit1["net_pnl_usd"] > 3.5

    # 4. Buy $5 YES and resolve via settlement = "yes" -> $1.00/contract + Brier calculation
    buy2 = bot.execute_paper_buy(uid, mock_mkt.market_ticker, "YES", stake_usd=5.0, ai_fair_prob=0.80, market_override=mock_mkt)
    assert buy2["status"] == "SUCCESS"
    settle = bot.execute_paper_close(uid, buy2["position_id"], exit_reason="SETTLED_AUTO", settled_result="yes")
    assert settle["status"] == "SUCCESS"
    assert settle["net_pnl_usd"] > 7.0

    trades = iso_db.get_prediction_trades(uid)
    settled_rows = [t for t in trades if t["exit_reason"] == "SETTLED_AUTO"]
    assert len(settled_rows) == 1
    assert settled_rows[0]["brier_ai"] == pytest.approx(0.04, abs=1e-4)
    assert settled_rows[0]["brier_market"] == pytest.approx((0.395 - 1.0) ** 2, abs=1e-4)
    assert settled_rows[0]["brier_ai"] < settled_rows[0]["brier_market"]


def test_prediction_fastapi_endpoints():
    """Verify all /api/predictions/* REST endpoints return valid 200 payloads."""
    with TestClient(app) as client:
        res_auth = client.post("/api/auth/select-profile", json={"profile_id": "uzair"})
        assert res_auth.status_code == 200
        token = res_auth.cookies.get(SESSION_COOKIE_NAME)
        assert token is not None
        client.cookies.set(SESSION_COOKIE_NAME, token)

        r_cal = client.get("/api/predictions/macro-calendar")
        assert r_cal.status_code == 200
        cal = r_cal.json()
        assert "calendar" in cal
        assert "finbert_status" in cal

        r_mkts = client.get("/api/predictions/markets?category=CENTRAL_BANKS&ws_only=true")
        assert r_mkts.status_code == 200
        mkts = r_mkts.json()
        assert "opportunities" in mkts
        assert "fx_usd_cad" in mkts

        if mkts["opportunities"]:
            ev_tkr = mkts["opportunities"][0]["event_ticker"]
            r_ev = client.get(f"/api/predictions/event/{ev_tkr}")
            assert r_ev.status_code == 200
            ev_data = r_ev.json()
            assert "outcomes" in ev_data
            assert "orderbook" in ev_data
            assert "news" in ev_data

        r_ap = client.post("/api/predictions/auto-pilot", json={"enabled": True, "max_stake_usd": 5.0})
        assert r_ap.status_code == 200

        r_port = client.get("/api/predictions/portfolio")
        assert r_port.status_code == 200
        p_data = r_port.json()
        assert "account" in p_data
        assert "calibration" in p_data
        assert "leaderboard" in p_data
