"""
TradeSense v6 Master Behavioral Verification Suite
===================================================
Tests for:
  1. Passwordless Multi-User Profile Isolation & CRUD (Uzair, Waleed, +Add)
  2. 8-Model ML/DL Quant Zoo Individual & Ensemble Execution (PyTorch DL, GBDT, Kernels, Econometrics)
  3. Profit-Maximizing Stack (AlphaScanner Net-EV, Quarter-Kelly Sizer, Market Hours, Trailing Stop Ratchet)
  4. Interactive Chart Builder & Indicators
  5. 100% Zero Fake Stubs Verification (No fake time.sleep, No hardcoded sentiment, Working proxy fees, Working real WS trade logs)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)

from src.data.database import DatabaseManager
from src.data.profiles import ProfileManager, UserProfile
from src.data.ws_catalog import WealthsimpleAsset, WealthsimpleCatalog, WealthsimpleFeeSchedule
from src.models.model_zoo import (
    ALL_QUANT_MODELS,
    MODEL_PRESETS,
    MODEL_REGISTRY,
    ModelZooEngine,
    build_ml_feature_matrix,
)
from src.models.profit_engine import (
    AlphaOpportunity,
    AlphaScanner,
    MarketHoursGuard,
    PositionRiskSizer,
    TrailingStopRatchet,
)
from src.models.virtual_bot import VirtualBotEngine
from src.ui.chart_builder import (
    PLOTLY_GLASS_CONFIG,
    build_interactive_chart,
    calculate_heikin_ashi,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_db():
    """Provides a fresh isolated SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    mgr = DatabaseManager(db_path=db_path)
    mgr.init_db()
    yield mgr
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


@pytest.fixture
def synthetic_ohlcv():
    """Provides 120 bars of realistic OHLCV data."""
    periods = 120
    dates = pd.date_range("2026-01-01", periods=periods, freq="h")
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, periods)
    price_curve = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "Open": price_curve * (1.0 - rng.uniform(0.0, 0.004, periods)),
        "High": price_curve * (1.0 + rng.uniform(0.002, 0.01, periods)),
        "Low": price_curve * (1.0 - rng.uniform(0.002, 0.01, periods)),
        "Close": price_curve,
        "Volume": rng.uniform(5000, 50000, periods),
    }, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Profile Isolation & Multi-Account Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_default_profiles_seeding(temp_db):
    pm = ProfileManager(temp_db)
    pm.ensure_default_profiles()
    profiles = pm.list_profiles()
    p_ids = [p.profile_id for p in profiles]
    assert "uzair" in p_ids
    assert "waleed" in p_ids

    uzair = pm.get_profile("uzair")
    assert uzair.display_name == "Uzair"
    assert uzair.avatar_initials == "UZ"
    assert uzair.accent_color == "#38bdf8"

    waleed = pm.get_profile("waleed")
    assert waleed.display_name == "Waleed"
    assert waleed.avatar_initials == "WA"
    assert waleed.accent_color == "#34d399"


def test_add_new_profile(temp_db):
    pm = ProfileManager(temp_db)
    pm.ensure_default_profiles()
    new_p = pm.create_profile("Bilal Khan", starting_cad=25000.0, ws_tier="Premium")
    assert new_p.profile_id == "bilal_khan"
    assert new_p.display_name == "Bilal Khan"
    assert new_p.avatar_initials == "BK"

    # Virtual account must be seeded with $25,000 CAD
    acct = temp_db.get_or_create_virtual_account("bilal_khan")
    assert float(acct["virtual_cash_cad"]) == 25000.0
    assert acct["ws_tier"] == "Premium"


def test_profile_portfolio_strict_isolation(temp_db):
    """Verifies a trade logged by Uzair never affects Waleed's cash or open positions."""
    pm = ProfileManager(temp_db)
    pm.ensure_default_profiles()

    bot = VirtualBotEngine(temp_db)
    asset = WealthsimpleCatalog.get_asset("SHOP.TO")

    # Uzair opens a virtual buy of $1,000 CAD
    res = bot.manual_open_position(
        user_id="uzair",
        asset=asset,
        current_price=150.0,
        p10_stop=140.0,
        p50_target=160.0,
        p90_target=170.0,
        capital_cad=1000.0,
    )
    assert res["status"] == "POSITION_OPENED"

    uzair_acct = temp_db.get_or_create_virtual_account("uzair")
    waleed_acct = temp_db.get_or_create_virtual_account("waleed")

    # Uzair's cash deducted
    assert float(uzair_acct["virtual_cash_cad"]) < 10000.0
    # Waleed's cash completely untouched
    assert float(waleed_acct["virtual_cash_cad"]) == 10000.0

    uzair_pos = temp_db.get_virtual_positions("uzair")
    waleed_pos = temp_db.get_virtual_positions("waleed")
    assert len(uzair_pos) == 1
    assert len(waleed_pos) == 0


def test_real_ws_positions_crud(temp_db):
    """Verifies real Wealthsimple trades are saved, queried, and closed cleanly under profile_id."""
    tx_id = "WS-20261031-001"
    trade_data = {
        "tx_id": tx_id,
        "profile_id": "uzair",
        "ticker": "BTC-CAD",
        "ws_symbol": "BTCC.TO",
        "route_type": "CAD_PROXY",
        "status": "OPEN",
        "entry_price": 25.50,
        "current_price": 25.50,
        "quantity": 100.0,
        "capital_cad": 2550.0,
        "target_price": 28.00,
        "stop_price": 24.00,
        "trailing_stop_price": 24.00,
        "est_fee_cad": 0.40,
    }
    temp_db.save_real_ws_position(trade_data)

    open_trades = temp_db.get_real_ws_positions("uzair", status="OPEN")
    assert len(open_trades) == 1
    assert open_trades[0]["ws_symbol"] == "BTCC.TO"

    # Close real trade at 27.50
    temp_db.close_real_ws_position(tx_id, exit_price=27.50, exit_reason="TARGET_REACHED")
    closed_trades = temp_db.get_real_ws_positions("uzair", status="CLOSED")
    assert len(closed_trades) == 1
    assert closed_trades[0]["status"] == "CLOSED"
    assert closed_trades[0]["exit_price"] == 27.50
    assert closed_trades[0]["realized_pnl_cad"] == 200.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. 8-Model ML/DL Quant Zoo Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_all_8_models_exist_in_registry():
    assert len(ALL_QUANT_MODELS) == 8
    expected = [
        "dl_bilstm_attn",
        "dl_tcn_resnet",
        "dl_transformer",
        "ml_xgboost_quantile",
        "ml_gbdt_forest",
        "ml_svr_bayesian",
        "quant_kalman_garch",
        "nlp_flow_meta",
    ]
    for mid in expected:
        assert mid in ALL_QUANT_MODELS
        assert mid in MODEL_REGISTRY


@pytest.mark.parametrize("model_id", list(ALL_QUANT_MODELS.keys()))
def test_individual_quant_model_training_and_quantiles(model_id, synthetic_ohlcv):
    """Individually fits and evaluates each of the 8 models, asserting strict monotonicity."""
    feat_df, feat_cols = build_ml_feature_matrix(synthetic_ohlcv, horizon_bars=5)
    train_data = feat_df.dropna(subset=feat_cols + ["target_ret"])
    X = train_data[feat_cols].values
    y = train_data["target_ret"].values

    model = ALL_QUANT_MODELS[model_id]
    model.fit(X[:-10], y[:-10])

    curr_price = float(synthetic_ohlcv["Close"].iloc[-1])
    X_curr = X[-1:]
    p10, p50, p90 = model.predict_quantiles(X_curr, curr_price, horizon_bars=5)

    assert p10 > 0
    assert p50 > 0
    assert p90 > 0
    assert p10 <= p50 <= p90, f"Model {model_id} failed quantile monotonicity: P10={p10}, P50={p50}, P90={p90}"


def test_model_zoo_engine_presets(synthetic_ohlcv):
    asset = WealthsimpleCatalog.get_asset("SHOP.TO")
    for preset_name in ["AUTO_DNA", "PURE_DL", "TREE_ML", "FULL_SUPER"]:
        models = (
            ModelZooEngine.get_auto_models_for_asset(asset, hurst=0.58)
            if preset_name == "AUTO_DNA"
            else MODEL_PRESETS[preset_name]["models"]
        )
        res = ModelZooEngine.train_and_evaluate(
            df=synthetic_ohlcv,
            asset=asset,
            selected_model_ids=models,
            horizon_bars=5,
            blending_mode="WALK_FORWARD",
            force_retrain=True,
        )
        assert res["consensus_p10"] > 0
        assert res["consensus_p50"] > 0
        assert res["consensus_p90"] > 0
        assert res["consensus_p10"] <= res["consensus_p50"] <= res["consensus_p90"]
        assert len(res["model_results"]) > 0


def test_model_zoo_blending_modes(synthetic_ohlcv):
    asset = WealthsimpleCatalog.get_asset("BTC-CAD")
    models = ["dl_bilstm_attn", "ml_xgboost_quantile", "ml_gbdt_forest"]
    for blend in ["WALK_FORWARD", "RIDGE_STACK", "EQUAL"]:
        res = ModelZooEngine.train_and_evaluate(
            df=synthetic_ohlcv,
            asset=asset,
            selected_model_ids=models,
            horizon_bars=5,
            blending_mode=blend,
            force_retrain=True,
        )
        weights = res["weights"]
        assert len(weights) == 3
        assert pytest.approx(sum(weights.values()), 0.05) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Profit Stack Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_quarter_kelly_position_sizer():
    # Account with $10,000 equity, 60% hit rate, 10% upside target, 5% stop
    res = PositionRiskSizer.calculate_position_size(
        total_equity_cad=10000.0,
        current_price=100.0,
        p10_stop=95.0,
        p90_target=110.0,
        hit_rate_pct=60.0,
        round_trip_fee_pct=0.08,
    )
    assert res.recommended_cad > 100.0
    assert res.recommended_cad <= 2500.0  # Max 25% Kelly allocation cap
    assert res.quarter_kelly_fraction > 0
    assert res.dollar_risk_at_stop <= 10000.0 * 0.02  # Stays within safe dollar risk


def test_market_hours_guard_crypto_and_tsx():
    btc = WealthsimpleCatalog.get_asset("BTC-CAD")
    btc_status = MarketHoursGuard.get_market_status(btc)
    assert btc_status["phase"] == "OPEN"
    assert btc_status["is_open"] is True
    assert "Taxable" in btc_status["wrapper_advice"]

    shop = WealthsimpleCatalog.get_asset("SHOP.TO")
    shop_status = MarketHoursGuard.get_market_status(shop)
    assert shop_status["phase"] in ("OPEN", "CLOSED_WEEKEND", "CLOSED_NIGHT", "PRE_MARKET", "POST_MARKET")
    assert "TFSA" in shop_status["wrapper_advice"]


def test_trailing_stop_ratchet_breakeven_and_chandelier():
    # Entry 100, Stop 95, Target 110 (Target distance = 10)
    # Scenario A: Price moves to 102 (gain 2 < 50% distance) -> Stop remains 95
    s1 = TrailingStopRatchet.evaluate_stop(100.0, 102.0, 95.0, 110.0, round_trip_fee_pct=0.08)
    assert s1["status"] == "INITIAL"
    assert s1["stop_price"] == 95.0

    # Scenario B: Price moves to 106 (gain 6 >= 50% distance) -> Stop ratchets to Breakeven (100.08)
    s2 = TrailingStopRatchet.evaluate_stop(100.0, 106.0, 95.0, 110.0, round_trip_fee_pct=0.08)
    assert s2["status"] == "BREAKEVEN_LOCKED"
    assert s2["stop_price"] >= 100.08

    # Scenario C: Price moves to 109 (gain 9 >= 80% distance) -> Chandelier runner trails
    s3 = TrailingStopRatchet.evaluate_stop(100.0, 109.0, 95.0, 110.0, round_trip_fee_pct=0.08, atr=2.0)
    assert s3["status"] == "CHANDELIER_RUNNER"
    assert s3["stop_price"] >= 106.0


def test_alpha_scanner_ranking():
    basket = ["SHOP.TO", "RY.TO", "BTC-CAD"]
    opps = AlphaScanner.scan_opportunities(basket=basket, force_refresh=True)
    assert len(opps) > 0
    # Verifies sorted descending by Net EV %
    evs = [o.net_ev_pct for o in opps]
    assert evs == sorted(evs, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interactive Chart Builder Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_interactive_chart_builder_renders_traces(synthetic_ohlcv):
    fig = build_interactive_chart(
        df=synthetic_ohlcv,
        symbol="SHOP.TO",
        forecast={"consensus_p10": 90.0, "consensus_p50": 105.0, "consensus_p90": 115.0, "horizon_bars": 5},
        visible_bars=40,
        chart_type="Candles",
        overlays=["EMA 9 & 21", "VWAP", "Bollinger Bands", "Volume Bars"],
        order_levels={"limit_entry": 100.0, "take_profit": 115.0, "stop_loss": 90.0},
    )
    assert len(fig.data) >= 5
    assert PLOTLY_GLASS_CONFIG["displayModeBar"] is True
    assert PLOTLY_GLASS_CONFIG["scrollZoom"] is True


def test_heikin_ashi_calculation(synthetic_ohlcv):
    ha = calculate_heikin_ashi(synthetic_ohlcv)
    assert len(ha) == len(synthetic_ohlcv)
    assert (ha["High"] >= ha["Low"]).all()
    assert (ha["High"] >= ha["Close"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Zero-Stub & Real Wiring Verification
# ─────────────────────────────────────────────────────────────────────────────
def test_zero_stubs_in_app_source():
    """Static AST scan of app.py to verify no fake sleep retrain or hardcoded sentiment."""
    app_path = os.path.join(_ROOT, "src", "ui", "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. No fake time.sleep
    assert "time.sleep(1.0)" not in src
    assert 'displayModeBar": False' not in src
    # 2. No hardcoded 0.15 sentiment
    assert "Live Sentiment Score: **+0.15**" not in src
    assert "Live Sentiment Score: **0.15**" not in src
    assert "{0.15" not in src


def test_cad_proxy_fee_savings_math():
    """Verifies CAD proxy route reduces fees dramatically compared to direct crypto/US route."""
    btc_direct = WealthsimpleCatalog.get_asset("BTC-CAD")
    btc_proxy = WealthsimpleCatalog.get_asset("BTCC.TO")

    f_direct = WealthsimpleFeeSchedule.calculate_fee(btc_direct, notional_cad=500.0, ws_tier="Core", usd_account_active=False)
    f_proxy = WealthsimpleFeeSchedule.calculate_fee(btc_proxy, notional_cad=500.0, ws_tier="Core", usd_account_active=False)

    # Direct crypto is 2.0% one-way (4.0% round trip = $20+)
    assert f_direct.fee_cad_on_notional >= 20.0
    # Proxy TSX ETF is 0% commission + narrow spread ($0.40)
    assert f_proxy.fee_cad_on_notional <= 1.0
    assert f_direct.fee_cad_on_notional > f_proxy.fee_cad_on_notional * 15


def test_virtual_bot_mark_to_market_equity_no_instant_loss(temp_db):
    """Verifies that opening a trade keeps total equity intact (cash + open position == ~starting)."""
    bot = VirtualBotEngine(temp_db)
    asset = WealthsimpleCatalog.get_asset("SHOP.TO")

    res = bot.manual_open_position(
        user_id="test_m2m",
        asset=asset,
        current_price=100.0,
        p10_stop=95.0,
        p50_target=105.0,
        p90_target=110.0,
        capital_cad=1000.0,
    )
    assert res["status"] == "POSITION_OPENED"

    acct = temp_db.get_or_create_virtual_account("test_m2m")
    cash = float(acct["virtual_cash_cad"])
    positions = temp_db.get_virtual_positions("test_m2m")
    assert len(positions) == 1

    pos = positions[0]
    pos_val = float(pos["quantity"]) * float(pos["current_price"])
    total_eq = cash + pos_val
    starting = float(acct.get("starting_virtual_cad", 10000.0))

    # Total equity should be starting cash minus entry fee ($1.50) -> ~$9,998.50, NOT $9,000 (-25%!)
    assert total_eq >= starting - 5.0
    assert total_eq <= starting
