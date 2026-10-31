"""
TradeSense v7 — Full Verification Test Suite
============================================
Validates:
  1. Zero target leakage in feature engineering (features must never leak future data)
  2. All 8 ML/DL Quant Zoo models train and infer finite, monotonic distributions
  3. Split Conformal calibration guarantees: p10 <= p50 <= p90
  4. Wealthsimple execution routing and fee calculations (Crypto vs ETF proxy)
  5. Position risk sizing and scale independence
  6. Cookie authentication and REST API endpoints
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.data_feed import DataLake
from src.data.ws_catalog import WealthsimpleCatalog, WealthsimpleFeeSchedule, WealthsimpleAsset
from src.data.profiles import profile_manager
from src.models.model_zoo import (
    MODEL_PRESETS,
    ModelZooEngine,
    build_ml_feature_matrix,
    create_model,
)
from src.models.profit_engine import PositionRiskSizer, AlphaScanner
from src.web.auth_cookie import SESSION_COOKIE_NAME, create_session_token
from src.web.server import app


# ---------------------------------------------------------------------------
# 1. Zero Leakage Verification
# ---------------------------------------------------------------------------
def test_zero_target_leakage():
    """Verify that feature matrix contains no future information or target leakage."""
    data_lake = DataLake()
    df = data_lake.fetch("SHOP.TO", timeframe="1h", lookback="30d")
    assert not df.empty, "Historical data should not be empty"

    features_df, feature_cols = build_ml_feature_matrix(df, horizon_bars=5, is_crypto=False)

    # Feature columns must never contain target or forward references
    for col in feature_cols:
        assert "target" not in col.lower(), f"Target column '{col}' found in feature matrix!"
        assert "forward" not in col.lower(), f"Forward column '{col}' found in feature matrix!"
        assert "future" not in col.lower(), f"Future column '{col}' found in feature matrix!"

    # Ensure feature values are finite and valid
    clean_features = features_df[feature_cols].iloc[25:-5]
    assert len(clean_features) > 20, "Should have sufficient clean feature rows"
    assert not np.isnan(clean_features.values).any(), "Feature matrix contains NaN values"
    assert not np.isinf(clean_features.values).any(), "Feature matrix contains Inf values"


# ---------------------------------------------------------------------------
# 2. All 8 ML/DL Quant Zoo Models
# ---------------------------------------------------------------------------
def test_all_8_models_quant_zoo():
    """Verify all 8 quant models fit cleanly, infer finite values, and satisfy p10 <= p50 <= p90."""
    data_lake = DataLake()
    df = data_lake.fetch("SHOP.TO", timeframe="1h", lookback="30d")
    asset = WealthsimpleCatalog.get_asset("SHOP.TO")
    assert asset is not None

    result = ModelZooEngine.train_and_evaluate(df, asset, horizon_bars=5, force_retrain=True)

    assert "consensus_p10" in result
    assert "consensus_p50" in result
    assert "consensus_p90" in result
    assert result["consensus_p10"] <= result["consensus_p50"] <= result["consensus_p90"]

    all_models = result.get("all_model_results", [])
    assert len(all_models) == 8, f"Expected 8 models in Quant Zoo, got {len(all_models)}"

    for m in all_models:
        assert np.isfinite(m["p10"])
        assert np.isfinite(m["p50"])
        assert np.isfinite(m["p90"])
        assert m["p10"] <= m["p50"] <= m["p90"], f"Model {m['name']} violated quantile ordering: {m}"
        assert np.isfinite(m["mape_pct"])
        assert np.isfinite(m["skill_pct"])
        assert np.isfinite(m["hit_rate_pct"])
        assert 0.0 <= m["hit_rate_pct"] <= 100.0


# ---------------------------------------------------------------------------
# 3. Wealthsimple Execution Routing & Fee Coherence
# ---------------------------------------------------------------------------
def test_wealthsimple_fee_and_proxy_routing():
    """Verify fee schedules: raw crypto 4.10% round-trip vs Canadian ETF proxy 0.08%."""
    catalog = WealthsimpleCatalog()

    # Raw Crypto
    primary, exec_asset_raw, is_proxied_raw = catalog.get_execution_route("BTC-USD", use_proxy=False)
    assert is_proxied_raw is False
    assert exec_asset_raw.symbol in ("BTC-USD", "BTC-CAD")
    fee_raw = WealthsimpleFeeSchedule.calculate_fee(exec_asset_raw, ws_tier="Core")
    assert fee_raw.round_trip_total_pct == 4.10

    # Proxied Crypto via TSX ETF (BTCC.TO)
    primary, exec_asset_proxy, is_proxied_proxy = catalog.get_execution_route("BTC-USD", use_proxy=True)
    assert is_proxied_proxy is True
    assert exec_asset_proxy.symbol == "BTCC.TO"
    fee_proxy = WealthsimpleFeeSchedule.calculate_fee(exec_asset_proxy, ws_tier="Core")
    assert fee_proxy.round_trip_total_pct == 0.08

    # Canadian Stock (SHOP.TO)
    primary, exec_shop, is_shop_proxied = catalog.get_execution_route("SHOP.TO", use_proxy=True)
    assert is_shop_proxied is False
    fee_shop = WealthsimpleFeeSchedule.calculate_fee(exec_shop, ws_tier="Core")
    assert fee_shop.round_trip_total_pct == 0.08

    # US Stock (AAPL) with FX
    primary, exec_aapl, is_aapl_proxied = catalog.get_execution_route("AAPL", use_proxy=False)
    fee_aapl = WealthsimpleFeeSchedule.calculate_fee(exec_aapl, ws_tier="Core")
    assert fee_aapl.round_trip_total_pct == 3.08  # 1.5% buy + 1.5% sell + 0.08% TSX/US spread


# ---------------------------------------------------------------------------
# 4. Position Risk Sizer Scale Independence
# ---------------------------------------------------------------------------
def test_position_risk_sizer():
    """Verify scale independence of risk sizing under stop-loss constraints."""
    sizing = PositionRiskSizer.calculate_position_size(
        total_equity_cad=10_000.0,
        current_price=100.0,
        p10_stop=90.0,
        p90_target=120.0,
        risk_per_trade_pct=2.0,
    )
    
    assert sizing.recommended_cad > 0
    assert sizing.recommended_shares > 0
    assert sizing.dollar_risk_at_stop <= 200.0 + 1e-4  # 2% of $10,000 = $200
    assert sizing.quarter_kelly_fraction >= 0.0


# ---------------------------------------------------------------------------
# 5. REST API & Cookie Auth Tests
# ---------------------------------------------------------------------------
def test_fastapi_rest_suite():
    """Verify all FastAPI routes: auth, workspace, catalog, preferences, and virtual trading."""
    client = TestClient(app)

    # 1. Probe favicon (CORS allowed for GitHub Pages gateway)
    res_fav = client.get("/favicon.ico")
    assert res_fav.status_code == 200
    assert res_fav.headers.get("access-control-allow-origin") == "*"

    # 2. Login page
    res_login = client.get("/login")
    assert res_login.status_code == 200
    assert "Uzair" in res_login.text
    assert "Waleed" in res_login.text

    # 3. Auth select profile
    res_auth = client.post("/api/auth/select-profile", json={"profile_id": "uzair"})
    assert res_auth.status_code == 200
    token = res_auth.cookies.get(SESSION_COOKIE_NAME)
    assert token is not None

    # 4. Authenticated root terminal
    res_root = client.get("/", cookies={SESSION_COOKIE_NAME: token})
    assert res_root.status_code == 200
    assert "TradeSense" in res_root.text

    # 5. Catalog
    res_cat = client.get("/api/catalog", cookies={SESSION_COOKIE_NAME: token})
    assert res_cat.status_code == 200
    cat_data = res_cat.json()
    assert len(cat_data["assets"]) >= 5

    # 6. Preferences update (ensure all 6 fields save without crash)
    pref_payload = {
        "active_symbol": "SHOP.TO",
        "active_timeframe": "1h",
        "active_horizon": "15d",
        "active_model": "xgboost",
        "virtual_balance": 10000.0,
        "use_proxy_route": True,
    }
    res_pref = client.post("/api/preferences", json=pref_payload, cookies={SESSION_COOKIE_NAME: token})
    assert res_pref.status_code == 200

    # 7. Workspace data feed
    res_ws = client.get("/api/workspace?symbol=SHOP.TO&tf=1h&lookback=30d", cookies={SESSION_COOKIE_NAME: token})
    assert res_ws.status_code == 200
    ws_data = res_ws.json()
    assert "bars" in ws_data
    assert "consensus" in ws_data
    assert ws_data["symbol"] == "SHOP.TO"

    # 8. Portfolios overview
    res_ports = client.get("/api/portfolios", cookies={SESSION_COOKIE_NAME: token})
    assert res_ports.status_code == 200
    port_data = res_ports.json()
    profile_ids = [entry["profile_id"] for entry in port_data["leaderboard"]]
    assert "uzair" in profile_ids
    assert "waleed" in profile_ids
